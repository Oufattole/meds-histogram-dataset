from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from meds import LabelSchema

from .config import MEDSTorchDataConfig
from .pytorch_dataset import MEDSPytorchDataset
from .types import WindowSamplingStrategy, MEDSTorchHistogramBatch, PaddingSide


HISTOGRAM_PLACEHOLDER_TOKEN = -1
ANCHOR_TOKEN_ID = -2
GAP_TOKEN_ID = -3


class WindowAnchoringStrategy(StrEnum):
    """Strategy for anchoring time windows."""
    RANDOM_EVENT = "random_event"
    END_OF_TIMELINE = "end_of_timeline"
    SPECIFIC_EVENT = "specific_event"


class EmptyWindowMode(StrEnum):
    """Mode for handling empty windows."""
    IGNORE = "ignore"  # Ignore all-zero histograms
    SINGLE_GAP = "single_gap"  # Single all-zero gap histogram representation


class FusionMode(StrEnum):
    """Controls how histogram windows are fused with raw code sequences."""

    HISTOGRAM_ONLY = "histogram_only"
    HISTOGRAM_AND_CODE = "histogram_and_code"


@dataclass
class FusionPayload:
    """Container for fused dynamic sequence data prior to padding."""

    codes: torch.LongTensor
    time_delta_days: torch.FloatTensor
    numeric_value: torch.FloatTensor | None
    numeric_value_mask: torch.BoolTensor | None
    placeholder_positions: List[int]
    static_prefix_len: int


@dataclass
class HistogramConfig:
    """Configuration for histogram dataset.

    Args:
        window_size_days: Number of days in each time window (default: 30)
        anchoring_strategy: Strategy for anchoring time windows (default: END_OF_TIMELINE)
        empty_window_mode: How to handle empty windows (default: SINGLE_GAP)
        empty_window_token: Special token ID for empty windows (default: -1)
        vocab_size: Size of the vocabulary for histograms (default: 1000)
        padding_side: Side to pad sequences on - left or right (default: RIGHT)
        specific_event_regex: Regex used with SPECIFIC_EVENT anchoring to identify anchor events
        include_anchor_token: Whether to append an explicit empty histogram token immediately after
            the anchor window (default: False)
        histogram_max_seq_len: Maximum fused sequence length when combining histograms with raw codes
            (default: 512). The dataset will always observe the full raw timeline when building windows
            and only apply this cutoff after histograms are computed.
        max_pre_anchor_tokens: Optional limit on the number of raw code tokens to retain prior to the
            anchor window. When set, the dataset will discard any earlier tokens beyond this budget.
        max_code_seq_len: Optional limit on the total number of raw code tokens (pre- plus post-anchor)
            retained in the fused sequence. Applied after truncating the pre-anchor context.
        
    Examples:
        >>> config = HistogramConfig(
        ...     window_size_days=7,
        ...     anchoring_strategy=WindowAnchoringStrategy.RANDOM_EVENT,
        ...     empty_window_mode=EmptyWindowMode.IGNORE
        ... )
        >>> config.window_size_days
        7
        >>> config.anchoring_strategy
        <WindowAnchoringStrategy.RANDOM_EVENT: 'random_event'>
    """
    window_size_days: int = 30
    anchoring_strategy: WindowAnchoringStrategy = WindowAnchoringStrategy.END_OF_TIMELINE
    empty_window_mode: EmptyWindowMode = EmptyWindowMode.SINGLE_GAP
    empty_window_token: int = -1
    vocab_size: int = 1000
    padding_side: PaddingSide = PaddingSide.RIGHT
    specific_event_regex: Optional[str] = None
    include_anchor_token: bool = False
    specific_event_codes: Optional[List[int]] = None
    histogram_max_seq_len: int = 512
    fusion_mode: FusionMode = FusionMode.HISTOGRAM_ONLY
    max_pre_anchor_tokens: int | None = None
    max_code_seq_len: int | None = None
    max_windows: int | None = None
    window_sampling_strategy: WindowSamplingStrategy = WindowSamplingStrategy.TO_END

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid
            
        Examples:
            >>> config = HistogramConfig(window_size_days=0)
            >>> config.validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: window_size_days must be positive
            
            >>> config = HistogramConfig(vocab_size=-1)
            >>> config.validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: vocab_size must be positive

            >>> config = HistogramConfig(
            ...     anchoring_strategy=WindowAnchoringStrategy.SPECIFIC_EVENT
            ... )
            >>> config.validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: specific_event_regex must be provided when using SPECIFIC_EVENT anchoring
        """
        if self.window_size_days <= 0:
            raise ValueError("window_size_days must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not isinstance(self.anchoring_strategy, WindowAnchoringStrategy):
            raise ValueError(f"Invalid anchoring_strategy: {self.anchoring_strategy}")
        if not isinstance(self.empty_window_mode, EmptyWindowMode):
            raise ValueError(f"Invalid empty_window_mode: {self.empty_window_mode}")
        if not isinstance(self.include_anchor_token, bool):
            raise ValueError("include_anchor_token must be a boolean")
        if self.anchoring_strategy == WindowAnchoringStrategy.SPECIFIC_EVENT and not (self.specific_event_codes or self.specific_event_regex):
            raise ValueError("specific_event_regex must be provided when using SPECIFIC_EVENT anchoring")
        if self.histogram_max_seq_len <= 0:
            raise ValueError("histogram_max_seq_len must be positive")
        if self.max_pre_anchor_tokens is not None and self.max_pre_anchor_tokens <= 0:
            raise ValueError("max_pre_anchor_tokens must be positive when provided")
        if self.max_code_seq_len is not None and self.max_code_seq_len <= 0:
            raise ValueError("max_code_seq_len must be positive when provided")
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError(f"max_windows must be positive when provided, but was: {self.max_windows}")


def build_fusion_payload(
    item: Dict,
    histogram_max_seq_len: int | None,
) -> FusionPayload:
    """Assemble a fused histogram/code sequence for a single sample.

    Args:
        item: Sample dictionary containing `_raw_sequence` tensors and the decoded `windows`.
        histogram_max_seq_len: Optional maximum length for the fused output sequence.

    Examples:
        >>> import torch
        >>> simple_item = {
        ...     "_raw_sequence": {
        ...         "code": torch.tensor([10, 11, 12], dtype=torch.long),
        ...         "time_delta_days": torch.zeros(3),
        ...         "numeric_value": torch.zeros(3),
        ...         "numeric_value_mask": torch.zeros(3, dtype=torch.bool),
        ...         "length": 3,
        ...         "static_prefix_len": 0,
        ...     },
        ...     "windows": [
        ...         {"codes": [0, 1]},
        ...         {"codes": [2], "is_anchor": True, "is_anchor_token": False},
        ...         {"codes": [], "is_anchor": False, "is_anchor_token": True},
        ...     ],
        ... }
        >>> payload = build_fusion_payload(simple_item, histogram_max_seq_len=None)
        >>> payload.codes.tolist()
        [-1, 10, 11, -1, 12, -2]
        >>> payload.placeholder_positions
        [0, 3]

        >>> gap_anchor_item = {
        ...     "_raw_sequence": {
        ...         "code": torch.tensor([300, 301, 302], dtype=torch.long),
        ...         "time_delta_days": torch.zeros(3),
        ...         "numeric_value": torch.zeros(3),
        ...         "numeric_value_mask": torch.zeros(3, dtype=torch.bool),
        ...         "length": 3,
        ...         "static_prefix_len": 0,
        ...     },
        ...     "windows": [
        ...         {"codes": [0, 1]},
        ...         {"codes": [], "is_gap": True},
        ...         {"codes": [2], "is_anchor": True, "is_anchor_token": False},
        ...         {"codes": [], "is_anchor": False, "is_anchor_token": True},
        ...     ],
        ... }
        >>> build_fusion_payload(gap_anchor_item, histogram_max_seq_len=None).codes.tolist()
        [-1, 300, 301, -3, -1, 302, -2]
        >>> # Test histogram_max_seq_len truncation
        >>> build_fusion_payload(gap_anchor_item, histogram_max_seq_len=5).codes.tolist()
        [-1, -3, -1, 302, -2]
        >>> build_fusion_payload(gap_anchor_item, histogram_max_seq_len=4).codes.tolist()
        [-1, -3, -1, -2]
        >>> build_fusion_payload(gap_anchor_item, histogram_max_seq_len=6).codes.tolist()
        [-1, 301, -3, -1, 302, -2]
    """

    raw_sequence = item.get("_raw_sequence")
    if raw_sequence is None:
        raise ValueError("Fusion mode requires raw sequence data per sample.")

    raw_code: torch.LongTensor = raw_sequence["code"]
    raw_time: torch.FloatTensor = raw_sequence["time_delta_days"]
    raw_numeric: torch.FloatTensor = raw_sequence["numeric_value"]
    raw_mask: torch.BoolTensor = raw_sequence["numeric_value_mask"]
    valid_length: int = int(raw_sequence["length"])
    static_prefix_len: int = int(raw_sequence["static_prefix_len"])

    codes_out: List[int] = []
    time_out: List[float] = []
    numeric_out: List[float] = []
    mask_out: List[bool] = []
    entry_types: List[str] = []
    placeholder_positions: List[int] = []
    hist_window_indices: List[int] = []

    for idx in range(min(static_prefix_len, valid_length)):
        codes_out.append(int(raw_code[idx].item()))
        time_out.append(float(raw_time[idx].item()))
        numeric_out.append(float(raw_numeric[idx].item()))
        mask_out.append(bool(raw_mask[idx].item()))
        entry_types.append("static")

    code_cursor = min(static_prefix_len, valid_length)

    for window_idx, window in enumerate(item["windows"]):
        has_gap = bool(window.get("is_gap", False))
        has_anchor_token = bool(window.get("is_anchor_token", False))
        codes_in_window = window.get("codes", []) or []
        window_code_count = len(codes_in_window)

        anchor_only_window = has_anchor_token and window_code_count == 0 and not has_gap

        if anchor_only_window:
            window["first_code_index"] = -1

            codes_out.append(ANCHOR_TOKEN_ID)
            time_out.append(0.0)
            numeric_out.append(0.0)
            mask_out.append(False)
            entry_types.append("anchor")
            continue

        placeholder_index = len(codes_out)
        placeholder_positions.append(placeholder_index)
        hist_window_indices.append(window_idx)
        window["first_code_index"] = placeholder_index

        placeholder_token = GAP_TOKEN_ID if has_gap else HISTOGRAM_PLACEHOLDER_TOKEN

        codes_out.append(placeholder_token)
        time_out.append(0.0)
        numeric_out.append(0.0)
        mask_out.append(False)
        entry_types.append("hist")

        for _ in range(window_code_count):
            if code_cursor >= valid_length:
                break
            codes_out.append(int(raw_code[code_cursor].item()))
            time_out.append(float(raw_time[code_cursor].item()))
            numeric_out.append(float(raw_numeric[code_cursor].item()))
            mask_out.append(bool(raw_mask[code_cursor].item()))
            entry_types.append("code")
            code_cursor += 1

        if has_anchor_token:
            codes_out.append(ANCHOR_TOKEN_ID)
            time_out.append(0.0)
            numeric_out.append(0.0)
            mask_out.append(False)
            entry_types.append("anchor")

        if has_gap and window_code_count > 0:
            codes_out.append(GAP_TOKEN_ID)
            time_out.append(0.0)
            numeric_out.append(0.0)
            mask_out.append(False)
            entry_types.append("gap")

    while code_cursor < valid_length:
        codes_out.append(int(raw_code[code_cursor].item()))
        time_out.append(float(raw_time[code_cursor].item()))
        numeric_out.append(float(raw_numeric[code_cursor].item()))
        mask_out.append(bool(raw_mask[code_cursor].item()))
        entry_types.append("code")
        code_cursor += 1

    max_seq_len = histogram_max_seq_len
    if max_seq_len is not None:
        non_code_tokens = sum(1 for t in entry_types if t != "code")
        if max_seq_len < non_code_tokens:
            raise ValueError(
                f"Unable to satisfy max_seq_len={max_seq_len}. Need at least {non_code_tokens} tokens to represent "
                "histogram placeholders and special tokens without any codes."
            )

        removal_budget = len(codes_out) - max_seq_len
        while removal_budget > 0:
            try:
                idx = entry_types.index("code")
            except ValueError as exc:
                raise ValueError("Unable to reduce fused sequence to requested max_seq_len; not enough code tokens.") from exc

            codes_out.pop(idx)
            time_out.pop(idx)
            numeric_out.pop(idx)
            mask_out.pop(idx)
            entry_types.pop(idx)
            removal_budget -= 1

    placeholder_positions = [
        idx for idx, entry_type in enumerate(entry_types) if entry_type == "hist"
    ]

    if len(placeholder_positions) != len(hist_window_indices):
        raise ValueError("Mismatch between histogram windows and placeholder positions in fusion payload.")

    for win_idx, new_pos in zip(hist_window_indices, placeholder_positions):
        item["windows"][win_idx]["first_code_index"] = new_pos

    code_tensor = torch.tensor(codes_out, dtype=torch.long)
    time_tensor = torch.tensor(time_out, dtype=torch.float32)
    numeric_tensor = torch.tensor(numeric_out, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_out, dtype=torch.bool)

    return FusionPayload(
        codes=code_tensor,
        time_delta_days=time_tensor,
        numeric_value=numeric_tensor,
        numeric_value_mask=mask_tensor,
        placeholder_positions=placeholder_positions,
        static_prefix_len=static_prefix_len,
    )


class HistogramPytorchDataset(MEDSPytorchDataset):
    """A simplified PyTorch Dataset that computes histograms over time windows.
    
    This dataset loads patient history and chops it into time windows with N days between each,
    then computes histograms for each window efficiently. It handles empty windows with special
    tokens to avoid wasting execution time.
    
    Examples:
        >>> # Test initialization with histogram config
        >>> hist_config = HistogramConfig(
        ...     window_size_days=7,
        ...     anchoring_strategy=WindowAnchoringStrategy.RANDOM_EVENT
        ... )
        >>> cfg = MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
        ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
        ... )
        >>> dataset = HistogramPytorchDataset(cfg, "train", hist_config)
        >>> dataset.window_size_days
        7
        >>> dataset.anchoring_strategy
        <WindowAnchoringStrategy.RANDOM_EVENT: 'random_event'>
    """

    MIN_REQUIRED_MAX_SEQ_LEN = 1_000_000
    """Minimum sequence length that effectively disables truncation for histograms."""

    MAX_EXPECTED_WINDOWS = 256
    """Upper bound on the number of histogram windows produced for any subject."""
    
    def __init__(
        self,
        cfg: MEDSTorchDataConfig,
        split: str,
        histogram_config: HistogramConfig,
    ):
        """Initialize histogram dataset.
        
        Args:
            cfg: Base configuration object used to load the MEDS tensors.
            split: Data split (train/val/test)
            histogram_config: Histogram-specific configuration
            fusion_mode: Controls whether histogram windows are returned standalone or fused with
                the raw code timeline
            
        Examples:
            >>> hist_cfg = HistogramConfig(window_size_days=7)
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)
            >>> dataset.window_size_days
            7
            
            >>> cfg_default = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> default_hist_cfg = HistogramConfig()
            >>> dataset_default = HistogramPytorchDataset(cfg_default, "train", default_hist_cfg)
            >>> dataset_default.window_size_days
            30
        """
        if cfg is None:
            raise ValueError("HistogramPytorchDataset requires a MEDSTorchDataConfig instance.")

        super().__init__(cfg, split)
        
        # Set up histogram configuration
        self.histogram_config = histogram_config
        self.fusion_mode = FusionMode(self.histogram_config.fusion_mode)
            
        # Validate configuration
        self.histogram_config.validate()

        # Set convenience properties
        self.window_size_days = self.histogram_config.window_size_days
        self.anchoring_strategy = self.histogram_config.anchoring_strategy
        self.empty_window_mode = self.histogram_config.empty_window_mode
        self.empty_window_token = self.histogram_config.empty_window_token
        self.specific_event_regex = self.histogram_config.specific_event_regex
        self.include_anchor_token = self.histogram_config.include_anchor_token
        self.specific_event_codes = self.histogram_config.specific_event_codes
        self.subject_to_start_time = dict(pl.concat(self.schema_dfs_by_shard.values())['subject_id', 'start_time'].iter_rows())

        if self.anchoring_strategy == WindowAnchoringStrategy.SPECIFIC_EVENT:
            if not self.specific_event_codes:
                if not self.specific_event_regex:
                    raise ValueError("specific_event_regex must be set for SPECIFIC_EVENT anchoring")
                self.specific_event_codes = (
                    pl.read_parquet(cfg.code_metadata_fp)
                    .filter(pl.col("code").str.contains(self.specific_event_regex))
                    ['code/vocab_index']
                    .to_list()
                )
                if len(self.specific_event_codes) == 0:
                    raise ValueError(f"No codes found matching regex: {self.specific_event_regex}")
        
        self._validate_max_seq_len(getattr(self.config, "max_seq_len", None), fusion_mode=self.fusion_mode)
    
    def create_time_windows(self, patient_data: pl.DataFrame) -> List[Dict]:
        """Create time windows from patient data using OPTIMIZED polars vectorized operations.
        
        Args:
            patient_data: Patient data with 'time' and 'code' columns
            
        Returns:
            List of window dictionaries with histograms
            
        Examples:
            >>> import polars as pl
            >>> from datetime import datetime, timedelta
            >>> import torch
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            
            >>> # Test with END_OF_TIMELINE anchoring
            >>> config = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
            ...     empty_window_mode=EmptyWindowMode.IGNORE,
            ...     vocab_size=10
            ... )
            >>> dataset = HistogramPytorchDataset(cfg, "train", config)
            
            >>> # Create test patient data - 3 events over 60 days
            >>> patient_data = pl.DataFrame({
            ...     "subject_id": [1, 1, 1],
            ...     "time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 15), 
            ...         datetime(2020, 3, 1)
            ...     ],
            ...     "code": [1, 2, 3]
            ... })
            
            >>> # Test window creation
            >>> windows = dataset.create_time_windows(patient_data)
            >>> len(windows) >= 1  # Should create at least 1 window
            True
            >>> all('histogram' in w for w in windows)  # All windows should have histograms
            True
            >>> all('start_time' in w and 'end_time' in w for w in windows)  # Time boundaries
            True
            
            >>> # Test with RANDOM_EVENT anchoring
            >>> config_random = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.RANDOM_EVENT,
            ...     empty_window_mode=EmptyWindowMode.IGNORE,
            ...     vocab_size=10
            ... )
            >>> dataset_random = HistogramPytorchDataset(cfg, "train", config_random)
            >>> # Two different random seeds can yield different windowing
            >>> random.seed(0)
            >>> windows_random_1 = dataset_random.create_time_windows(patient_data)
            >>> random.seed(1)
            >>> windows_random_2 = dataset_random.create_time_windows(patient_data)
            >>> # Either number of windows or event counts per window should differ
            >>> (len(windows_random_1) != len(windows_random_2)) or any(len(a['codes']) != len(b['codes']) for a, b in zip(windows_random_1, windows_random_2))
            True
            
            >>> # Test empty patient data
            >>> empty_data = pl.DataFrame({"subject_id": [], "time": [], "code": []}, 
            ...                          schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.Int64})
            >>> try:
            ...     dataset.create_time_windows(empty_data)
            ...     False  # Should raise exception
            ... except ValueError:
            ...     True  # Expected
            True
            
            >>> # Test GAP WINDOW GENERATION with SINGLE_GAP mode (END_OF_TIMELINE)
            >>> config_gaps = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
            ...     empty_window_mode=EmptyWindowMode.SINGLE_GAP,
            ...     vocab_size=10
            ... )
            >>> dataset_gaps = HistogramPytorchDataset(cfg, "train", config_gaps)
            
            >>> # Create patient data with 13-month gap (like user's example)
            >>> patient_data_gaps = pl.DataFrame({
            ...     "subject_id": [1, 1, 1, 1],
            ...     "time": [
            ...         datetime(2020, 1, 15),   # Jan 2020
            ...         datetime(2020, 1, 20),   # Jan 2020  
            ...         datetime(2020, 1, 25),   # Jan 2020
            ...         datetime(2021, 3, 15)    # Mar 2021 - 13 month gap!
            ...     ],
            ...     "code": [1, 2, 3, 4]
            ... })
            
            >>> # Test gap window generation
            >>> windows_with_gaps = dataset_gaps.create_time_windows(patient_data_gaps)
            >>> len(windows_with_gaps) > 2  # Should have data windows AND gap windows
            True
            >>> gap_windows = [w for w in windows_with_gaps if w['is_gap']]
            >>> len(gap_windows) > 0  # Should have at least one gap window
            True
            >>> data_windows = [w for w in windows_with_gaps if not w['is_gap']]
            >>> len(data_windows) >= 2  # Should have data windows for Jan 2020 and Mar 2021
            True

            >>> # Boundary semantics for END_OF_TIMELINE with SINGLE_GAP:
            >>> # Right-most event should be included; left boundary should be exclusive
            >>> config_boundary = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
            ...     empty_window_mode=EmptyWindowMode.SINGLE_GAP,
            ...     vocab_size=10
            ... )
            >>> dataset_boundary = HistogramPytorchDataset(cfg, "train", config_boundary)
            >>> patient_boundary = pl.DataFrame({
            ...     "subject_id": [1, 1, 1],
            ...     "time": [
            ...         datetime(2020, 1, 1),         # t0
            ...         datetime(2020, 1, 31),        # t0 + 30 days (left boundary of last window)
            ...         datetime(2020, 3, 1)          # t0 + 60 days (max_time, right boundary)
            ...     ],
            ...     "code": [1, 2, 3]
            ... })
            >>> windows_boundary = dataset_boundary.create_time_windows(patient_boundary)
            >>> last_window = max([w for w in windows_boundary if not w['is_gap']], key=lambda x: x['end_time'])
            >>> 3 in last_window['codes']  # includes right-most event
            True
            >>> 2 in last_window['codes']  # excludes event exactly on left boundary of last window
            False

            >>> # SPECIFIC_EVENT anchoring without anchor token (quantizer training)
            >>> config_quant = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.SPECIFIC_EVENT,
            ...     empty_window_mode=EmptyWindowMode.IGNORE,
            ...     include_anchor_token=False,
            ...     specific_event_regex="^DISCHARGE$",
            ...     vocab_size=6,
            ...     specific_event_codes=[5],
            ... )
            >>> dataset_quant = HistogramPytorchDataset(cfg, "train", config_quant)
            >>> dataset_quant.specific_event_codes = [5]
            >>> patient_quant = pl.DataFrame({
            ...     "subject_id": [1, 1, 1],
            ...     "time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 2, 1),
            ...         datetime(2020, 3, 1)
            ...     ],
            ...     "code": [1, 5, 2]
            ... })
            >>> windows_quant = dataset_quant.create_time_windows(patient_quant)
            >>> len(windows_quant) >= 1
            True
            >>> any(w["is_anchor"] for w in windows_quant)
            True
            >>> any(w["is_anchor_token"] for w in windows_quant)
            False

            >>> # SPECIFIC_EVENT anchoring with anchor token (autoregressive training)
            >>> config_ar = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.SPECIFIC_EVENT,
            ...     empty_window_mode=EmptyWindowMode.SINGLE_GAP,
            ...     include_anchor_token=True,
            ...     specific_event_regex="^DISCHARGE$",
            ...     vocab_size=6,
            ...     specific_event_codes=[5],
            ... )
            >>> dataset_ar = HistogramPytorchDataset(cfg, "train", config_ar)
            >>> patient_ar = pl.DataFrame({
            ...     "subject_id": [1, 1, 1, 1],
            ...     "time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 2, 1),
            ...         datetime(2020, 2, 15),
            ...         datetime(2020, 3, 1)
            ...     ],
            ...     "code": [1, 5, 3, 4]
            ... })
            >>> windows_ar = dataset_ar.create_time_windows(patient_ar)
            >>> sum(1 for w in windows_ar if w["is_anchor_token"])
            1
            >>> any(w["is_anchor"] for w in windows_ar)
            True

            >>> # END_OF_TIMELINE anchoring with anchor token (inference)
            >>> config_infer = HistogramConfig(
            ...     window_size_days=30,
            ...     anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
            ...     empty_window_mode=EmptyWindowMode.SINGLE_GAP,
            ...     include_anchor_token=True,
            ...     vocab_size=6
            ... )
            >>> dataset_infer = HistogramPytorchDataset(cfg, "train", config_infer)
            >>> patient_infer = pl.DataFrame({
            ...     "subject_id": [2, 2, 2],
            ...     "time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 3, 10),
            ...         datetime(2020, 3, 20)
            ...     ],
            ...     "code": [2, 3, 4]
            ... })
            >>> windows_infer = dataset_infer.create_time_windows(patient_infer)
            >>> windows_infer[-1]["is_anchor_token"]
            True
            >>> any(w["is_gap"] for w in windows_infer)
            True
            >>> patient_infer = pl.DataFrame({
            ...     "subject_id": [2, 2],
            ...     "time": pl.Series(
            ...             [1583020800000000123, 1584662400000000789],
            ...             dtype=pl.Datetime("ns"),
            ...         ),
            ...     "code": [2, 3]
            ... })
            >>> windows_infer = dataset_infer.create_time_windows(patient_infer)
            >>> windows_infer[-1]["is_anchor_token"]
            True
        """
        if patient_data.is_empty():
            raise ValueError("Empty Patient Data")
            
        # OPTIMIZED: Use vectorized window assignment
        return self._create_windows_vectorized(patient_data)

    def _extract_raw_sequence(
        self,
        dynamic_data,
        base_data: Dict,
        windows: List[Dict],
    ) -> Dict[str, torch.Tensor | int]:
        """Slice the dynamic tensors to the usable portion for fusion workflows."""

        tensors = dynamic_data.tensors
        code_tensor = torch.as_tensor(tensors["dim0/code"]).long()
        time_tensor = torch.as_tensor(tensors["dim0/time_delta_days"], dtype=torch.float32)

        numeric_tensor = None
        if "dim0/numeric_value" in tensors:
            numeric_tensor = torch.as_tensor(tensors["dim0/numeric_value"], dtype=torch.float32)

        numeric_mask_tensor = None
        if "dim0/numeric_value_mask" in tensors:
            numeric_mask_tensor = torch.as_tensor(tensors["dim0/numeric_value_mask"], dtype=torch.bool)

        static_prefix_len = int(base_data.get("n_static_seq_els", 0) or 0)
        dynamic_code_count = sum(len(window.get("codes", [])) for window in windows)
        usable_length = static_prefix_len + dynamic_code_count
        usable_length = min(int(code_tensor.shape[0]), usable_length)

        # Ensure tensors exist even when originating data omits optional fields
        if numeric_tensor is None:
            numeric_tensor = torch.zeros_like(code_tensor, dtype=torch.float32)
        if numeric_mask_tensor is None:
            numeric_mask_tensor = torch.zeros_like(code_tensor, dtype=torch.bool)

        return {
            "code": code_tensor[:usable_length].clone(),
            "time_delta_days": time_tensor[:usable_length].clone(),
            "numeric_value": numeric_tensor[:usable_length].clone(),
            "numeric_value_mask": numeric_mask_tensor[:usable_length].clone(),
            "length": usable_length,
            "static_prefix_len": static_prefix_len,
        }

    def _apply_context_truncation(
        self,
        base_data: Dict,
        windows: List[Dict],
        histogram_tensors: List[torch.Tensor],
        gap_counts_list: List[int],
        anchor_index: int,
        raw_sequence: Dict[str, torch.Tensor | int] | None,
    ) -> tuple[
        List[Dict],
        List[torch.Tensor],
        List[int],
        int,
        Dict[str, torch.Tensor | int] | None,
    ]:
        """Apply optional pre/post-anchor truncation to raw code sequences and window metadata."""

        max_pre = self.histogram_config.max_pre_anchor_tokens
        max_total = self.histogram_config.max_code_seq_len

        if max_pre is None and max_total is None:
            return windows, histogram_tensors, gap_counts_list, anchor_index, raw_sequence

        code_lengths = [len(window.get("codes", []) or []) for window in windows]
        total_dynamic_codes = sum(code_lengths)
        if total_dynamic_codes == 0:
            return windows, histogram_tensors, gap_counts_list, anchor_index, raw_sequence

        if anchor_index < 0 or anchor_index >= len(windows):
            raise ValueError("Invalid anchor index when applying context truncation")

        anchor_start = sum(code_lengths[:anchor_index])

        keep_start = 0 if max_pre is None else max(0, anchor_start - max_pre)
        keep_end = total_dynamic_codes
        if max_total is not None:
            keep_end = min(total_dynamic_codes, keep_start + max_total)

        anchor_window_len = code_lengths[anchor_index]
        if keep_end < anchor_start and anchor_window_len > 0:
            raise ValueError(
                "Context truncation removed the anchor window codes; increase max_code_seq_len or "
                "reduce max_pre_anchor_tokens."
            )

        if keep_start == 0 and keep_end == total_dynamic_codes:
            return windows, histogram_tensors, gap_counts_list, anchor_index, raw_sequence

        static_prefix_len = int(base_data.get("n_static_seq_els", 0) or 0)
        if raw_sequence is not None:
            static_prefix_len = int(raw_sequence.get("static_prefix_len", static_prefix_len))

        new_windows: List[Dict] = []
        new_histograms: List[torch.Tensor] = []
        new_gap_counts: List[int] = []
        new_anchor_index: int | None = None
        new_dynamic_cursor = 0

        curr_dynamic_idx = 0
        for idx, window in enumerate(windows):
            raw_codes = window.get("codes", []) or []
            window_codes = list(raw_codes)
            window_len = len(window_codes)
            window_start = curr_dynamic_idx
            window_end = window_start + window_len
            curr_dynamic_idx = window_end

            is_anchor = bool(window.get("is_anchor"))
            is_anchor_token = bool(window.get("is_anchor_token", False))
            is_gap = bool(window.get("is_gap", False))

            kept_codes: List[int] = []
            if window_len > 0:
                overlap_start = max(keep_start, window_start)
                overlap_end = min(keep_end, window_end)
                if overlap_start < overlap_end:
                    local_start = overlap_start - window_start
                    local_end = overlap_end - window_start
                    kept_codes = window_codes[local_start:local_end]

            should_keep = bool(kept_codes)
            if not should_keep:
                if is_anchor or is_anchor_token:
                    should_keep = True
                elif is_gap and new_windows and keep_end > window_start:
                    should_keep = True

            if not should_keep:
                continue

            new_window = dict(window)
            new_window_codes = kept_codes if window_len > 0 else []
            new_window["codes"] = new_window_codes

            if window_len > 0:
                hist_tensor = self.compute_histogram(new_window_codes)
            else:
                hist_tensor = histogram_tensors[idx]
            new_window["histogram"] = hist_tensor

            if new_window_codes:
                first_idx = static_prefix_len + new_dynamic_cursor
                new_window["first_code_index"] = first_idx
                new_dynamic_cursor += len(new_window_codes)
            else:
                new_window["first_code_index"] = -1

            new_windows.append(new_window)
            new_histograms.append(hist_tensor)
            new_gap_counts.append(int(window.get("num_gap_windows", 0) or 0))

            if is_anchor:
                new_anchor_index = len(new_windows) - 1

        if new_anchor_index is None:
            raise ValueError("Anchor window not present after context truncation")

        kept_dynamic_len = max(0, keep_end - keep_start)
        if kept_dynamic_len != new_dynamic_cursor:
            raise ValueError("Mismatch between expected and actual dynamic code counts after truncation")

        dynamic_keep_len = kept_dynamic_len
        drop_front = keep_start

        dynamic_data = base_data["dynamic"]
        dynamic_segments = []
        if static_prefix_len > 0:
            dynamic_segments.append(dynamic_data[:static_prefix_len])

        start_idx = static_prefix_len + drop_front
        end_idx = start_idx + dynamic_keep_len
        if dynamic_keep_len > 0:
            dynamic_segments.append(dynamic_data[start_idx:end_idx])

        if not dynamic_segments:
            new_dynamic = dynamic_data[:0]
        elif len(dynamic_segments) == 1:
            new_dynamic = dynamic_segments[0]
        else:
            new_dynamic = dynamic_data.__class__.concatenate(dynamic_segments)

        base_data["dynamic"] = new_dynamic

        if raw_sequence is not None:
            def _slice_tensor(tensor: torch.Tensor) -> torch.Tensor:
                pieces: List[torch.Tensor] = []
                if static_prefix_len > 0:
                    pieces.append(tensor[:static_prefix_len])
                if dynamic_keep_len > 0:
                    pieces.append(tensor[start_idx:end_idx])
                if not pieces:
                    return tensor[:0].clone()
                if len(pieces) == 1:
                    return pieces[0].clone()
                return torch.cat(pieces, dim=0).clone()

            raw_sequence["code"] = _slice_tensor(raw_sequence["code"])  # type: ignore[index]
            raw_sequence["time_delta_days"] = _slice_tensor(raw_sequence["time_delta_days"])  # type: ignore[index]
            raw_sequence["numeric_value"] = _slice_tensor(raw_sequence["numeric_value"])  # type: ignore[index]
            raw_sequence["numeric_value_mask"] = _slice_tensor(raw_sequence["numeric_value_mask"])  # type: ignore[index]
            raw_sequence["length"] = int(raw_sequence["code"].shape[0])  # type: ignore[index]

        return new_windows, new_histograms, new_gap_counts, new_anchor_index, raw_sequence

    def _apply_window_subsampling(
        self,
        windows: List[Dict],
    ) -> tuple[List[Dict], int]:
        """Optionally subsample a contiguous window slice while preserving anchor windows."""
        max_windows = self.histogram_config.max_windows
        latest_start = max(0, len(windows) - max_windows)

        strategy = WindowSamplingStrategy(self.histogram_config.window_sampling_strategy)
        match strategy:
            case WindowSamplingStrategy.RANDOM:
                start_idx = random.randint(0, latest_start)
            case WindowSamplingStrategy.TO_END:
                start_idx = latest_start
            case _:
                raise ValueError(f"Invalid window_sampling_strategy: {strategy}")

        end_idx = start_idx + max_windows
        return windows[start_idx:end_idx]

    def _create_windows_vectorized(self, patient_data: pl.DataFrame) -> List[Dict]:
        """OPTIMIZED: vectorization using a manual integer window index.

        Supports both anchoring strategies by aligning a right-inclusive
        binning to a chosen anchor end-time and grouping by the resulting index.
        """
        # Ensure time column is properly typed (ns resolution for stability)
        if patient_data["time"].dtype != pl.Datetime:
            patient_data = patient_data.with_columns([pl.col("time").cast(pl.Datetime("ns"))])
        else:
            patient_data = patient_data.with_columns([pl.col("time").cast(pl.Datetime("ns"))])

        patient_data = (
            patient_data
            .with_row_count(name="event_index")
            .with_columns(pl.col("event_index").cast(pl.Int64))
        )

        min_time = patient_data["time"].min()
        max_time = patient_data["time"].max()
        if min_time is None or max_time is None:
            return []

        window_size_td = timedelta(days=self.window_size_days)
        NS_PER_DAY = 86_400_000_000_000
        W_NS = self.window_size_days * NS_PER_DAY

        # Work with integer nanoseconds to avoid precision loss when casting to Python datetimes
        time_ns_values = patient_data["time"].to_numpy().astype("datetime64[ns]").astype("int64")
        if time_ns_values.size == 0:
            return []
        time_ns_list = time_ns_values.tolist()
        max_ns = int(time_ns_values.max())

        # Compute final anchor end-time used for bin alignment
        if self.anchoring_strategy == WindowAnchoringStrategy.END_OF_TIMELINE:
            anchor_ns = max_ns
        elif self.anchoring_strategy == WindowAnchoringStrategy.RANDOM_EVENT:
            if len(patient_data) == 0:
                raise ValueError("Empty Patient Data")
            anchor_idx = random.randint(0, len(patient_data) - 1)
            event_anchor_ns = int(time_ns_values[anchor_idx])
            # Move anchor forward so one of the windows ends exactly at the event
            # and we still cover up to max_time (same logic as original)
            time_diff_ns = max_ns - event_anchor_ns
            windows_to_end = time_diff_ns // W_NS + 1
            anchor_ns = event_anchor_ns + windows_to_end * W_NS
        elif self.anchoring_strategy == WindowAnchoringStrategy.SPECIFIC_EVENT:
            codes_array = patient_data["code"].to_numpy()
            matching_mask = np.isin(codes_array, self.specific_event_codes or [])
            if matching_mask.any():
                anchor_ns = int(random.choice(time_ns_values[matching_mask].tolist()))
            else:
                anchor_ns = int(random.choice(time_ns_list))
        else:
            raise ValueError(f"Invalid anchoring_strategy: {self.anchoring_strategy}")

        # Convert anchor nanoseconds back to datetime for downstream use (Python datetime has microsecond precision)
        anchor_time_dt = pl.from_epoch(pl.Series([anchor_ns]), "ns").to_list()[0]

        # Assign right-inclusive window index: j = floor((t - A + W - 1) / W)
        with_index = (
            patient_data
            .with_columns(
                ((pl.col("time").dt.epoch("ns") - pl.lit(anchor_ns) + (W_NS - 1)) // W_NS).alias("window_idx")
            )
        )

        # Aggregate codes per window index
        grouped = (
            with_index
            .group_by("window_idx")
            .agg(
                [
                    pl.col("code").alias("codes"),
                    pl.col("event_index").min().cast(pl.Int64).alias("first_code_index"),
                ]
            )
            .sort("window_idx")
        )

        if grouped.is_empty():
            return []

        # Compute concrete window boundaries from index
        grouped = (
            grouped
            .with_columns([
                (pl.lit(anchor_ns) + pl.col("window_idx") * W_NS).alias("end_ns"),
                (pl.lit(anchor_ns) + (pl.col("window_idx") - 1) * W_NS).alias("start_ns"),
            ])
            .with_columns([
                pl.from_epoch(pl.col("start_ns"), "ns").alias("start_time"),
                pl.from_epoch(pl.col("end_ns"), "ns").alias("end_time"),
            ])
            .sort("window_idx")
        )

        windows: List[Dict] = []
        anchor_window_idx: Optional[int] = None

        if self.empty_window_mode == EmptyWindowMode.SINGLE_GAP:
            # Compute number of missing windows before each data window
            grouped = grouped.with_columns(
                (pl.col("window_idx") - pl.col("window_idx").shift(1) - 1)
                .clip(lower_bound=0)
                .cast(pl.Int32)
                .alias("num_gaps_before")
            )

            for row in grouped.iter_rows(named=True):
                window_start = row["start_time"]
                window_end = row["end_time"]
                codes = row["codes"]
                num_gaps_before = row["num_gaps_before"] or 0
                first_code_index = int(row["first_code_index"])

                if num_gaps_before > 0:
                    gap_end = window_start
                    gap_start = gap_end - num_gaps_before * window_size_td
                    windows.append({
                        "start_time": gap_start,
                        "end_time": gap_end,
                        "codes": [],
                        "histogram": self.get_empty_histogram(),
                        "is_gap": True,
                        "num_gap_windows": int(num_gaps_before),
                        "is_anchor": False,
                        "is_anchor_token": False,
                        "first_code_index": -1,
                    })

                histogram = self.compute_histogram(codes)
                is_anchor_window = row["end_ns"] == anchor_ns
                windows.append({
                    "start_time": window_start,
                    "end_time": window_end,
                    "codes": codes,
                    "histogram": histogram,
                    "is_gap": False,
                    "num_gap_windows": 0,
                    "is_anchor": bool(is_anchor_window),
                    "is_anchor_token": False,
                    "first_code_index": first_code_index,
                })
                if is_anchor_window:
                    anchor_window_idx = len(windows) - 1

            # If the last observed window index is below 0, we are missing
            # trailing windows up to the final anchor end (j_end == 0)
            last_j = grouped.select(pl.col("window_idx").max()).to_series()[0]
            trailing = 0 - last_j
            if trailing > 0:
                gap_end = anchor_time_dt  # end of j=0 window
                gap_start = gap_end - trailing * window_size_td
                windows.append({
                    "start_time": gap_start,
                    "end_time": gap_end,
                    "codes": [],
                    "histogram": self.get_empty_histogram(),
                    "is_gap": True,
                    "num_gap_windows": int(trailing),
                    "is_anchor": False,
                    "is_anchor_token": False,
                    "first_code_index": -1,
                })
        else:
            # IGNORE empty windows: only return data windows
            for row in grouped.iter_rows(named=True):
                window_start = row["start_time"]
                window_end = row["end_time"]
                codes = row["codes"]
                histogram = self.compute_histogram(codes)
                is_anchor_window = row["end_ns"] == anchor_ns
                windows.append({
                    "start_time": window_start,
                    "end_time": window_end,
                    "codes": codes,
                    "histogram": histogram,
                    "is_gap": False,
                    "num_gap_windows": 0,
                    "is_anchor": bool(is_anchor_window),
                    "is_anchor_token": False,
                    "first_code_index": int(row["first_code_index"]),
                })
                if is_anchor_window:
                    anchor_window_idx = len(windows) - 1

        if anchor_window_idx is None:
            windows.append({
                "start_time": anchor_time_dt,
                "end_time": anchor_time_dt,
                "codes": [],
                "histogram": self.get_empty_histogram(),
                "is_gap": False,
                "num_gap_windows": 0,
                "is_anchor": True,
                "is_anchor_token": False,
                "first_code_index": -1,
            })
            anchor_window_idx = len(windows) - 1

        if self.include_anchor_token:
            windows.insert(anchor_window_idx + 1, {
                "start_time": anchor_time_dt,
                "end_time": anchor_time_dt,
                "codes": [],
                "histogram": self.get_empty_histogram(),
                "is_gap": False,
                "num_gap_windows": 0,
                "is_anchor": False,
                "is_anchor_token": True,
                "first_code_index": -1,
            })

        return windows
    
    def compute_histogram(self, codes: List[int]) -> torch.Tensor:
        """Compute histogram from codes.
        
        Args:
            codes: List of medical codes
            
        Returns:
            Histogram tensor
            
        Examples:
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> hist_cfg = HistogramConfig()
            >>> dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)
            >>> hist = dataset.compute_histogram([1, 2, 1, 3])
            >>> hist.shape[0] == dataset.histogram_config.vocab_size
            True
            >>> hist[1].item() == 2  # Code 1 appears twice
            True
            
            >>> # Test with custom vocab size
            >>> config = HistogramConfig(vocab_size=5)
            >>> dataset = HistogramPytorchDataset(cfg, "train", config)
            >>> hist = dataset.compute_histogram([1, 2, 1, 3])
            >>> hist.shape[0] == 5
            True
            >>> hist[1].item() == 2
            True
        """
        vocab_size = self.histogram_config.vocab_size
        histogram = torch.zeros(vocab_size, dtype=torch.float32)
        
        for code in codes:
            if 0 <= code < vocab_size:
                histogram[code] += 1
                
        return histogram

    def get_empty_histogram(self) -> torch.Tensor:
        """Get histogram representing empty windows.
        
        Returns:
            All-zeros histogram tensor
            
        Examples:
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> hist_cfg = HistogramConfig()
            >>> dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)
            >>> empty_hist = dataset.get_empty_histogram()
            >>> empty_hist.sum().item() == 0  # All zeros
            True
            >>> empty_hist.shape[0] == dataset.histogram_config.vocab_size
            True
        """
        vocab_size = self.histogram_config.vocab_size
        return torch.zeros(vocab_size, dtype=torch.float32)
    
    def benchmark_performance(self, num_subjects: int = 100, avg_events_per_subject: int = 50) -> Dict[str, float]:
        """Benchmark the performance of histogram computation.
        
        Args:
            num_subjects: Number of subjects to simulate
            avg_events_per_subject: Average number of events per subject
            
        Returns:
            Dictionary with performance metrics
            
        Examples:
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> hist_cfg = HistogramConfig()
            >>> dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)
            >>> # Test benchmark method exists and callable (psutil may not be available)
            >>> callable(dataset.benchmark_performance)
            True
            >>> # Would return performance metrics if psutil is available:
            >>> # metrics = dataset.benchmark_performance(num_subjects=2, avg_events_per_subject=5)
            >>> # "time_per_subject" in metrics -> True
        """
        import time
        import psutil
        import os
        from datetime import datetime, timedelta
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Simulate processing multiple subjects
        for subject_id in range(num_subjects):
            # Generate mock patient data
            base_time = datetime(2020, 1, 1)
            times = [base_time + timedelta(days=i*7) for i in range(avg_events_per_subject)]
            codes = [random.randint(1, min(100, self.histogram_config.vocab_size)) for _ in range(avg_events_per_subject)]
            
            # Create mock dataframe
            mock_data = pl.DataFrame({
                "subject_id": [subject_id] * len(times),
                "time": times,
                "code": codes
            })
            
            # Process time windows
            windows = self.create_time_windows(mock_data)
            
            # Compute histograms
            for window in windows:
                _ = window["histogram"]
        
        end_time = time.time()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        
        return {
            "total_time_seconds": total_time,
            "time_per_subject": total_time / num_subjects,
            "memory_usage_mb": memory_usage,
            "subjects_processed": num_subjects,
            "avg_events_per_subject": avg_events_per_subject
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single patient's windowed histogram data.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Dictionary containing windowed histogram data
            
        Examples:
            >>> # Test basic structure (full test would need real dataset)
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            >>> hist_cfg = HistogramConfig()
            >>> dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)
            >>> # Verify dataset has required attributes
            >>> hasattr(dataset, 'histogram_config')
            True
            >>> hasattr(dataset, 'window_size_days')
            True
        """
        # Get base data from parent class
        base_data = super().__getitem__(idx)
        
        # Extract patient data for windowing
        subject_id, _ = self.index[idx]

        prediction_time = None
        if LabelSchema.prediction_time_name in self.schema_df.columns:
            schema_row = self.schema_df.row(idx, named=True)
            prediction_time = schema_row.get(LabelSchema.prediction_time_name)
        
        # Get dynamic data from JointNestedRaggedTensorDict
        dynamic_data = base_data["dynamic"]
        
        # Extract codes and time deltas from the tensors
        codes = dynamic_data.tensors["dim0/code"].tolist()
        time_deltas = dynamic_data.tensors["dim0/time_delta_days"].tolist()
        
        if len(codes) == 0 or len(time_deltas) == 0:
            raise ValueError("Empty patient data")
        
        # Convert time deltas to absolute timestamps
        # Assume first event is at reference time (day 0)
        base_time = self.subject_to_start_time[subject_id]  # Reference date
        cum_days = np.cumsum(np.nan_to_num(np.asarray(time_deltas, dtype=float), nan=0.0))
        NS_PER_DAY = 86_400_000_000_000  # 24*60*60*1e9
        cum_td_ns = np.round(cum_days * NS_PER_DAY).astype('int64').astype('timedelta64[ns]')
        base_ns = np.datetime64(base_time, 'ns')
        absolute_times = base_ns + cum_td_ns
        
        # Create patient dataframe with proper datetime objects
        patient_df = pl.DataFrame({
            "subject_id": [subject_id] * len(codes),
            "time": absolute_times,
            "code": codes
        })
        
        # Create time windows
        windows = self.create_time_windows(patient_df)

        if not windows:
            raise ValueError("No histogram windows generated for patient data")

        anchor_index = next((idx for idx, window in enumerate(windows) if window.get("is_anchor")), None)
        if anchor_index is None:
            raise ValueError("Anchor window not found in generated histogram windows")

        if self.histogram_config.max_windows is not None:
            if not self.fusion_mode == FusionMode.HISTOGRAM_ONLY:
                raise ValueError("max_windows is only supported for fusion_mode=histogram_only")
            windows = self._apply_window_subsampling(windows)

        raw_sequence = None
        if self.fusion_mode == FusionMode.HISTOGRAM_AND_CODE:
            raw_sequence = self._extract_raw_sequence(dynamic_data, base_data, windows)

        # Collect per-window tensors prior to optional truncation
        histogram_tensors = [w["histogram"] for w in windows]
        gap_counts_list = [int(w["num_gap_windows"]) for w in windows]

        windows, histogram_tensors, gap_counts_list, anchor_index, raw_sequence = self._apply_context_truncation(
            base_data,
            windows,
            histogram_tensors,
            gap_counts_list,
            anchor_index,
            raw_sequence,
        )

        device = histogram_tensors[0].device if histogram_tensors else torch.device("cpu")
        hist_dtype = histogram_tensors[0].dtype if histogram_tensors else torch.float32
        histograms = torch.stack([h.to(device=device, dtype=hist_dtype) for h in histogram_tensors])
        gap_counts = torch.tensor(gap_counts_list, dtype=torch.long, device=device)

        first_code_indices = torch.tensor(
            [w.get("first_code_index", -1) for w in windows],
            dtype=torch.long,
        )
        anchor_token_indices = torch.tensor(
            [idx for idx, w in enumerate(windows) if w.get("is_anchor_token", False)],
            dtype=torch.long,
        )
        anchor_index_tensor = torch.tensor([anchor_index], dtype=torch.long)

        # In fusion mode, build the fusion payload and use its placeholder_positions
        # as the authoritative source for first_code_indices
        if self.fusion_mode == FusionMode.HISTOGRAM_AND_CODE:
            if raw_sequence is None:
                raise ValueError("Fusion mode requires raw sequence data")

            fusion_payload = build_fusion_payload({
                **base_data,
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "windows": windows,
                "histograms": histograms,
                "gap_counts": gap_counts,
                "anchor_index": anchor_index_tensor,
                "anchor_token_indices": anchor_token_indices,
                "first_code_indices": first_code_indices,
                "_raw_sequence": raw_sequence,
            }, self.histogram_config.histogram_max_seq_len)

            # Use window metadata populated during fusion for first_code_indices
            first_code_indices = torch.tensor(
                [w.get("first_code_index", -1) for w in windows],
                dtype=torch.long,
            )

            result = {
                **base_data,
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "windows": windows,
                "histograms": histograms,
                "gap_counts": gap_counts,
                "anchor_index": anchor_index_tensor,
                "anchor_token_indices": anchor_token_indices,
                "first_code_indices": first_code_indices,
                "_fusion_payload": fusion_payload,  # Store for collate
            }
        else:
            result = {
                **base_data,
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "windows": windows,
                "histograms": histograms,
                "gap_counts": gap_counts,
                "anchor_index": anchor_index_tensor,
                "anchor_token_indices": anchor_token_indices,
                "first_code_indices": first_code_indices,
            }

        return result
    
    def collate(self, batch: List[Dict]) -> MEDSTorchHistogramBatch:
        """Collate a batch of windowed histogram data.
        
        Args:
            batch: List of patient data dictionaries
            
        Returns:
            MEDSTorchHistogramBatch with collated histogram data
            
        Examples:
            >>> import torch
            >>> from meds_torchdata.types import PaddingSide
            >>> cfg = MEDSTorchDataConfig(
            ...     tensorized_cohort_dir=tensorized_MEDS_dataset,
            ...     max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
            ... )
            
            >>> # Test padding side configuration is properly set
            >>> config_right = HistogramConfig(vocab_size=3, padding_side=PaddingSide.RIGHT)
            >>> config_right.padding_side == PaddingSide.RIGHT
            True
            >>> config_left = HistogramConfig(vocab_size=3, padding_side=PaddingSide.LEFT)
            >>> config_left.padding_side == PaddingSide.LEFT
            True
            
            >>> # Test that we can create datasets with different padding configurations
            >>> dataset_right = HistogramPytorchDataset(cfg, "train", config_right)
            >>> dataset_right.histogram_config.padding_side == PaddingSide.RIGHT
            True
            >>> dataset_left = HistogramPytorchDataset(cfg, "train", config_left)
            >>> dataset_left.histogram_config.padding_side == PaddingSide.LEFT
            True
            
            >>> # Test that empty histograms have the correct shape
            >>> empty_right = dataset_right.get_empty_histogram()
            >>> empty_right.shape
            torch.Size([3])
            >>> empty_left = dataset_left.get_empty_histogram()
            >>> empty_left.shape
            torch.Size([3])

            >>> # ARCHITECTURE: Separation of fusion logic from padding logic
            >>> # Fusion payloads are now built in __getitem__ with correct placeholder_positions
            >>> # Collate just handles padding and offsetting - no reconciliation needed
            >>> import torch
            >>>
            >>> # Simulate two samples with different structures (as built by __getitem__)
            >>> # Sample 0: 2 placeholders in fusion payload, first_code_indices matches exactly
            >>> sample0_payload = FusionPayload(
            ...     codes=torch.tensor([-1, 101, -1, 102, -2], dtype=torch.long),  # 2 hist placeholders, 1 anchor
            ...     time_delta_days=torch.zeros(5),
            ...     numeric_value=torch.zeros(5),
            ...     numeric_value_mask=torch.zeros(5, dtype=torch.bool),
            ...     placeholder_positions=[0, 2],  # 2 placeholders
            ...     static_prefix_len=0,
            ... )
            >>> # Sample 1: 2 placeholders, first_code_indices matches exactly
            >>> sample1_payload = FusionPayload(
            ...     codes=torch.tensor([-1, 201, -1, 202], dtype=torch.long),  # 2 hist placeholders
            ...     time_delta_days=torch.zeros(4),
            ...     numeric_value=torch.zeros(4),
            ...     numeric_value_mask=torch.zeros(4, dtype=torch.bool),
            ...     placeholder_positions=[0, 2],  # 2 placeholders
            ...     static_prefix_len=0,
            ... )
            >>>
            >>> # Batch items now have CORRECT first_code_indices (from fusion payload placeholder_positions)
            >>> batch_correct = [
            ...     {
            ...         "histograms": torch.rand(2, 308),  # 2 histogram windows (may differ from placeholders)
            ...         "gap_counts": torch.zeros(2, dtype=torch.long),
            ...         "anchor_index": torch.tensor([2]),
            ...         "anchor_token_indices": torch.tensor([]),
            ...         "first_code_indices": torch.tensor([0, 2]),  # CORRECT: matches placeholder_positions
            ...         "windows": [{"start_time": 0, "end_time": 1, "is_gap": False, "num_gap_windows": 0, "is_anchor": False, "is_anchor_token": False, "first_code_index": 0}] * 2,
            ...         "subject_id": 1,
            ...         "prediction_time": None,
            ...         "_fusion_payload": sample0_payload,  # Pre-built in __getitem__
            ...     },
            ...     {
            ...         "histograms": torch.rand(2, 308),  # 2 histogram windows
            ...         "gap_counts": torch.zeros(2, dtype=torch.long),
            ...         "anchor_index": torch.tensor([1]),
            ...         "anchor_token_indices": torch.tensor([]),
            ...         "first_code_indices": torch.tensor([0, 2]),  # CORRECT: matches placeholder_positions
            ...         "windows": [{"start_time": 0, "end_time": 1, "is_gap": False, "num_gap_windows": 0, "is_anchor": False, "is_anchor_token": False, "first_code_index": 0}] * 2,
            ...         "subject_id": 2,
            ...         "prediction_time": None,
            ...         "_fusion_payload": sample1_payload,  # Pre-built in __getitem__
            ...     }
            ... ]
            >>>
            >>> # Verify: first_code_indices always matches placeholder_positions (single source of truth)
            >>> print(f"Sample 0: first_code_indices={batch_correct[0]['first_code_indices'].tolist()}, placeholders={batch_correct[0]['_fusion_payload'].placeholder_positions}")
            Sample 0: first_code_indices=[0, 2], placeholders=[0, 2]
            >>> print(f"Sample 1: first_code_indices={batch_correct[1]['first_code_indices'].tolist()}, placeholders={batch_correct[1]['_fusion_payload'].placeholder_positions}")
            Sample 1: first_code_indices=[0, 2], placeholders=[0, 2]
            >>>
            >>> # Collate just handles padding offsets - no mismatch possible
            >>> # The fusion payloads are extracted as-is, indices are already correct
            >>> HistogramPytorchDataset._align_first_code_indices(
            ...     torch.tensor([0, -1, 2]),
            ...     [5, 12],
            ... ).tolist()
            [5, -1, 12]
        """
        # Get base collated data
        base_batch = super().collate(batch)

        # Extract pre-built fusion payloads if in fusion mode
        fusion_payloads: List[FusionPayload] | None = None
        if self.fusion_mode == FusionMode.HISTOGRAM_AND_CODE:
            fusion_payloads = [item["_fusion_payload"] for item in batch]
            # Clean up the temporary payload from batch items
            for item in batch:
                del item["_fusion_payload"]

        # Collate histograms and gap_counts with padding
        all_histograms = [item["histograms"] for item in batch]
        all_gap_counts = [item["gap_counts"] for item in batch]
        all_anchor_indices = [item["anchor_index"] for item in batch]
        all_anchor_token_indices = [item["anchor_token_indices"] for item in batch]
        all_first_code_indices = [item["first_code_indices"] for item in batch]
        subject_ids = [item["subject_id"] for item in batch]
        prediction_times = [item.get("prediction_time") for item in batch]

        # Preserve per-window metadata off-device so downstream code can access timestamps directly
        subject_window_metadata: list[list[dict]] = []
        for item in batch:
            subj_windows: list[dict] = []
            subj_id = item["subject_id"]
            for window in item["windows"]:
                subj_windows.append({
                    "subject_id": subj_id,
                    "start_time": window["start_time"],
                    "end_time": window["end_time"],
                    "is_gap": window["is_gap"],
                    "num_gap_windows": window["num_gap_windows"],
                    "is_anchor": window["is_anchor"],
                    "is_anchor_token": window.get("is_anchor_token", False),
                    "first_code_index": window.get("first_code_index", -1),
                })
            subject_window_metadata.append(subj_windows)

        if all_histograms and len(all_histograms[0]) > 0:
            # Pad sequences to same length
            max_windows = max(len(h) for h in all_histograms)
            vocab_size = all_histograms[0].shape[1]

            padded_histograms = []
            padded_gap_counts = []
            padded_anchor_indices = []
            padded_anchor_token_indices = []
            padded_first_code_indices = []

            for idx, (histograms, gap_counts, first_code_indices) in enumerate(
                zip(all_histograms, all_gap_counts, all_first_code_indices)
            ):
                pad_amount = max_windows - len(histograms)

                if pad_amount > 0:
                    # Pad with empty histograms and zero gap counts
                    hist_padding = torch.zeros((pad_amount, vocab_size))
                    gap_padding = torch.zeros((pad_amount,), dtype=torch.long)
                    index_padding = torch.full((pad_amount,), -1, dtype=torch.long)

                    # Apply padding based on padding_side configuration
                    if self.histogram_config.padding_side == PaddingSide.LEFT:
                        padded_hist = torch.cat([hist_padding, histograms], dim=0)
                        padded_gaps = torch.cat([gap_padding, gap_counts], dim=0)
                        padded_indices = torch.cat([index_padding, first_code_indices], dim=0)
                    else:  # RIGHT padding (default)
                        padded_hist = torch.cat([histograms, hist_padding], dim=0)
                        padded_gaps = torch.cat([gap_counts, gap_padding], dim=0)
                        padded_indices = torch.cat([first_code_indices, index_padding], dim=0)
                else:
                    padded_hist = histograms
                    padded_gaps = gap_counts
                    padded_indices = first_code_indices

                padded_histograms.append(padded_hist)
                padded_gap_counts.append(padded_gaps)
                padded_first_code_indices.append(padded_indices)

                anchor_tensor = all_anchor_indices[idx]
                if anchor_tensor.numel() != 1:
                    raise ValueError("anchor_index tensors must contain exactly one element")
                anchor_value = anchor_tensor.item()
                if self.histogram_config.padding_side == PaddingSide.LEFT:
                    anchor_value += pad_amount
                padded_anchor_indices.append(torch.tensor([anchor_value], dtype=torch.long))

                anchor_token_tensor = all_anchor_token_indices[idx]
                if pad_amount > 0 and self.histogram_config.padding_side == PaddingSide.LEFT:
                    anchor_token_tensor = anchor_token_tensor + pad_amount
                padded_anchor_token_indices.append(anchor_token_tensor)

            batch_histograms = torch.stack(padded_histograms)
            batch_gap_counts = torch.stack(padded_gap_counts)
            batch_anchor_indices = torch.stack(padded_anchor_indices)
            batch_anchor_token_indices = torch.nn.utils.rnn.pad_sequence(
                padded_anchor_token_indices,
                batch_first=True,
                padding_value=-1,
            )
            batch_first_code_indices = torch.stack(padded_first_code_indices)
        else:
            # Handle empty case
            batch_histograms = torch.zeros((len(batch), 0, self.histogram_config.vocab_size))
            batch_gap_counts = torch.zeros((len(batch), 0), dtype=torch.long)
            batch_anchor_indices = torch.stack(all_anchor_indices)
            batch_anchor_token_indices = torch.nn.utils.rnn.pad_sequence(
                all_anchor_token_indices,
                batch_first=True,
                padding_value=-1,
            )
            batch_first_code_indices = torch.full((len(batch), 0), -1, dtype=torch.long)

        base_data = dict(base_batch.items())

        if any(pt is not None for pt in prediction_times):
            prediction_array = np.array(
                [
                    np.datetime64(pt).astype("datetime64[ns]") if pt is not None else np.datetime64("NaT")
                    for pt in prediction_times
                ],
                dtype="datetime64[ns]",
            )
        else:
            prediction_array = None

        code_tensor_out = base_data.get("code")
        time_tensor_out = base_data.get("time_delta_days")
        numeric_value_out = base_data.get("numeric_value")
        numeric_mask_out = base_data.get("numeric_value_mask")
        static_mask_out = base_data.get("static_mask")

        if self.fusion_mode == FusionMode.HISTOGRAM_AND_CODE and fusion_payloads is not None:
            padding_side = self.config.padding_side
            max_seq_len = max(payload.codes.shape[0] for payload in fusion_payloads) if fusion_payloads else 0

            padded_codes: List[torch.Tensor] = []
            padded_time_deltas: List[torch.Tensor] = []
            padded_numeric_values: List[torch.Tensor] = []
            padded_numeric_masks: List[torch.Tensor] = []
            padded_placeholder_positions: List[List[int]] = []

            static_rows: List[torch.Tensor] | None = [] if static_mask_out is not None else None

            for payload in fusion_payloads:
                seq_len = payload.codes.shape[0]
                pad_amount = max_seq_len - seq_len
                if pad_amount < 0:
                    raise ValueError("Computed negative padding for fusion payload.")

                if padding_side == PaddingSide.LEFT:
                    code_pad = torch.zeros(pad_amount, dtype=payload.codes.dtype)
                    time_pad = torch.zeros(pad_amount, dtype=payload.time_delta_days.dtype)
                    numeric_pad = torch.zeros(pad_amount, dtype=payload.numeric_value.dtype)
                    mask_pad = torch.zeros(pad_amount, dtype=payload.numeric_value_mask.dtype)

                    padded_codes.append(torch.cat([code_pad, payload.codes], dim=0))
                    padded_time_deltas.append(torch.cat([time_pad, payload.time_delta_days], dim=0))
                    padded_numeric_values.append(torch.cat([numeric_pad, payload.numeric_value], dim=0))
                    padded_numeric_masks.append(torch.cat([mask_pad, payload.numeric_value_mask], dim=0))
                    padded_placeholder_positions.append(
                        [pos + pad_amount for pos in payload.placeholder_positions]
                    )
                else:
                    code_pad = torch.zeros(pad_amount, dtype=payload.codes.dtype)
                    time_pad = torch.zeros(pad_amount, dtype=payload.time_delta_days.dtype)
                    numeric_pad = torch.zeros(pad_amount, dtype=payload.numeric_value.dtype)
                    mask_pad = torch.zeros(pad_amount, dtype=payload.numeric_value_mask.dtype)

                    padded_codes.append(torch.cat([payload.codes, code_pad], dim=0))
                    padded_time_deltas.append(torch.cat([payload.time_delta_days, time_pad], dim=0))
                    padded_numeric_values.append(torch.cat([payload.numeric_value, numeric_pad], dim=0))
                    padded_numeric_masks.append(torch.cat([payload.numeric_value_mask, mask_pad], dim=0))
                    padded_placeholder_positions.append(list(payload.placeholder_positions))

                if static_rows is not None:
                    static_row = torch.zeros(max_seq_len, dtype=static_mask_out.dtype)
                    static_len = payload.static_prefix_len
                    if static_len > 0:
                        if padding_side == PaddingSide.LEFT:
                            start = pad_amount
                            static_row[start : start + static_len] = True
                        else:
                            static_row[:static_len] = True
                    static_rows.append(static_row)

            code_tensor_out = torch.stack(padded_codes)
            time_tensor_out = torch.stack(padded_time_deltas)
            numeric_value_out = torch.stack(padded_numeric_values)
            numeric_mask_out = torch.stack(padded_numeric_masks)
            if static_rows is not None and static_rows:
                static_mask_out = torch.stack(static_rows)
            batch_first_code_indices = torch.stack(padded_first_code_indices)

            for batch_idx, positions in enumerate(padded_placeholder_positions):
                batch_first_code_indices[batch_idx] = self._align_first_code_indices(
                    batch_first_code_indices[batch_idx],
                    positions,
                )

        histogram_kwargs = {**base_data}
        histogram_kwargs.update(
            {
                "histograms": batch_histograms,
                "gap_counts": batch_gap_counts,
                "anchor_indices": batch_anchor_indices,
                "anchor_token_indices": batch_anchor_token_indices,
                "first_code_indices": batch_first_code_indices,
            }
        )
        if code_tensor_out is not None:
            histogram_kwargs["code"] = code_tensor_out
        if time_tensor_out is not None:
            histogram_kwargs["time_delta_days"] = time_tensor_out
        if numeric_value_out is not None:
            histogram_kwargs["numeric_value"] = numeric_value_out
        if numeric_mask_out is not None:
            histogram_kwargs["numeric_value_mask"] = numeric_mask_out
        if static_mask_out is not None:
            histogram_kwargs["static_mask"] = static_mask_out

        histogram_batch = MEDSTorchHistogramBatch(**histogram_kwargs)
        object.__setattr__(histogram_batch, "subject_ids", subject_ids)
        object.__setattr__(histogram_batch, "subject_id", torch.as_tensor(subject_ids, dtype=torch.long))
        object.__setattr__(histogram_batch, "prediction_time", prediction_array)
        object.__setattr__(histogram_batch, "window_metadata", subject_window_metadata)
        object.__setattr__(histogram_batch, "padding_side", self.histogram_config.padding_side)

        return histogram_batch

    @staticmethod
    def _align_first_code_indices(
        row: torch.Tensor,
        placeholder_positions: List[int],
    ) -> torch.Tensor:
        """Align per-window indices with updated placeholder positions."""

        if row.ndim != 1:
            raise ValueError("first_code_indices rows must be 1-dimensional tensors.")

        result = row.clone()
        positive_count = int((result >= 0).sum().item())
        if positive_count != len(placeholder_positions):
            raise ValueError(
                "Mismatch between histogram windows and placeholder positions in fusion payload."
            )

        position_idx = 0
        for idx in range(result.shape[0]):
            if result[idx] >= 0:
                result[idx] = placeholder_positions[position_idx]
                position_idx += 1

        return result

    @classmethod
    def _validate_max_seq_len(
        cls,
        max_seq_len: Optional[int],
        *,
        fusion_mode: FusionMode = FusionMode.HISTOGRAM_ONLY,
    ) -> None:
        """Ensure histogram dataset sees full patient sequences.

        Args:
            max_seq_len: Maximum sequence length from the base config.
            fusion_mode: Selected fusion mode for the dataset.

        Raises:
            ValueError: If the maximum sequence length would truncate patient timelines.
        """
        if max_seq_len is None:
            return

        if max_seq_len <= 0:
            raise ValueError("HistogramPytorchDataset requires config.max_seq_len to be positive.")

        if max_seq_len < cls.MIN_REQUIRED_MAX_SEQ_LEN:
            raise ValueError(
                "HistogramPytorchDataset requires config.max_seq_len to be "
                f"None or >= {cls.MIN_REQUIRED_MAX_SEQ_LEN:,} but got {max_seq_len}."
            )

    @classmethod
    def _validate_window_count(cls, window_count: int) -> None:
        """Ensure window generation stays within expected bounds."""
        if window_count > cls.MAX_EXPECTED_WINDOWS:
            print(
                f"HistogramPytorchDataset expected <= {cls.MAX_EXPECTED_WINDOWS} windows per subject "
                f"but received {window_count}."
            )
            return False
        return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()
