from dataclasses import replace

import torch
import pytest

from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.histogram_dataset import (
    FusionMode,
    HistogramConfig,
    HistogramPytorchDataset,
    WindowAnchoringStrategy,
)
from meds_torchdata.types import PaddingSide


def test_validate_max_seq_len_accepts_none_and_large() -> None:
    HistogramPytorchDataset._validate_max_seq_len(None)
    HistogramPytorchDataset._validate_max_seq_len(
        HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN
    )


def test_validate_max_seq_len_rejects_small_values() -> None:
    with pytest.raises(ValueError):
        HistogramPytorchDataset._validate_max_seq_len(
            HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN - 1
        )


def test_validate_window_count_enforces_expected_limit() -> None:
    limit = HistogramPytorchDataset.MAX_EXPECTED_WINDOWS
    HistogramPytorchDataset._validate_window_count(limit)
    with pytest.raises(ValueError):
        HistogramPytorchDataset._validate_window_count(limit + 1)


def test_validate_max_seq_len_requires_large_even_in_fusion() -> None:
    with pytest.raises(ValueError):
        HistogramPytorchDataset._validate_max_seq_len(
            HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN - 1,
            fusion_mode=FusionMode.HISTOGRAM_AND_CODE,
        )


def test_histogram_first_code_indices_alignment(
    sample_dataset_config: MEDSTorchDataConfig,
) -> None:
    hist_cfg = HistogramConfig()
    cfg = replace(
        sample_dataset_config,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )
    dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)

    sample = dataset[0]
    raw_codes = sample["dynamic"].tensors["dim0/code"].tolist()
    first_code_indices = sample["first_code_indices"]

    assert first_code_indices.shape[0] == sample["histograms"].shape[0]

    for idx, window in enumerate(sample["windows"]):
        window_first_idx = window["first_code_index"]
        tensor_first_idx = first_code_indices[idx].item()
        assert tensor_first_idx == window_first_idx
        if window["codes"]:
            assert tensor_first_idx >= 0
            assert raw_codes[tensor_first_idx] == window["codes"][0]
        else:
            assert tensor_first_idx == -1

    collated = dataset.collate([sample])
    assert torch.equal(collated.first_code_indices[0], first_code_indices)
    for idx, metadata in enumerate(collated.window_metadata[0]):
        assert metadata["first_code_index"] == sample["windows"][idx]["first_code_index"]


def test_histogram_first_code_indices_left_padding(
    sample_dataset_config: MEDSTorchDataConfig,
) -> None:
    hist_cfg = HistogramConfig(padding_side=PaddingSide.LEFT, window_size_days=0.5)
    cfg = replace(
        sample_dataset_config,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )
    dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)

    dataset_used = None
    shorter = longer = None
    candidates = [dataset[idx] for idx in range(len(dataset))]

    for i, item_a in enumerate(candidates):
        for item_b in candidates[i + 1 :]:
            if len(item_a["windows"]) != len(item_b["windows"]):
                if len(item_a["windows"]) < len(item_b["windows"]):
                    shorter, longer = item_a, item_b
                else:
                    shorter, longer = item_b, item_a
                dataset_used = dataset
                break
        if shorter is not None:
            break

    if shorter is None or longer is None or dataset_used is None:
        raise ValueError(f"No subjects with differing histogram lengths found to test left padding {[dataset[idx]['histograms'].shape[0] for idx in range(len(dataset))]}")

    batch = dataset_used.collate([shorter, longer])
    max_windows = batch.histograms.shape[1]
    pad_amount = max_windows - len(shorter["windows"])
    assert pad_amount >= 0

    padded_first_indices = batch.first_code_indices[0]
    assert torch.all(padded_first_indices[:pad_amount] == -1)
    assert torch.equal(
        padded_first_indices[pad_amount:],
        shorter["first_code_indices"],
    )
    assert torch.equal(batch.first_code_indices[1], longer["first_code_indices"])


def test_fusion_sequence_inserts_histogram_tokens(
    sample_dataset_config: MEDSTorchDataConfig,
) -> None:
    cfg = replace(
        sample_dataset_config,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )
    hist_cfg = HistogramConfig(histogram_max_seq_len=64, fusion_mode=FusionMode.HISTOGRAM_AND_CODE, include_anchor_token=True)
    dataset = HistogramPytorchDataset(
        cfg,
        "train",
        hist_cfg,
    )

    sample = dataset[0]
    batch = dataset.collate([sample])

    window_count = sample["histograms"].shape[0]
    placeholder_indices = batch.first_code_indices[0, :window_count]
    codes = batch.code[0]

    # Placeholders can be -1 (normal histogram) or -3 (gap)
    for idx in placeholder_indices.tolist():
        if idx == -1:
            continue
        else:
            assert codes[idx] in [-1, -3], f"Expected placeholder at {idx}, got {codes[idx]}"

    # Anchor token comes AFTER codes for that window, not immediately after histogram
    anchor_window_idx = sample["anchor_index"].item()
    hist_pos = placeholder_indices[anchor_window_idx].item()

    # Find anchor token by scanning forward from histogram position
    # Structure: [hist, code1, code2, ..., codeN, anchor]
    anchor_found = False
    for offset in range(1, len(codes) - hist_pos):
        if codes[hist_pos + offset] == -2:  # ANCHOR_TOKEN_ID
            anchor_found = True
            break
    assert anchor_found, f"Anchor token not found after histogram at position {hist_pos}"


def test_context_truncation_limits_tokens(
    sample_dataset_config: MEDSTorchDataConfig,
) -> None:
    cfg = replace(
        sample_dataset_config,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )

    full_cfg = HistogramConfig(fusion_mode=FusionMode.HISTOGRAM_AND_CODE)
    full_dataset = HistogramPytorchDataset(cfg, "train", full_cfg)
    full_sample = full_dataset[0]
    full_windows = full_sample["windows"]
    full_anchor_idx = full_sample["anchor_index"].item()

    def _dynamic_code_count(windows: list[dict], stop: int | None = None) -> int:
        out = 0
        for idx, window in enumerate(windows):
            if stop is not None and idx >= stop:
                break
            if window.get("is_gap", False):
                continue
            out += len(window.get("codes", []) or [])
        return out

    pre_anchor_full = _dynamic_code_count(full_windows, full_anchor_idx)
    total_full = _dynamic_code_count(full_windows)

    trunc_cfg = HistogramConfig(
        fusion_mode=FusionMode.HISTOGRAM_AND_CODE,
        max_pre_anchor_tokens=3,
        max_code_seq_len=7,
    )
    trunc_dataset = HistogramPytorchDataset(cfg, "train", trunc_cfg)
    trunc_sample = trunc_dataset[0]
    trunc_windows = trunc_sample["windows"]
    trunc_anchor_idx = trunc_sample["anchor_index"].item()

    pre_anchor_trunc = _dynamic_code_count(trunc_windows, trunc_anchor_idx)
    total_trunc = _dynamic_code_count(trunc_windows)

    expected_pre = min(pre_anchor_full, 3)
    assert pre_anchor_trunc == expected_pre
    assert total_trunc <= 7
    if total_full <= 7:
        assert total_trunc == total_full
    else:
        assert total_trunc == 7

    payload = trunc_sample["_fusion_payload"]
    static_prefix_len = int(payload.static_prefix_len)
    raw_dynamic_len = payload.codes.shape[0] - static_prefix_len
    assert raw_dynamic_len >= total_trunc  # placeholders inflate fused length

    dynamic_codes = trunc_sample["dynamic"].tensors["dim0/code"].tolist()
    assert len(dynamic_codes) - static_prefix_len == total_trunc


def test_context_truncation_left_padding(
    sample_dataset_config: MEDSTorchDataConfig,
) -> None:
    cfg = replace(
        sample_dataset_config,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )
    hist_cfg = HistogramConfig(
        padding_side=PaddingSide.LEFT,
        fusion_mode=FusionMode.HISTOGRAM_AND_CODE,
        max_pre_anchor_tokens=2,
    )
    dataset = HistogramPytorchDataset(cfg, "train", hist_cfg)

    sample = dataset[0]
    collated = dataset.collate([sample])

    window_count = len(sample["windows"])
    pad_amount = collated.histograms.shape[1] - window_count
    assert pad_amount >= 0
    assert torch.all(collated.first_code_indices[0, :pad_amount] == -1)
    assert torch.equal(
        collated.first_code_indices[0, pad_amount:],
        sample["first_code_indices"],
    )
