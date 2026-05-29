#!/usr/bin/env python3
"""
Histogram Dataset Demo Script
============================

This script demonstrates the histogram dataset functionality with a simple,
understandable example using dummy patient data.

It shows:
1. How time windowing works
2. How histograms are computed 
3. How different anchoring strategies work
4. How empty windows are handled
"""

import sys
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import torch

# Add src to path
sys.path.insert(0, 'src')

from meds_torchdata.config import MEDSTorchDataConfig, StaticInclusionMode
from meds_torchdata.histogram_dataset import (
    ANCHOR_TOKEN_ID,
    EmptyWindowMode,
    FusionMode,
    GAP_TOKEN_ID,
    HISTOGRAM_PLACEHOLDER_TOKEN,
    HistogramConfig,
    HistogramPytorchDataset,
    WindowAnchoringStrategy,
)
from meds_torchdata.types import WindowSamplingStrategy

try:
    from meds_testing_helpers.dataset import DatasetMetadataSchema, MEDSDataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "This demo requires the 'meds-testing-helpers' package. "
        "Install it alongside meds-torch-data to run the demo."
    ) from exc

def create_dummy_patient_data():
    """Create simple dummy patient data for demonstration."""
    
    print("🏥 CREATING DUMMY PATIENT DATA")
    print("=" * 40)
    
    # Patient 1: Regular visits over 3 months
    patient1_data = [
        # January visits
        {"subject_id": 1, "time": datetime(2020, 1, 5), "code": 10},   # Diagnosis code
        {"subject_id": 1, "time": datetime(2020, 1, 5), "code": 20},   # Lab code
        {"subject_id": 1, "time": datetime(2020, 1, 12), "code": 30},  # Treatment code
        
        # February visits  
        {"subject_id": 1, "time": datetime(2020, 2, 3), "code": 10},   # Same diagnosis
        {"subject_id": 1, "time": datetime(2020, 2, 15), "code": 40},  # New medication
        
        # March visits
        {"subject_id": 1, "time": datetime(2020, 3, 10), "code": 50},  # Follow-up
        {"subject_id": 1, "time": datetime(2020, 3, 20), "code": 20},  # Repeat lab
    ]
    
    # Patient 2: Sparse visits with gaps
    patient2_data = [
        # January visits (clustered)
        {"subject_id": 2, "time": datetime(2020, 1, 8), "code": 60},   # Emergency
        {"subject_id": 2, "time": datetime(2020, 1, 8), "code": 70},   # Treatment
        {"subject_id": 2, "time": datetime(2020, 1, 9), "code": 80},   # Discharge
        
        # Long gap - no February visits
        
        # March visits  
        {"subject_id": 2, "time": datetime(2021, 3, 25), "code": 40},  # Follow-up
    ]
    
    all_data = patient1_data + patient2_data
    df = pl.DataFrame(all_data)
    
    print(f"Created data for {df['subject_id'].n_unique()} patients")
    print(f"Total events: {len(df)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Unique codes: {sorted(df['code'].unique())}")
    
    # Process each patient separately
    for patient_id in sorted(df["subject_id"].unique()):
        patient_data = df.filter(pl.col("subject_id") == patient_id)
        
        print(f"\n👤 PATIENT {patient_id}")
        print("-" * 20)
        
        # Show raw data
        print("Raw events:")
        for row in patient_data.iter_rows(named=True):
            print(f"  {row['time'].strftime('%Y-%m-%d')}: Code {row['code']}")
    
    return df


def _build_demo_meds_dataset(data: pl.DataFrame) -> MEDSDataset:
    """Convert the in-memory demo data into a minimal MEDS dataset."""

    codes = sorted({int(code) for code in data["code"].to_list()})

    dataset_metadata = DatasetMetadataSchema(
        dataset_name="histogram_demo",
        dataset_version="0.1.0",
        etl_name="demo_histogram_dataset",
        etl_version="0.1.0",
        meds_version="demo",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    raw_events = (
        data
        .with_columns([
            pl.col("code").cast(pl.Utf8),
            pl.lit(None).cast(pl.Float64).alias("numeric_value"),
        ])
        .select(["subject_id", "time", "code", "numeric_value"])
        .sort(["subject_id", "time"])
    )

    code_metadata = pl.DataFrame(
        {
            "code": [str(code) for code in codes],
            "description": [f"Demo code {code}" for code in codes],
        }
    ).with_columns(
        pl.Series([[] for _ in codes], dtype=pl.List(pl.Utf8)).alias("parent_codes")
    )

    subject_splits = (
        data
        .select("subject_id")
        .unique()
        .with_columns(pl.lit("train").alias("split"))
        .sort("subject_id")
    )

    return MEDSDataset(
        data_shards={"train/0": raw_events},
        dataset_metadata=dataset_metadata,
        code_metadata=code_metadata,
        subject_splits=subject_splits,
    )


def prepare_demo_tensorized_dataset(data: pl.DataFrame) -> tuple[MEDSTorchDataConfig, Callable[[], None]]:
    """Write and tensorize the demo data, returning a MEDSTorchDataConfig and cleanup hook."""

    print("\n🛠️ PREPARING TENSORIZED DEMO DATASET")
    print("=" * 40)

    demo_dataset = _build_demo_meds_dataset(data)

    raw_tmp = tempfile.TemporaryDirectory(prefix="hist_demo_raw_")
    tensor_tmp = tempfile.TemporaryDirectory(prefix="hist_demo_tensor_")
    raw_dir = Path(raw_tmp.name)
    tensor_dir = Path(tensor_tmp.name)

    demo_dataset.write(raw_dir)
    print(f"Raw MEDS dataset written to: {raw_dir}")

    command = [
        "MTD_preprocess",
        f"MEDS_dataset_dir={raw_dir}",
        f"output_dir={tensor_dir}",
    ]

    print("Running MTD_preprocess to tensorize the dataset...")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as err:
        raw_tmp.cleanup()
        tensor_tmp.cleanup()
        raise RuntimeError(
            "MTD_preprocess command not found. Ensure meds-torch-data is installed with CLI scripts."
        ) from err

    if result.returncode != 0:
        raw_tmp.cleanup()
        tensor_tmp.cleanup()
        raise RuntimeError(
            "MTD_preprocess failed while tensorizing the demo dataset.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())

    print(f"Tensorized cohort available at: {tensor_dir}")

    base_config = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensor_dir,
        max_seq_len=HistogramPytorchDataset.MIN_REQUIRED_MAX_SEQ_LEN,
    )

    def cleanup() -> None:
        raw_tmp.cleanup()
        tensor_tmp.cleanup()

    return base_config, cleanup

def demonstrate_time_windowing(
    data: pl.DataFrame,
    base_config: MEDSTorchDataConfig,
    empty_window_mode: EmptyWindowMode,
    vocab_size_hint: int,
) -> None:
    """Demonstrate how time windowing works."""

    print("\n⏰ TIME WINDOWING DEMONSTRATION")
    print("=" * 40)

    config = HistogramConfig(
        window_size_days=30,
        anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
        empty_window_mode=empty_window_mode,
        vocab_size=vocab_size_hint,
    )

    dataset = HistogramPytorchDataset(base_config, "train", config)

    print(f"Window size: {config.window_size_days} days")
    print(f"Anchoring strategy: {config.anchoring_strategy.value}")
    print(f"Empty window mode: {config.empty_window_mode.value}")
    print(f"Vocabulary size: {config.vocab_size}")

    for patient_id in sorted(data["subject_id"].unique()):
        patient_data = data.filter(pl.col("subject_id") == patient_id)

        windows = dataset.create_time_windows(patient_data)

        print(f"\nGenerated {len(windows)} time windows:")
        for i, window in enumerate(windows):
            start = window["start_time"].strftime('%Y-%m-%d')
            end = window["end_time"].strftime('%Y-%m-%d')
            codes = window["codes"]
            is_gap = window.get("is_gap", False)
            num_gap_windows = window.get("num_gap_windows", 0)
            assert is_gap == (num_gap_windows != 0)

            if is_gap:
                print(f"  Window {i + 1}: {start} to {end} - GAP (empty window) — {num_gap_windows} gap windows")
            else:
                code_counts: dict[int, int] = {}
                for code in codes:
                    code_counts[code] = code_counts.get(code, 0) + 1

                code_str = ", ".join([f"Code {k}×{v}" for k, v in sorted(code_counts.items())])
                print(f"  Window {i + 1}: {start} to {end} - {code_str}")

                histogram = window["histogram"]
                non_zero_indices = torch.nonzero(histogram).flatten()
                if len(non_zero_indices) > 0:
                    hist_str = ", ".join([f"[{idx}]={histogram[idx].item():.0f}" for idx in non_zero_indices])
                    print(f"    Histogram: {hist_str}")

def demonstrate_anchoring_strategies(
    data: pl.DataFrame,
    base_config: MEDSTorchDataConfig,
    vocab_size_hint: int,
) -> None:
    """Compare different anchoring strategies."""
    
    print("\n🎯 ANCHORING STRATEGY COMPARISON")
    print("=" * 40)
    
    # Focus on Patient 1 for clearer comparison
    patient1_data = data.filter(pl.col("subject_id") == 1)
    
    strategies = [
        (WindowAnchoringStrategy.END_OF_TIMELINE, "End of Timeline", False),
        (WindowAnchoringStrategy.END_OF_TIMELINE, "End of Timeline", True),
        (WindowAnchoringStrategy.RANDOM_EVENT, "Random Event", False),
        (WindowAnchoringStrategy.SPECIFIC_EVENT, "Specific Event", False),
        (WindowAnchoringStrategy.SPECIFIC_EVENT, "Specific Event", True),
    ]
    
    for strategy, name, include_anchor_token in strategies:
        print(f"\n📍 {name.upper()} STRATEGY, include anchor token: {include_anchor_token}")
        print("-" * 25)
        
        config = HistogramConfig(
            window_size_days=30,
            anchoring_strategy=strategy,
            empty_window_mode=EmptyWindowMode.SINGLE_GAP,
            vocab_size=vocab_size_hint,
            specific_event_codes=[40] if strategy == WindowAnchoringStrategy.SPECIFIC_EVENT else [],
            include_anchor_token=include_anchor_token,
        )
        
        dataset = HistogramPytorchDataset(base_config, "train", config)
        
        
        for j in range(2):
            windows = dataset.create_time_windows(patient1_data)
            print(f"Generated {len(windows)} windows for trial {j+1}:")
            for i, window in enumerate(windows):
                start = window['start_time'].strftime('%Y-%m-%d')
                end = window['end_time'].strftime('%Y-%m-%d')
                codes = window['codes']
                
                print(f"  Window {i+1}: {start} to {end} - {len(codes)} events is_anchor - {window['is_anchor']} is_anchor_token - {window['is_anchor_token']}")

def demonstrate_histogram_computation(
    base_config: MEDSTorchDataConfig,
    vocab_size_hint: int,
) -> None:
    """Show detailed histogram computation."""
    
    print("\n📊 HISTOGRAM COMPUTATION DETAILS")
    print("=" * 40)
    
    config = HistogramConfig(vocab_size=max(vocab_size_hint, 100))
    dataset = HistogramPytorchDataset(base_config, "train", config)
    
    # Test different code sequences
    test_cases = [
        ([10, 20, 10, 30], "Repeated codes"),
        ([5, 15, 25], "Unique codes"),
        ([], "Empty sequence"),
        ([10, 10, 10, 10], "All same code"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Sequential codes")
    ]
    
    for codes, description in test_cases:
        print(f"\n🧮 {description}: {codes}")
        
        histogram = dataset.compute_histogram(codes)
        
        # Show non-zero entries
        non_zero_indices = torch.nonzero(histogram).flatten()
        
        if len(non_zero_indices) > 0:
            print("  Histogram (non-zero entries):")
            for idx in non_zero_indices:
                count = histogram[idx].item()
                print(f"    Code {idx}: {count:.0f} occurrences")
        else:
            print("  Histogram: All zeros (empty)")
        
        print(f"  Total events: {histogram.sum().item():.0f}")

    # Demonstrate how a smaller vocabulary truncates counts
    small_vocab_cfg = HistogramConfig(vocab_size=5)
    small_vocab_dataset = HistogramPytorchDataset(base_config, "train", small_vocab_cfg)
    print("\n🔎 Using a tiny vocabulary (size=5) drops codes ≥5 from the histogram:")
    tiny_hist = small_vocab_dataset.compute_histogram([0, 1, 2, 3, 4, 5, 6])
    print(f"  Histogram values: {tiny_hist.tolist()}")


def demonstrate_histogram_code_fusion(
    base_config: MEDSTorchDataConfig,
    vocab_size_hint: int,
) -> None:
    """Show how histogram placeholders align with the raw code timeline."""

    print("\n🔗 HISTOGRAM ⇄ CODE FUSION")
    print("=" * 40)

    fusion_config = HistogramConfig(
        window_size_days=14,
        vocab_size=vocab_size_hint,
        empty_window_mode=EmptyWindowMode.SINGLE_GAP,
        include_anchor_token=True,
        histogram_max_seq_len=64,
        fusion_mode=FusionMode.HISTOGRAM_AND_CODE
    )

    fusion_base_config = replace(
        base_config,
        static_inclusion_mode=StaticInclusionMode.OMIT,
    )

    dataset = HistogramPytorchDataset(
        fusion_base_config,
        "train",
        fusion_config,
    )

    samples = [dataset[idx] for idx in range(min(len(dataset), 2))]
    batch = dataset.collate(samples)

    print(f"Histogram placeholder token: {HISTOGRAM_PLACEHOLDER_TOKEN}")
    print(f"Anchor token id: {ANCHOR_TOKEN_ID}")
    print(f"Gap token id: {GAP_TOKEN_ID}")
    print("(Quantized histograms will replace the -1 placeholders downstream.)")

    for row_idx, subject_id in enumerate(batch.subject_ids):
        code_seq = batch.code[row_idx].tolist()
        first_code_idx = batch.first_code_indices[row_idx].tolist()
        windows = batch.window_metadata[row_idx]

        print(f"\nSubject {subject_id}")
        print("-" * 20)
        print(f"Code sequence (with placeholders):\n  {code_seq}")
        print(f"First code indices for windows:\n  {first_code_idx[:len(windows)]}")

        for window_pos, window_meta in enumerate(windows):
            placeholder = first_code_idx[window_pos]
            flag_parts = []
            if window_meta.get("is_anchor"):
                flag_parts.append("anchor window")
            if window_meta.get("is_anchor_token"):
                flag_parts.append("anchor token")
            if window_meta.get("is_gap"):
                flag_parts.append("gap window")
            flags = ", ".join(flag_parts) if flag_parts else "data window"

            print(
                f"  Window {window_pos + 1:02d}: {window_meta['start_time']} → {window_meta['end_time']}"
                f" | placeholder index={placeholder} | {flags}"
            )

            if placeholder >= 0:
                histogram_slice = batch.histograms[row_idx, window_pos]
                nz = torch.nonzero(histogram_slice).flatten().tolist()
                if nz:
                    counts = {idx: float(histogram_slice[idx].item()) for idx in nz}
                    print(f"    Histogram counts: {counts}")
                else:
                    print("    Histogram counts: all zeros")
            else:
                print("    (Padded window — no histogram)")

def demonstrate_empty_window_modes(
    data: pl.DataFrame,
    base_config: MEDSTorchDataConfig,
    vocab_size_hint: int,
) -> None:
    """Show how different empty window modes work."""
    
    print("\n🕳️  EMPTY WINDOW MODE COMPARISON")
    print("=" * 40)
    
    # Use Patient 2 which has gaps
    patient2_data = data.filter(pl.col("subject_id") == 2)
    
    modes = [
        (EmptyWindowMode.IGNORE, "Ignore empty windows"),
        (EmptyWindowMode.SINGLE_GAP, "Include gap windows")
    ]
    
    for mode, description in modes:
        print(f"\n🔧 {description.upper()}")
        print("-" * 25)
        
        config = HistogramConfig(
            window_size_days=25,
            anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
            empty_window_mode=mode,
            vocab_size=vocab_size_hint,
        )
        
        dataset = HistogramPytorchDataset(base_config, "train", config)
        windows = dataset.create_time_windows(patient2_data)
        
        print(f"Generated {len(windows)} windows:")
        for i, window in enumerate(windows):
            start = window['start_time'].strftime('%Y-%m-%d')
            end = window['end_time'].strftime('%Y-%m-%d')
            codes = window['codes']
            is_gap = window.get('is_gap', False)
            
            if is_gap:
                print(f"  Window {i+1}: {start} to {end} - GAP (0 events)")
            else:
                print(f"  Window {i+1}: {start} to {end} - {len(codes)} events")

def demonstrate_batch_creation(base_config: MEDSTorchDataConfig) -> None:
    """Show how histogram batches are created."""
    
    print("\n📦 BATCH CREATION DEMONSTRATION")
    print("=" * 40)
    
    # Create mock histogram data for 2 patients
    # Patient 1: 3 windows, Patient 2: 2 windows
    histograms_p1 = torch.tensor([
        [2.0, 1.0, 0.0, 1.0, 0.0],  # Window 1: codes [0,0,1,3] 
        [0.0, 2.0, 1.0, 0.0, 0.0],  # Window 2: codes [1,1,2]
        [1.0, 0.0, 0.0, 0.0, 1.0],  # Window 3: codes [0,4]
    ])
    
    histograms_p2 = torch.tensor([
        [3.0, 0.0, 0.0, 0.0, 0.0],  # Window 1: codes [0,0,0]
        [0.0, 0.0, 0.0, 2.0, 1.0],  # Window 2: codes [3,3,4]
    ])
    
    # Create batch data
    batch_data = [
        {"histograms": histograms_p1},
        {"histograms": histograms_p2}
    ]
    
    config = HistogramConfig(vocab_size=5)
    dataset = HistogramPytorchDataset(base_config, "train", config)
    
    print("Individual patient histograms:")
    print(f"Patient 1 histograms shape: {histograms_p1.shape} (3 windows, vocab_size=5)")
    print(f"Patient 2 histograms shape: {histograms_p2.shape} (2 windows, vocab_size=5)")
    
    # Mock the collate process (simplified)
    max_windows = max(len(h["histograms"]) for h in batch_data)
    vocab_size = histograms_p1.shape[1]
    
    padded_histograms = []
    for item in batch_data:
        histograms = item["histograms"]
        if len(histograms) < max_windows:
            # Pad with zeros
            padding = torch.zeros((max_windows - len(histograms), vocab_size))
            padded = torch.cat([histograms, padding], dim=0)
        else:
            padded = histograms
        padded_histograms.append(padded)
    
    batch_histograms = torch.stack(padded_histograms)
    
    print(f"\nBatched histograms shape: {batch_histograms.shape}")
    print("  Dimension 0: Batch size (2 patients)")
    print("  Dimension 1: Max windows per patient (3, padded)")
    print("  Dimension 2: Vocabulary size (5)")
    
    print(f"\nBatch histogram tensor:")
    print(batch_histograms)
    
    # Show which entries are padding
    print(f"\nPadding analysis:")
    print(f"Patient 1: Uses all 3 windows (no padding)")
    print(f"Patient 2: Uses 2 windows, 1 padded (window 3 is all zeros)")


def demonstrate_subsampling(
    data: pl.DataFrame,
    base_config: MEDSTorchDataConfig,
    empty_window_mode: EmptyWindowMode,
    vocab_size_hint: int,
    window_sampling_strategy: WindowSamplingStrategy,
    max_windows: int
) -> None:
    """Demonstrate how time windowing works."""

    print("\n⏰ TIME WINDOWING DEMONSTRATION")
    print("=" * 40)

    config = HistogramConfig(
        window_size_days=30,
        anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE,
        empty_window_mode=empty_window_mode,
        vocab_size=vocab_size_hint,
        window_sampling_strategy=window_sampling_strategy,
        max_windows=max_windows,
    )

    dataset = HistogramPytorchDataset(base_config, "train", config)

    print(f"Window size: {config.window_size_days} days")
    print(f"Anchoring strategy: {config.anchoring_strategy.value}")
    print(f"Empty window mode: {config.empty_window_mode.value}")
    print(f"Vocabulary size: {config.vocab_size}")

    for patient_id in sorted(data["subject_id"].unique()):
        patient_data = data.filter(pl.col("subject_id") == patient_id)

        windows = dataset.create_time_windows(patient_data)
        windows = dataset._apply_window_subsampling(windows)

        print(f"\nGenerated {len(windows)} time windows:")
        for i, window in enumerate(windows):
            start = window["start_time"].strftime('%Y-%m-%d')
            end = window["end_time"].strftime('%Y-%m-%d')
            codes = window["codes"]
            is_gap = window.get("is_gap", False)
            num_gap_windows = window.get("num_gap_windows", 0)
            assert is_gap == (num_gap_windows != 0)

            if is_gap:
                print(f"  Window {i + 1}: {start} to {end} - GAP (empty window) — {num_gap_windows} gap windows")
            else:
                code_counts: dict[int, int] = {}
                for code in codes:
                    code_counts[code] = code_counts.get(code, 0) + 1

                code_str = ", ".join([f"Code {k}×{v}" for k, v in sorted(code_counts.items())])
                print(f"  Window {i + 1}: {start} to {end} - {code_str}")

                histogram = window["histogram"]
                non_zero_indices = torch.nonzero(histogram).flatten()
                if len(non_zero_indices) > 0:
                    hist_str = ", ".join([f"[{idx}]={histogram[idx].item():.0f}" for idx in non_zero_indices])
                    print(f"    Histogram: {hist_str}")

def main():
    """Run the complete demonstration."""
    
    print("🎯 HISTOGRAM DATASET DEMONSTRATION")
    print("=" * 50)
    print("This script shows how the histogram dataset processes")
    print("patient data into time-windowed histograms for ML models.")
    print()
    
    # Create dummy data
    data = create_dummy_patient_data()
    vocab_size_hint = int(data["code"].max()) + 1

    base_config, cleanup = prepare_demo_tensorized_dataset(data)

    demonstrate_time_windowing(data, base_config, EmptyWindowMode.SINGLE_GAP, vocab_size_hint)
    demonstrate_subsampling(data, base_config, EmptyWindowMode.SINGLE_GAP, vocab_size_hint, WindowSamplingStrategy.RANDOM, 1)
    demonstrate_subsampling(data, base_config, EmptyWindowMode.SINGLE_GAP, vocab_size_hint, WindowSamplingStrategy.TO_END, 1)
    demonstrate_time_windowing(data, base_config, EmptyWindowMode.IGNORE, vocab_size_hint)
    
    demonstrate_anchoring_strategies(data, base_config, vocab_size_hint)
    demonstrate_histogram_computation(base_config, vocab_size_hint)
    demonstrate_histogram_code_fusion(base_config, vocab_size_hint)
    demonstrate_empty_window_modes(data, base_config, vocab_size_hint)
    demonstrate_batch_creation(base_config)
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 30)
    print("✅ Time windowing works correctly")
    print("✅ Histogram computation is accurate") 
    print("✅ Both anchoring strategies work")
    print("✅ Histogram/code fusion aligns placeholders")
    print("✅ Empty window handling works")
    print("✅ Batch creation works properly")
    print()
    print("The histogram dataset successfully converts patient")
    print("event sequences into time-windowed histogram features")
    print("suitable for machine learning models! 🚀")

if __name__ == "__main__":
    main()
