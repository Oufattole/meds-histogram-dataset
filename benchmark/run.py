import json
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rootutils
import torch
from mixins import MemTrackableMixin, TimeableMixin, add_mixin

from meds_torchdata.pytorch_dataset import MEDSPytorchDataset, MEDSTorchDataConfig
from meds_torchdata.histogram_dataset import HistogramPytorchDataset, HistogramConfig, WindowAnchoringStrategy, EmptyWindowMode

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

OUTPUT_DIR = root / "benchmark" / "outputs"

CNT_PREFIX = "Count: "


def tensor_size(a: torch.Tensor) -> int:
    return sys.getsizeof(a) + torch.numel(a) * a.element_size()


def to_val(k: str, v: Any) -> dict:
    match v:
        case list() as samples:
            vs = [to_val(k, val) for val in samples]
            cnt = 0
            sum_val = 0
            sum_val_sq = 0

            for v in vs:
                if "value" not in v:
                    raise ValueError(f"Value not found in {v}")

                if "extra" in v and v["extra"].startswith(CNT_PREFIX):
                    new_cnt = int(v["extra"][len(CNT_PREFIX) :])
                    cnt += new_cnt
                    sum_val += new_cnt * v["value"]

                    if "range" not in v:
                        raise ValueError(f"Range not found in {v} but count is")
                    v_stdev = v["range"]
                    v_var = v_stdev**2
                    v_sum_sq = (v_var + (v["value"] ** 2)) * new_cnt
                    sum_val_sq += v_sum_sq
                else:
                    cnt += 1
                    sum_val += v["value"]
                    sum_val_sq += v["value"] ** 2

            mean_val = sum_val / cnt
            std_val = np.sqrt((sum_val_sq - (sum_val**2) / cnt) / (cnt - 1))
            return {"value": mean_val, "range": std_val, "extra": f"{CNT_PREFIX}{cnt}"}
        case tuple() as stats if len(stats) == 3:
            mean_val, cnt, std_val = stats
            return {"value": mean_val, "range": std_val, "extra": f"{CNT_PREFIX}{cnt}"}
        case dict() as mem_stats if "metadata" in mem_stats and "peak_memory" in mem_stats["metadata"]:
            return {"value": mem_stats["metadata"]["peak_memory"]}
        case int() | float() as val:
            return {"value": val}
        case timedelta() as duration:
            return {"value": duration.total_seconds()}
        case _:
            raise ValueError(f"Unsupported type for {k}: {v}")


def summarize_output(out: dict) -> list[dict]:
    """Summarizes the output into a minimal format suitable for monitoring with github-actions-benchmark."""

    summary = []

    for single_key, name, unit in [
        ("epoch_durations", "Usage/Duration/Epoch", "seconds"),
    ]:
        summary.append({"name": name, "unit": unit, **to_val(single_key, out[single_key])})

    for nested_key, name, unit in [
        ("memory_stats", "Usage/Memory", "bytes"),
        ("duration_stats", "Usage/Duration", "seconds"),
        ("batch_sizes", "Outputs/BatchSize", "bytes"),
    ]:
        for k, v in out[nested_key].items():
            summary.append({"name": f"{name}/{k}", "unit": unit, **to_val(k, v)})

    return summary


def benchmark(dataset, batch_size: int, num_epochs: int = 1) -> tuple[dict[str, list[int]], list[timedelta]]:
    torch.manual_seed(1)

    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    sizes = defaultdict(list)
    epoch_durations = []

    with dataset._track_memory_as("benchmark"):
        for _epoch in range(num_epochs):
            epoch_start = datetime.now(tz=UTC)
            for batch in dataloader:
                for k, v in batch.items():
                    sizes[k].append(tensor_size(v))
            epoch_durations.append(datetime.now(tz=UTC) - epoch_start)

    return sizes, epoch_durations


@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("num_epochs", [5])
def test_profile(benchmark_dataset: Path, batch_size: int, max_seq_len: int, num_epochs: int):
    methods_to_track = ["__getitem__", "collate"]

    TrackableDataset: type[MEDSPytorchDataset] = add_mixin(  # noqa: N806
        add_mixin(
            MEDSPytorchDataset,
            TimeableMixin,
            dict.fromkeys(methods_to_track, TimeableMixin.TimeAs),
        ),
        MemTrackableMixin,
        {},
    )

    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=benchmark_dataset,
        max_seq_len=max_seq_len,
    )

    D = TrackableDataset(config, split="train")
    batch_sizes, epoch_durations = benchmark(D, batch_size=batch_size, num_epochs=num_epochs)

    out = {}

    out["batch_sizes"] = batch_sizes
    out["epoch_durations"] = epoch_durations
    out["memory_stats"] = D._memory_stats
    out["duration_stats"] = D._duration_stats

    final_json_output = summarize_output(out)

    output_fp = OUTPUT_DIR / f"output_{batch_size}_{max_seq_len}_{num_epochs}.json"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text(json.dumps(final_json_output))


@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("window_size_days", [30])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("num_epochs", [5])
@pytest.mark.parametrize("anchoring_strategy", ["end_of_timeline", "random_event"])
def test_profile_histogram(benchmark_dataset: Path, batch_size: int, window_size_days: int, vocab_size: int, num_epochs: int, anchoring_strategy: str):
    """Benchmark the histogram dataset implementation."""
    methods_to_track = ["__getitem__", "collate", "create_time_windows", "compute_histogram"]

    # Create trackable histogram dataset class with mixins
    TrackableHistogramDataset: type[HistogramPytorchDataset] = add_mixin(  # noqa: N806
        add_mixin(
            HistogramPytorchDataset,
            TimeableMixin,
            dict.fromkeys(methods_to_track, TimeableMixin.TimeAs),
        ),
        MemTrackableMixin,
        {},
    )

    # Create base config for the underlying MEDSPytorchDataset
    base_config = MEDSTorchDataConfig(
        tensorized_cohort_dir=benchmark_dataset,
        max_seq_len=1_000_000,
    )
    
    # Create histogram-specific configuration
    histogram_config = HistogramConfig(
        window_size_days=window_size_days,
        anchoring_strategy=WindowAnchoringStrategy.END_OF_TIMELINE if anchoring_strategy == "end_of_timeline" else WindowAnchoringStrategy.RANDOM_EVENT,
        empty_window_mode=EmptyWindowMode.SINGLE_GAP,
        vocab_size=vocab_size
    )

    # Create histogram dataset instance
    D = TrackableHistogramDataset(base_config, split="train", histogram_config=histogram_config)
    
    # Run benchmark
    batch_sizes, epoch_durations = benchmark(D, batch_size=batch_size, num_epochs=num_epochs)

    # Collect output statistics
    out = {}
    out["batch_sizes"] = batch_sizes
    out["epoch_durations"] = epoch_durations
    out["memory_stats"] = D._memory_stats
    out["duration_stats"] = D._duration_stats
    
    # Add histogram-specific metrics
    out["histogram_config"] = {
        "window_size_days": window_size_days,
        "vocab_size": vocab_size,
        "anchoring_strategy": histogram_config.anchoring_strategy.value,
        "empty_window_mode": histogram_config.empty_window_mode.value
    }

    final_json_output = summarize_output(out)

    # Output to histogram-specific file
    output_fp = OUTPUT_DIR / f"histogram_output_{batch_size}_{window_size_days}_{vocab_size}_{num_epochs}.json"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text(json.dumps(final_json_output))
