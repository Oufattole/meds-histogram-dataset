"""Tests the full, multi-stage pre-processing pipeline in parallel mode."""

import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from . import PREPROCESS_SCRIPT, assert_df_equal, check_NRT_output
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_SCHEMAS

PARALLEL_STAGE_RUNNER_YAML = """
parallelize:
  n_workers: 2
  launcher: "joblib"
"""


@pytest.mark.parallelized
def test_preprocess_parallel(simple_static_MEDS: Path):
    with tempfile.TemporaryDirectory() as root_dir:
        runner_fp = Path(root_dir) / "stage_runner.yaml"
        runner_fp.write_text(PARALLEL_STAGE_RUNNER_YAML)

        cohort_dir = Path(root_dir) / "cohort"
        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={simple_static_MEDS!s}",
            f"output_dir={cohort_dir!s}",
            f"stage_runner_fp={runner_fp!s}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        cohort_dir = Path(cohort_dir)

        for shard, want_schema in WANT_SCHEMAS.items():
            fp = cohort_dir / f"tokenization/{shard}.parquet"
            err_str = f"Expected output file {fp} not found. Directory contents:\n" + "\n".join(
                f"  - {f.relative_to(cohort_dir)}" for f in cohort_dir.rglob("*")
            )

            assert fp.exists(), err_str
            got_schema = pl.read_parquet(fp)
            assert_df_equal(got_schema, want_schema, check_column_order=False)

        for shard, want_NRT in WANT_NRTS.items():
            fp = cohort_dir / f"data/{shard}"
            err_str = f"Expected output file {fp} not found. Directory contents:\n" + "\n".join(
                f"  - {f.relative_to(cohort_dir)}" for f in cohort_dir.rglob("*")
            )

            assert fp.exists(), err_str
            check_NRT_output(fp, want_NRT, f"{shard} NRT differs!")
