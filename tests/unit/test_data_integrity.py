"""Integrity, physical range, and reproducibility tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mlops_experiment_tracking.data.integrity import (
    DataIntegrityError,
    digest,
    verify_prepared_dataset,
    verify_raw_dataset,
)
from mlops_experiment_tracking.data.profiling import build_profile
from mlops_experiment_tracking.data.schema import (
    EXPECTED_COLUMNS,
    DataValidationError,
    validate_dataset,
)
from mlops_experiment_tracking.data.splitting import split_dataset


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    records = []

    for index in range(100):
        failure = int(index % 10 == 0)
        records.append(
            (
                index + 1,
                f"L{index + 1}",
                "L",
                300.0,
                310.0,
                1500,
                40.0,
                100,
                failure,
                failure,
                0,
                0,
                0,
                0,
            )
        )

    return pd.DataFrame(records, columns=EXPECTED_COLUMNS)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Air temperature [K]", float("inf")),
        ("Torque [Nm]", float("-inf")),
        ("Rotational speed [rpm]", -100),
        ("Tool wear [min]", -1),
        ("Process temperature [K]", 0),
    ],
)
def test_rejects_invalid_physical_values(
    sample_frame: pd.DataFrame,
    column: str,
    value: float,
) -> None:
    sample_frame.loc[0, column] = value

    with pytest.raises(DataValidationError):
        validate_dataset(sample_frame)


def write_raw(tmp_path: Path, frame: pd.DataFrame) -> Path:
    raw = tmp_path / "raw.csv"
    content = frame.to_csv(index=False).encode("utf-8")
    raw.write_bytes(content)

    raw.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "sha256": digest(content),
                "rows": len(frame),
                "columns": len(frame.columns),
            }
        ),
        encoding="utf-8",
    )
    return raw


def write_partitions(raw: Path, output: Path) -> None:
    output.mkdir()

    frame = pd.read_csv(raw)
    splits = split_dataset(frame)

    partitions = {}
    for name, data in {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }.items():
        content = data.to_csv(index=False).encode("utf-8")
        (output / f"{name}.csv").write_bytes(content)

        indices = ",".join(map(str, data.index)).encode("utf-8")
        partitions[name] = {
            "rows": len(data),
            "failures": int(data["Machine failure"].sum()),
            "failure_rate": round(float(data["Machine failure"].mean()), 6),
            "sha256": digest(content),
            "source_row_indices_sha256": digest(indices),
        }

    (output / "profile.json").write_text(json.dumps(build_profile(splits.train)), encoding="utf-8")

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": digest(raw.read_bytes()),
                "random_state": 42,
                "split_strategy": "stratified_random_60_20_20",
                "target": "Machine failure",
                "partitions": partitions,
            }
        ),
        encoding="utf-8",
    )


def test_raw_integrity_passes(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    raw = write_raw(tmp_path, sample_frame)
    assert verify_raw_dataset(raw) == digest(raw.read_bytes())


def test_detects_raw_tampering(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    raw = write_raw(tmp_path, sample_frame)
    raw.write_bytes(raw.read_bytes() + b"\n")

    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        verify_raw_dataset(raw)


def test_prepared_integrity_passes(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    raw = write_raw(tmp_path, sample_frame)
    output = tmp_path / "processed"
    write_partitions(raw, output)

    verify_prepared_dataset(raw, output)


def test_detects_processed_tampering(tmp_path: Path, sample_frame: pd.DataFrame) -> None:
    raw = write_raw(tmp_path, sample_frame)
    output = tmp_path / "processed"
    write_partitions(raw, output)

    partition = output / "train.csv"
    partition.write_bytes(partition.read_bytes() + b"\n")

    with pytest.raises(DataIntegrityError, match="Partition differs"):
        verify_prepared_dataset(raw, output)
