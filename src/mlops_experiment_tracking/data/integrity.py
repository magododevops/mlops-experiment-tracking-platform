"""Integrity and reproducibility verification for local ML datasets."""

import hashlib
import json
from pathlib import Path
from typing import Any

from mlops_experiment_tracking.data.ingestion import load_dataset
from mlops_experiment_tracking.data.profiling import build_profile
from mlops_experiment_tracking.data.schema import TARGET_COLUMN
from mlops_experiment_tracking.data.splitting import RANDOM_STATE, split_dataset


class DataIntegrityError(ValueError):
    """Raised when dataset files or manifests do not match."""


def digest(content: bytes) -> str:
    """Return a SHA-256 hexadecimal digest."""
    return hashlib.sha256(content).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest and ensure it contains an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataIntegrityError(f"Cannot read manifest: {path}") from error

    if not isinstance(data, dict):
        raise DataIntegrityError(f"Invalid manifest structure: {path}")
    return data


def verify_raw_dataset(path: Path) -> str:
    """Verify the original CSV against its provenance manifest."""
    manifest = read_manifest(path.with_suffix(".manifest.json"))

    try:
        content = path.read_bytes()
    except OSError as error:
        raise DataIntegrityError(f"Cannot read dataset: {path}") from error

    actual = digest(content)

    if actual != manifest.get("sha256"):
        raise DataIntegrityError("Raw dataset SHA-256 mismatch")

    frame = load_dataset(path)

    if manifest.get("rows") != len(frame):
        raise DataIntegrityError("Raw dataset row count mismatch")

    if manifest.get("columns") != len(frame.columns):
        raise DataIntegrityError("Raw dataset column count mismatch")

    return actual


def verify_prepared_dataset(raw_path: Path, output: Path) -> None:
    """Regenerate expected splits and compare persisted outputs."""
    raw_hash = verify_raw_dataset(raw_path)
    manifest = read_manifest(output / "manifest.json")

    if manifest.get("source_sha256") != raw_hash:
        raise DataIntegrityError("Processed source SHA-256 mismatch")

    if manifest.get("random_state") != RANDOM_STATE:
        raise DataIntegrityError("Unexpected random seed")

    if manifest.get("split_strategy") != "stratified_random_60_20_20":
        raise DataIntegrityError("Unexpected splitting strategy")

    if manifest.get("target") != TARGET_COLUMN:
        raise DataIntegrityError("Unexpected prediction target")

    splits = split_dataset(load_dataset(raw_path), random_state=RANDOM_STATE)

    expected = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(expected):
        raise DataIntegrityError("Invalid partition manifest")

    for name, frame in expected.items():
        metadata = partitions[name]
        if not isinstance(metadata, dict):
            raise DataIntegrityError(f"Invalid metadata: {name}")

        path = output / f"{name}.csv"

        try:
            actual_content = path.read_bytes()
        except OSError as error:
            raise DataIntegrityError(f"Missing partition: {name}") from error

        expected_content = frame.to_csv(index=False).encode("utf-8")
        expected_hash = digest(expected_content)

        if actual_content != expected_content:
            raise DataIntegrityError(f"Partition differs from source: {name}")

        expected_indices_hash = digest(",".join(map(str, frame.index)).encode("utf-8"))

        expected_metadata = {
            "rows": len(frame),
            "failures": int(frame[TARGET_COLUMN].sum()),
            "failure_rate": round(float(frame[TARGET_COLUMN].mean()), 6),
            "sha256": expected_hash,
            "source_row_indices_sha256": expected_indices_hash,
        }

        if metadata != expected_metadata:
            raise DataIntegrityError(f"Partition manifest mismatch: {name}")

    expected_profile = build_profile(splits.train)
    actual_profile = read_manifest(output / "profile.json")

    if actual_profile != expected_profile:
        raise DataIntegrityError("Training profile mismatch")
