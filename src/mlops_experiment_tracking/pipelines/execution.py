"""Validation rules for grouped MLflow training executions."""

import re
from collections.abc import Mapping, Sequence

EXPECTED_RUN_NAMES = (
    "dummy-baseline",
    "logistic-default",
    "logistic-selected",
)

RUN_ID_PATTERN = re.compile(r"^Run ID:\s*([a-f0-9]{32})\s*$", re.MULTILINE)


def validate_git_commit(value: str) -> str:
    """Validate the Git revision supplied by a pipeline launcher."""
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Expected a full lowercase Git commit SHA")

    return value


def extract_run_ids(output: str) -> tuple[str, ...]:
    """Extract the exact three run IDs emitted by our training script."""
    ids = tuple(RUN_ID_PATTERN.findall(output))

    if len(ids) != len(EXPECTED_RUN_NAMES):
        raise ValueError(f"Expected {len(EXPECTED_RUN_NAMES)} runs, got {len(ids)}")

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate MLflow Run IDs")

    return ids


def validate_run_records(
    records: Sequence[Mapping],
    expected_sha256: str,
) -> None:
    """Ensure runs are comparable before grouping them."""
    if len(records) != len(EXPECTED_RUN_NAMES):
        raise ValueError("Unexpected number of training runs")

    names = tuple(record["name"] for record in records)
    if names != EXPECTED_RUN_NAMES:
        raise ValueError("Unexpected training run names or order")

    run_ids = [record["run_id"] for record in records]

    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate MLflow Run IDs")

    for record in records:
        if record["status"] != "FINISHED":
            raise ValueError(f"Unfinished run: {record['name']}")

        if record["dataset_sha256"] != expected_sha256:
            raise ValueError("Training dataset provenance mismatch")

        if record["validation_records"] != 2000:
            raise ValueError("Unexpected validation dataset size")

        if not 0 <= record["metrics"]["f1"] <= 1:
            raise ValueError("Invalid F1 score")

        if not 0 <= record["metrics"]["recall"] <= 1:
            raise ValueError("Invalid recall")

        if not 0 <= record["metrics"]["precision"] <= 1:
            raise ValueError("Invalid precision")
