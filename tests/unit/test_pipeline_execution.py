"""Pure unit tests for grouped training executions."""

from copy import deepcopy

import pytest

from mlops_experiment_tracking.pipelines.execution import (
    extract_run_ids,
    validate_git_commit,
    validate_run_records,
)

HASH = "a" * 64
IDS = ("1" * 32, "2" * 32, "3" * 32)


@pytest.fixture
def records() -> list[dict]:
    names = (
        "dummy-baseline",
        "logistic-default",
        "logistic-selected",
    )

    return [
        {
            "name": name,
            "run_id": run_id,
            "status": "FINISHED",
            "dataset_sha256": HASH,
            "validation_records": 2000,
            "metrics": {
                "precision": 0.4,
                "recall": 0.44,
                "f1": 0.42,
            },
        }
        for name, run_id in zip(names, IDS, strict=True)
    ]


def test_extracts_three_run_ids() -> None:
    output = "\n".join(f"Run ID: {run_id}" for run_id in IDS)
    assert extract_run_ids(output) == IDS


def test_rejects_missing_run_id() -> None:
    with pytest.raises(ValueError, match="Expected"):
        extract_run_ids(f"Run ID: {IDS[0]}")


def test_rejects_duplicate_run_id() -> None:
    output = "\n".join(f"Run ID: {run_id}" for run_id in (IDS[0], IDS[0], IDS[2]))
    with pytest.raises(ValueError, match="Duplicate"):
        extract_run_ids(output)


def test_accepts_valid_run_records(records: list[dict]) -> None:
    validate_run_records(records, HASH)


def test_rejects_different_dataset(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[1]["dataset_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="provenance"):
        validate_run_records(changed, HASH)


def test_rejects_unfinished_run(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[0]["status"] = "FAILED"

    with pytest.raises(ValueError, match="Unfinished"):
        validate_run_records(changed, HASH)


def test_rejects_wrong_run_order(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed.reverse()

    with pytest.raises(ValueError, match="order"):
        validate_run_records(changed, HASH)


def test_rejects_invalid_f1(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[2]["metrics"]["f1"] = 1.5

    with pytest.raises(ValueError, match="F1"):
        validate_run_records(changed, HASH)


def test_accepts_full_git_commit_sha() -> None:
    assert validate_git_commit("a" * 40) == "a" * 40


@pytest.mark.parametrize(
    "value",
    ["", "abc1234", "A" * 40, "z" * 40, "a" * 41],
)
def test_rejects_invalid_git_commit(value: str) -> None:
    with pytest.raises(ValueError, match="Git commit SHA"):
        validate_git_commit(value)
