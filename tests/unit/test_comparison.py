"""Tests for MLflow comparison rules without network dependencies."""

from copy import deepcopy

import pytest

from mlops_experiment_tracking.tracking.comparison import (
    build_markdown,
    validate_comparison,
)


@pytest.fixture
def records() -> list[dict]:
    def make(
        name: str,
        model: str,
        threshold: float,
        ap: float,
    ) -> dict:
        return {
            "name": name,
            "run_id": f"test-{name}",
            "model": model,
            "threshold": threshold,
            "partition": "validation",
            "source_sha256": "same-hash",
            "validation_rows": 2000,
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.40,
                "recall": 0.44,
                "f1": 0.42,
                "average_precision": ap,
                "true_negative": 1900,
                "false_positive": 32,
                "false_negative": 36,
                "true_positive": 32,
            },
        }

    return [
        make("dummy-baseline", "dummy", 0.50, 0.034),
        make("logistic-default", "logistic_regression", 0.50, 0.405),
        make("logistic-selected", "logistic_regression", 0.20, 0.405),
    ]


def test_accepts_matching_runs(records: list[dict]) -> None:
    validate_comparison(records)


def test_generates_markdown(records: list[dict]) -> None:
    content = build_markdown(records)
    assert "logistic-selected" in content
    assert "test-logistic-selected" in content
    assert "## Run provenance" in content


def test_rejects_different_datasets(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[1]["source_sha256"] = "different"

    with pytest.raises(ValueError, match="dataset hashes"):
        validate_comparison(changed)


def test_rejects_test_partition(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[2]["partition"] = "test"

    with pytest.raises(ValueError, match="validation"):
        validate_comparison(changed)


def test_rejects_missing_run(records: list[dict]) -> None:
    with pytest.raises(ValueError, match="three"):
        validate_comparison(records[:2])


def test_rejects_inconsistent_counts(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[0]["metrics"]["true_negative"] = 1800

    with pytest.raises(ValueError, match="Confusion matrix"):
        validate_comparison(changed)


def test_rejects_inconsistent_average_precision(
    records: list[dict],
) -> None:
    changed = deepcopy(records)
    changed[2]["metrics"]["average_precision"] = 0.100

    with pytest.raises(ValueError, match="ranking metrics"):
        validate_comparison(changed)


def test_rejects_invalid_threshold(records: list[dict]) -> None:
    changed = deepcopy(records)
    changed[2]["threshold"] = 1.50

    with pytest.raises(ValueError, match="threshold"):
        validate_comparison(changed)
