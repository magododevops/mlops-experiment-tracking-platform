"""Tests for candidate integrity before final evaluation."""

import pytest

from mlops_experiment_tracking.training.finalization import (
    CandidateIntegrityError,
    verify_selection,
)
from mlops_experiment_tracking.training.selection import SelectionPolicy


@pytest.fixture
def candidates() -> list[dict]:
    return [
        {
            "threshold": 0.10,
            "recall": 0.60,
            "f1": 0.37,
            "false_positive": 114,
        },
        {
            "threshold": 0.20,
            "recall": 0.44,
            "f1": 0.42,
            "false_positive": 45,
        },
    ]


@pytest.fixture
def report(candidates: list[dict]) -> dict:
    policy = SelectionPolicy()

    return {
        "model": "logistic_regression",
        "partition": "validation",
        "status": "candidate_selected",
        "source_sha256": "example-source-hash",
        "validation_rows": 2000,
        "policy": {
            "min_recall": policy.min_recall,
            "max_false_positives": policy.max_false_positives,
        },
        "selection_criteria": [
            "highest_f1",
            "highest_recall",
            "lowest_threshold",
        ],
        "candidate_count": len(candidates),
        "selected": candidates[1],
    }


def test_accepts_unchanged_candidate(
    report: dict,
    candidates: list[dict],
) -> None:
    selected = verify_selection(report, candidates, "example-source-hash", 2000)

    assert selected["threshold"] == 0.20


def test_rejects_modified_threshold(
    report: dict,
    candidates: list[dict],
) -> None:
    report["selected"] = {
        **report["selected"],
        "threshold": 0.10,
    }

    with pytest.raises(CandidateIntegrityError):
        verify_selection(report, candidates, "example-source-hash", 2000)


def test_rejects_modified_dataset_hash(
    report: dict,
    candidates: list[dict],
) -> None:
    with pytest.raises(
        CandidateIntegrityError,
        match="source_sha256",
    ):
        verify_selection(report, candidates, "different-hash", 2000)


def test_rejects_modified_policy(
    report: dict,
    candidates: list[dict],
) -> None:
    report["policy"]["min_recall"] = 0.10

    with pytest.raises(CandidateIntegrityError, match="policy"):
        verify_selection(report, candidates, "example-source-hash", 2000)


def test_rejects_modified_validation_size(
    report: dict,
    candidates: list[dict],
) -> None:
    with pytest.raises(
        CandidateIntegrityError,
        match="validation_rows",
    ):
        verify_selection(report, candidates, "example-source-hash", 1999)
