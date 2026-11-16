"""Tests for validation-only model selection policy."""

import pytest

from mlops_experiment_tracking.training.selection import (
    NoEligibleCandidateError,
    SelectionPolicy,
    select_candidate,
)


def candidate(
    threshold: float,
    recall: float,
    f1: float,
    fp: int,
) -> dict:
    return {
        "threshold": threshold,
        "recall": recall,
        "f1": f1,
        "false_positive": fp,
    }


def test_selects_only_eligible_candidate() -> None:
    candidates = [
        candidate(0.05, 0.72, 0.27, 249),
        candidate(0.10, 0.60, 0.37, 114),
        candidate(0.20, 0.44, 0.42, 45),
        candidate(0.30, 0.31, 0.38, 21),
        candidate(0.50, 0.19, 0.31, 4),
    ]

    selected = select_candidate(candidates, SelectionPolicy())

    assert selected["threshold"] == 0.20


def test_prefers_highest_f1_among_eligible() -> None:
    candidates = [
        candidate(0.10, 0.70, 0.50, 30),
        candidate(0.20, 0.60, 0.60, 20),
    ]

    selected = select_candidate(candidates, SelectionPolicy())

    assert selected["threshold"] == 0.20


def test_tie_breakers_are_deterministic() -> None:
    candidates = [
        candidate(0.30, 0.60, 0.50, 20),
        candidate(0.20, 0.70, 0.50, 20),
        candidate(0.10, 0.70, 0.50, 20),
    ]

    selected = select_candidate(candidates, SelectionPolicy())

    assert selected["threshold"] == 0.10


def test_rejects_when_no_candidate_qualifies() -> None:
    candidates = [candidate(0.50, 0.10, 0.20, 5)]

    with pytest.raises(NoEligibleCandidateError):
        select_candidate(candidates, SelectionPolicy())


def test_rejects_empty_candidates() -> None:
    with pytest.raises(NoEligibleCandidateError):
        select_candidate([], SelectionPolicy())


def test_rejects_duplicate_thresholds() -> None:
    candidates = [
        candidate(0.20, 0.50, 0.40, 20),
        candidate(0.20, 0.60, 0.50, 10),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        select_candidate(candidates, SelectionPolicy())


@pytest.mark.parametrize(
    ("threshold", "recall", "f1", "fp"),
    [
        (0.20, float("nan"), 0.50, 10),
        (0.20, 1.50, 0.50, 10),
        (0.20, 0.60, float("inf"), 10),
        (0.20, 0.60, 0.50, -1),
        (1.50, 0.60, 0.50, 10),
    ],
)
def test_rejects_invalid_metrics(
    threshold: float,
    recall: float,
    f1: float,
    fp: int,
) -> None:
    with pytest.raises(ValueError):
        select_candidate(
            [candidate(threshold, recall, f1, fp)],
            SelectionPolicy(),
        )


def test_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError):
        SelectionPolicy(min_recall=1.5)

    with pytest.raises(ValueError):
        SelectionPolicy(max_false_positives=-1)
