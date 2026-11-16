"""Unit tests for model promotion eligibility."""

from dataclasses import replace

import pytest

from mlops_experiment_tracking.registry.promotion import (
    evaluate_promotion,
)
from mlops_experiment_tracking.training.evaluation import (
    EvaluationResult,
)

HASH = "a" * 64


@pytest.fixture
def metrics() -> EvaluationResult:
    return EvaluationResult(
        accuracy=0.96,
        precision=0.4,
        recall=0.44,
        f1=0.42,
        average_precision=0.405,
        true_negative=1887,
        false_positive=45,
        false_negative=38,
        true_positive=30,
    )


@pytest.fixture
def tags() -> dict[str, str]:
    return {
        "lifecycle": "candidate",
        "approval": "not_approved",
        "model_algorithm": "logistic_regression",
        "dataset_sha256": HASH,
        "classification_threshold": "0.2",
        "validation_recall": "0.44",
        "validation_f1": "0.42",
    }


def check(
    metrics: EvaluationResult,
    tags: dict[str, str],
):
    return evaluate_promotion(
        metrics,
        tags,
        expected_sha256=HASH,
        expected_threshold=0.2,
    )


def test_accepts_eligible_candidate(metrics, tags) -> None:
    decision = check(metrics, tags)
    assert decision.eligible
    assert decision.reasons == ()


def test_rejects_low_recall(metrics, tags) -> None:
    changed = replace(metrics, recall=0.2)
    tags["validation_recall"] = "0.2"
    decision = check(changed, tags)
    assert not decision.eligible
    assert "Recall below required minimum" in decision.reasons


def test_rejects_excessive_false_positives(metrics, tags) -> None:
    changed = replace(metrics, false_positive=51)
    decision = check(changed, tags)
    assert not decision.eligible
    assert "Too many false positives" in decision.reasons


def test_rejects_wrong_dataset(metrics, tags) -> None:
    tags["dataset_sha256"] = "b" * 64
    assert not check(metrics, tags).eligible


def test_rejects_changed_threshold(metrics, tags) -> None:
    tags["classification_threshold"] = "0.3"
    assert not check(metrics, tags).eligible


def test_rejects_approved_or_unknown_lifecycle(metrics, tags) -> None:
    tags["lifecycle"] = "production"
    assert not check(metrics, tags).eligible


def test_rejects_changed_f1(metrics, tags) -> None:
    tags["validation_f1"] = "0.9"
    assert not check(metrics, tags).eligible


@pytest.mark.parametrize("value", ["nan", "inf", "invalid"])
def test_rejects_bad_numeric_tags(metrics, tags, value) -> None:
    tags["classification_threshold"] = value
    assert not check(metrics, tags).eligible


def test_rejects_missing_metadata(metrics, tags) -> None:
    del tags["approval"]
    assert not check(metrics, tags).eligible


def test_decision_serializes(metrics, tags) -> None:
    result = check(metrics, tags).to_dict()
    assert result == {"eligible": True, "reasons": ()}
