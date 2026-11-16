"""Tests for model registry candidate metadata."""

import pytest

from mlops_experiment_tracking.registry.metadata import (
    MODEL_NAME,
    CandidateMetadata,
)

HASH = "a" * 64


def test_builds_candidate_tags() -> None:
    metadata = CandidateMetadata(0.2, HASH, 0.42, 0.44)

    tags = metadata.tags()

    assert tags["lifecycle"] == "candidate"
    assert tags["approval"] == "not_approved"
    assert tags["classification_threshold"] == "0.2"
    assert tags["dataset_sha256"] == HASH


def test_registry_name_is_stable() -> None:
    assert MODEL_NAME == "predictive-maintenance-classifier"


@pytest.mark.parametrize(
    "threshold",
    [0.0, 1.0, -0.1, float("nan"), float("inf")],
)
def test_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        CandidateMetadata(threshold, HASH, 0.42, 0.44)


def test_rejects_invalid_dataset_hash() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        CandidateMetadata(0.2, "invalid", 0.42, 0.44)


@pytest.mark.parametrize(
    "metric",
    [-0.1, 1.1, float("nan")],
)
def test_rejects_invalid_metrics(metric: float) -> None:
    with pytest.raises(ValueError):
        CandidateMetadata(0.2, HASH, metric, 0.44)
