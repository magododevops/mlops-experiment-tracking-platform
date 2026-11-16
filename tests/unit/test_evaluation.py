"""Unit tests for classification metrics and validation contracts."""

import pytest

from mlops_experiment_tracking.training.evaluation import (
    evaluate_binary_classifier,
)


def test_perfect_predictions() -> None:
    result = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
    )

    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.average_precision == 1.0


def test_confusion_matrix_and_metrics() -> None:
    result = evaluate_binary_classifier(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0.1, 0.7, 0.3, 0.9],
    )

    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.true_positive == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_majority_baseline_has_zero_recall() -> None:
    result = evaluate_binary_classifier(
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0.0, 0.0, 0.0, 0.0],
    )

    assert result.accuracy == 0.75
    assert result.recall == 0.0
    assert result.precision == 0.0
    assert result.f1 == 0.0


@pytest.mark.parametrize(
    ("actual", "predicted", "scores"),
    [
        ([0, 1], [0], [0.1, 0.9]),
        ([0, 1], [0, 2], [0.1, 0.9]),
        ([0, 1], [0, 1], [0.1, 1.5]),
        ([0, 0], [0, 0], [0.1, 0.2]),
        ([], [], []),
    ],
)
def test_rejects_invalid_evaluation_inputs(
    actual: list[int],
    predicted: list[int],
    scores: list[float],
) -> None:
    with pytest.raises(ValueError):
        evaluate_binary_classifier(actual, predicted, scores)


def test_result_is_json_compatible() -> None:
    import json

    result = evaluate_binary_classifier([0, 1], [0, 1], [0.1, 0.9])

    assert json.loads(json.dumps(result.to_dict()))["recall"] == 1.0
