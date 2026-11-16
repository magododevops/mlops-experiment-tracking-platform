"""Binary classification evaluation for predictive maintenance."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Model quality metrics for the positive failure class."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible metrics."""
        return asdict(self)


def evaluate_binary_classifier(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike,
) -> EvaluationResult:
    """Evaluate binary predictions and positive-class probabilities."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    scores = np.asarray(y_score)

    if actual.ndim != 1 or predicted.ndim != 1 or scores.ndim != 1:
        raise ValueError("Expected one-dimensional inputs")

    if not (len(actual) == len(predicted) == len(scores)):
        raise ValueError("Input lengths must match")

    if len(actual) == 0:
        raise ValueError("Evaluation inputs must not be empty")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Ground truth must be binary")

    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predictions must be binary")

    if not np.isfinite(scores).all():
        raise ValueError("Probabilities must be finite")

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Probabilities must be between zero and one")

    if len(np.unique(actual)) != 2:
        raise ValueError("Both classes must be present in evaluation data")

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

    return EvaluationResult(
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1=float(f1_score(actual, predicted, zero_division=0)),
        average_precision=float(average_precision_score(actual, scores)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
