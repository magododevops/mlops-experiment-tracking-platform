"""Read-only promotion eligibility policy for registered models."""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from mlops_experiment_tracking.training.evaluation import EvaluationResult
from mlops_experiment_tracking.training.selection import SelectionPolicy


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """An eligibility decision, not a deployment approval."""

    eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return JSON-compatible decision evidence."""
        return asdict(self)


def evaluate_promotion(
    metrics: EvaluationResult,
    tags: Mapping[str, str],
    *,
    expected_sha256: str,
    expected_threshold: float,
) -> PromotionDecision:
    """Check a candidate using validation-only metrics and provenance."""
    policy = SelectionPolicy()
    reasons: list[str] = []

    expected_tags = {
        "lifecycle": "candidate",
        "approval": "not_approved",
        "model_algorithm": "logistic_regression",
        "dataset_sha256": expected_sha256,
    }

    for key, expected in expected_tags.items():
        if tags.get(key) != expected:
            reasons.append(f"Invalid registry tag: {key}")

    try:
        threshold = float(tags["classification_threshold"])
        tagged_recall = float(tags["validation_recall"])
        tagged_f1 = float(tags["validation_f1"])
    except (KeyError, TypeError, ValueError):
        reasons.append("Invalid numeric registry metadata")
    else:
        numeric_values = (threshold, tagged_recall, tagged_f1)
        if not all(math.isfinite(v) for v in numeric_values):
            reasons.append("Non-finite registry metadata")
        else:
            if not math.isclose(threshold, expected_threshold, rel_tol=0, abs_tol=1e-12):
                reasons.append("Threshold differs from selection")

            if not math.isclose(tagged_recall, metrics.recall, rel_tol=0, abs_tol=1e-12):
                reasons.append("Recall differs from registry metadata")

            if not math.isclose(tagged_f1, metrics.f1, rel_tol=0, abs_tol=1e-12):
                reasons.append("F1 differs from registry metadata")

    if metrics.recall < policy.min_recall:
        reasons.append("Recall below required minimum")

    if metrics.false_positive > policy.max_false_positives:
        reasons.append("Too many false positives")

    return PromotionDecision(
        eligible=not reasons,
        reasons=tuple(reasons),
    )
