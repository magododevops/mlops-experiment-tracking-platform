"""Validation-based model candidate selection policies."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Explicit experimental eligibility criteria."""

    min_recall: float = 0.40
    max_false_positives: int = 50

    def __post_init__(self) -> None:
        if not 0 <= self.min_recall <= 1:
            raise ValueError("min_recall must be between 0 and 1")
        if self.max_false_positives < 0:
            raise ValueError("max_false_positives must be nonnegative")


class NoEligibleCandidateError(ValueError):
    """Raised when no candidate satisfies the policy."""


def select_candidate(
    candidates: list[dict[str, Any]],
    policy: SelectionPolicy,
) -> dict[str, Any]:
    """Select the best eligible validation candidate.

    Ranking:
      1. Highest F1
      2. Highest recall
      3. Lowest decision threshold
    """
    if not candidates:
        raise NoEligibleCandidateError("No candidates supplied")

    seen_thresholds: set[float] = set()
    eligible: list[dict[str, Any]] = []

    for candidate in candidates:
        threshold = float(candidate["threshold"])
        recall = float(candidate["recall"])
        f1 = float(candidate["f1"])
        false_positives = candidate["false_positive"]

        if not all(math.isfinite(x) for x in (threshold, recall, f1)):
            raise ValueError("Non-finite candidate metric")

        if not all(0 <= x <= 1 for x in (threshold, recall, f1)):
            raise ValueError("Invalid candidate metric range")

        if (
            isinstance(false_positives, bool)
            or not isinstance(false_positives, int)
            or false_positives < 0
        ):
            raise ValueError("Invalid false-positive count")

        if threshold in seen_thresholds:
            raise ValueError("Duplicate candidate threshold")

        seen_thresholds.add(threshold)

        if recall >= policy.min_recall and false_positives <= policy.max_false_positives:
            eligible.append(candidate)

    if not eligible:
        raise NoEligibleCandidateError("No candidate satisfies the model quality gates")

    return max(
        eligible,
        key=lambda item: (
            float(item["f1"]),
            float(item["recall"]),
            -float(item["threshold"]),
        ),
    )
