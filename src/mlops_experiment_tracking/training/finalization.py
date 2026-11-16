"""Verification of a selected model before final evaluation."""

import json
from pathlib import Path
from typing import Any

from mlops_experiment_tracking.training.selection import (
    SelectionPolicy,
    select_candidate,
)


class CandidateIntegrityError(ValueError):
    """Raised when a saved candidate differs from validation results."""


def load_selection(path: Path) -> dict[str, Any]:
    """Read a previously generated model selection report."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CandidateIntegrityError(f"Cannot read selection report: {path}") from error

    if not isinstance(report, dict):
        raise CandidateIntegrityError("Invalid selection report")

    return report


def verify_selection(
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_hash: str,
    validation_rows: int,
) -> dict[str, Any]:
    """Verify selection against independently recomputed validation metrics."""
    expected_policy = SelectionPolicy()

    expected_metadata = {
        "model": "logistic_regression",
        "partition": "validation",
        "status": "candidate_selected",
        "source_sha256": source_hash,
        "validation_rows": validation_rows,
        "policy": {
            "min_recall": expected_policy.min_recall,
            "max_false_positives": expected_policy.max_false_positives,
        },
        "selection_criteria": [
            "highest_f1",
            "highest_recall",
            "lowest_threshold",
        ],
        "candidate_count": len(candidates),
    }

    for key, value in expected_metadata.items():
        if report.get(key) != value:
            raise CandidateIntegrityError(f"Selection metadata mismatch: {key}")

    expected = select_candidate(candidates, expected_policy)

    if report.get("selected") != expected:
        raise CandidateIntegrityError(
            "Selected candidate differs from recomputed validation results"
        )

    return expected
