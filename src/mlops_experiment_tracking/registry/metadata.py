"""Validated metadata for registered model candidates."""

import math
import re
from dataclasses import dataclass

MODEL_NAME = "predictive-maintenance-classifier"


@dataclass(frozen=True, slots=True)
class CandidateMetadata:
    """Metadata describing a model candidate."""

    threshold: float
    dataset_sha256: str
    validation_f1: float
    validation_recall: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not (0 < self.threshold < 1):
            raise ValueError("Invalid classification threshold")

        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_sha256):
            raise ValueError("Invalid dataset SHA-256")

        for metric in (self.validation_f1, self.validation_recall):
            if not math.isfinite(metric) or not 0 <= metric <= 1:
                raise ValueError("Invalid validation metric")

    def tags(self) -> dict[str, str]:
        """Build traceable model version tags."""
        return {
            "lifecycle": "candidate",
            "model_algorithm": "logistic_regression",
            "classification_threshold": str(self.threshold),
            "dataset_sha256": self.dataset_sha256,
            "validation_f1": str(self.validation_f1),
            "validation_recall": str(self.validation_recall),
            "approval": "not_approved",
        }
