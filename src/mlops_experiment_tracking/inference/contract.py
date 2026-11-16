"""Input validation and decision logic for predictive maintenance."""

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlops_experiment_tracking.data.schema import FEATURE_COLUMNS

CATEGORICAL_VALUES = frozenset({"L", "M", "H"})

SAMPLE_EQUIPMENT: dict[str, str | float | int] = {
    "Type": "M",
    "Air temperature [K]": 298.0,
    "Process temperature [K]": 308.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 45.0,
    "Tool wear [min]": 120,
}


def validate_equipment(data: Mapping[str, Any]) -> pd.DataFrame:
    """Return a single-row feature frame with a strict input contract."""
    expected = set(FEATURE_COLUMNS)
    received = set(data)

    missing = expected - received
    unexpected = received - expected

    if missing or unexpected:
        raise ValueError(
            f"Invalid feature columns: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    if data["Type"] not in CATEGORICAL_VALUES:
        raise ValueError("Type must be L, M, or H")

    numeric_columns = [column for column in FEATURE_COLUMNS if column != "Type"]

    row: dict[str, str | float] = {"Type": data["Type"]}

    for column in numeric_columns:
        value = data[column]

        if type(value) not in (int, float):
            raise ValueError(f"{column} must be numeric")

        number = float(value)

        if not math.isfinite(number):
            raise ValueError(f"{column} must be finite")

        if (
            column
            in (
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
            )
            and number <= 0
        ):
            raise ValueError(f"{column} must be positive")

        if column in ("Torque [Nm]", "Tool wear [min]") and number < 0:
            raise ValueError(f"{column} must not be negative")

        row[column] = number

    return pd.DataFrame(
        [{column: row[column] for column in FEATURE_COLUMNS}],
        columns=list(FEATURE_COLUMNS),
    )


def classify_failure(probability: float, threshold: float) -> str:
    """Apply the approved candidate's explicit decision threshold."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid predicted probability")

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid decision threshold")

    return "FAILURE" if probability >= threshold else "NORMAL"
