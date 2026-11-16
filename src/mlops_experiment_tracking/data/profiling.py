"""Exploratory statistics calculated exclusively from training data."""

import pandas as pd

from mlops_experiment_tracking.data.schema import (
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def build_profile(train: pd.DataFrame) -> dict:
    """Build JSON-compatible descriptive statistics."""
    counts = train[TARGET_COLUMN].value_counts()

    numeric = train.loc[:, NUMERIC_COLUMNS].agg(["min", "max", "mean", "std"])

    return {
        "scope": "train_only",
        "rows": len(train),
        "class_distribution": {
            str(label): int(count) for label, count in counts.sort_index().items()
        },
        "failure_rate": round(float(train[TARGET_COLUMN].mean()), 6),
        "equipment_types": {
            str(label): int(count)
            for label, count in train["Type"].value_counts().sort_index().items()
        },
        "numeric_features": {
            column: {
                statistic: round(float(value), 4) for statistic, value in numeric[column].items()
            }
            for column in NUMERIC_COLUMNS
        },
    }
