"""Training-only data profiling tests."""

import pandas as pd

from mlops_experiment_tracking.data.profiling import build_profile
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def test_profile_contains_training_statistics() -> None:
    frame = pd.DataFrame(
        [
            ["L", 300.0, 310.0, 1500, 40.0, 100, 0],
            ["M", 301.0, 311.0, 1300, 60.0, 200, 1],
        ],
        columns=[*FEATURE_COLUMNS, TARGET_COLUMN],
    )

    profile = build_profile(frame)

    assert profile["scope"] == "train_only"
    assert profile["rows"] == 2
    assert profile["class_distribution"] == {"0": 1, "1": 1}
    assert profile["failure_rate"] == 0.5
    assert profile["numeric_features"]["Torque [Nm]"]["mean"] == 50.0
