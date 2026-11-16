"""Tests for dataset contracts and leakage prevention."""

import pandas as pd
import pytest

from mlops_experiment_tracking.data.schema import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    DataValidationError,
    select_features,
    validate_dataset,
)


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                1,
                "L1",
                "L",
                300.0,
                310.0,
                1500,
                40.0,
                100,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                2,
                "M2",
                "M",
                301.0,
                311.0,
                1300,
                60.0,
                200,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_valid_dataset(valid_frame: pd.DataFrame) -> None:
    validate_dataset(valid_frame)


def test_rejects_missing_values(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Torque [Nm]"] = float("nan")

    with pytest.raises(DataValidationError, match="missing values"):
        validate_dataset(valid_frame)


def test_rejects_duplicate_ids(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[1, "UDI"] = 1

    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_dataset(valid_frame)


def test_rejects_invalid_target(valid_frame: pd.DataFrame) -> None:
    valid_frame.loc[0, "Machine failure"] = 3

    with pytest.raises(DataValidationError, match="binary"):
        validate_dataset(valid_frame)


def test_rejects_missing_feature(valid_frame: pd.DataFrame) -> None:
    valid_frame = valid_frame.drop(columns="Torque [Nm]")

    with pytest.raises(DataValidationError, match="Schema mismatch"):
        validate_dataset(valid_frame)


def test_excludes_leakage_columns(valid_frame: pd.DataFrame) -> None:
    features = select_features(valid_frame)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert "Machine failure" not in features.columns
    assert "TWF" not in features.columns
    assert "Product ID" not in features.columns
