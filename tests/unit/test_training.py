"""Tests for baseline training and preprocessing."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError

from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.training.models import (
    build_model,
    train_model,
)


@pytest.fixture
def training_frame() -> pd.DataFrame:
    rows = []

    for index in range(100):
        failure = int(index % 10 == 0)
        rows.append(
            [
                ("L", "M", "H")[index % 3],
                295.0 + index % 8,
                305.0 + index % 8,
                1200 + index * 5,
                30.0 + index % 20,
                float(index),
                failure,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[*FEATURE_COLUMNS, TARGET_COLUMN],
    )


def test_model_is_unfitted_when_created(
    training_frame: pd.DataFrame,
) -> None:
    model = build_model("logistic_regression")

    with pytest.raises(NotFittedError):
        model.predict(training_frame.loc[:, FEATURE_COLUMNS])


@pytest.mark.parametrize("name", ["dummy", "logistic_regression"])
def test_model_generates_binary_predictions(
    training_frame: pd.DataFrame,
    name: str,
) -> None:
    model = train_model(name, training_frame)

    predictions = model.predict(training_frame.loc[:, FEATURE_COLUMNS])

    assert len(predictions) == len(training_frame)
    assert set(predictions).issubset({0, 1})


def test_logistic_regression_produces_probabilities(
    training_frame: pd.DataFrame,
) -> None:
    model = train_model("logistic_regression", training_frame)

    probabilities = model.predict_proba(training_frame.loc[:, FEATURE_COLUMNS])

    assert probabilities.shape == (100, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert (abs(probabilities.sum(axis=1) - 1.0) < 1e-10).all()


def test_training_is_reproducible(
    training_frame: pd.DataFrame,
) -> None:
    first = train_model("logistic_regression", training_frame)
    second = train_model("logistic_regression", training_frame)

    x = training_frame.loc[:, FEATURE_COLUMNS]

    assert (abs(first.predict_proba(x) - second.predict_proba(x)) < 1e-10).all()


def test_preprocessing_is_fitted_only_on_training_data(
    training_frame: pd.DataFrame,
) -> None:
    model = train_model("logistic_regression", training_frame)

    scaler = model.named_steps["preprocessing"].named_transformers_["numeric"]

    original_means = scaler.mean_.copy()

    validation = training_frame.loc[:, FEATURE_COLUMNS].copy()
    validation["Torque [Nm]"] = 9999.0

    validation_before = validation.copy(deep=True)
    model.predict(validation)

    assert (scaler.mean_ == original_means).all()
    assert_frame_equal(validation, validation_before)


def test_unknown_category_can_be_transformed(
    training_frame: pd.DataFrame,
) -> None:
    model = train_model("logistic_regression", training_frame)
    new_data = training_frame.loc[:0, FEATURE_COLUMNS].copy()
    new_data["Type"] = "UNKNOWN"

    predictions = model.predict(new_data)

    assert len(predictions) == 1


def test_rejects_target_leakage_column(
    training_frame: pd.DataFrame,
) -> None:
    training_frame["TWF"] = 0

    with pytest.raises(ValueError, match="Unexpected training columns"):
        train_model("dummy", training_frame)
