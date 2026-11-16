"""Baseline classifiers and their training pipelines."""

from typing import Literal

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.features.preprocessing import (
    build_preprocessor,
)

ModelName = Literal["dummy", "logistic_regression"]


def build_model(name: ModelName) -> Pipeline:
    """Create a fresh, unfitted classifier pipeline."""
    if name == "dummy":
        estimator = DummyClassifier(strategy="most_frequent")
    elif name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def train_model(name: ModelName, train: pd.DataFrame) -> Pipeline:
    """Fit a classifier using training data only."""
    missing = set((*FEATURE_COLUMNS, TARGET_COLUMN)) - set(train.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    if set(train.columns) != set((*FEATURE_COLUMNS, TARGET_COLUMN)):
        raise ValueError("Unexpected training columns")

    if train.empty:
        raise ValueError("Training dataset must not be empty")

    if set(train[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training target must contain both binary classes")

    model = build_model(name)

    features = train.loc[:, FEATURE_COLUMNS]
    target = train[TARGET_COLUMN]

    model.fit(features, target)
    return model
