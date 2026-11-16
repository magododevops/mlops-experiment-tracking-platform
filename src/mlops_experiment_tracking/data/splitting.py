"""Deterministic, stratified dataset splitting."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataset,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


@dataclass(frozen=True)
class DatasetSplits:
    """Non-overlapping train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    frame: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """Produce 60/20/20 stratified partitions."""
    validate_dataset(frame)

    if frame[TARGET_COLUMN].value_counts().min() < 3:
        raise ValueError("Each target class requires at least 3 samples")

    indices = frame.index.to_numpy()

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )

    train_val = frame.loc[train_val_idx]

    # 0.25 of the remaining 80% gives 20% overall.
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=random_state,
        stratify=train_val[TARGET_COLUMN],
    )

    columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    def select(index: object) -> pd.DataFrame:
        return frame.loc[index, columns].sort_index().copy()

    return DatasetSplits(
        train=select(train_idx),
        validation=select(validation_idx),
        test=select(test_idx),
    )
