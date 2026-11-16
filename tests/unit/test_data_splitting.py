"""Dataset split reproducibility and leakage tests."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlops_experiment_tracking.data.schema import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.data.splitting import split_dataset


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    rows = []
    for index in range(100):
        failure = int(index % 10 == 0)
        rows.append(
            (
                index + 1,
                f"L{index + 1}",
                "L",
                300.0,
                310.0,
                1500,
                40.0,
                100,
                failure,
                failure,
                0,
                0,
                0,
                0,
            )
        )

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_split_sizes(sample_dataset: pd.DataFrame) -> None:
    splits = split_dataset(sample_dataset)

    assert len(splits.train) == 60
    assert len(splits.validation) == 20
    assert len(splits.test) == 20


def test_splits_do_not_overlap(sample_dataset: pd.DataFrame) -> None:
    splits = split_dataset(sample_dataset)

    train = set(splits.train.index)
    validation = set(splits.validation.index)
    test = set(splits.test.index)

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert len(train | validation | test) == len(sample_dataset)


def test_split_is_deterministic(sample_dataset: pd.DataFrame) -> None:
    first = split_dataset(sample_dataset)
    second = split_dataset(sample_dataset)

    assert_frame_equal(first.train, second.train)
    assert_frame_equal(first.validation, second.validation)
    assert_frame_equal(first.test, second.test)


def test_split_preserves_class_balance(sample_dataset: pd.DataFrame) -> None:
    splits = split_dataset(sample_dataset)

    for partition in (splits.train, splits.validation, splits.test):
        assert partition[TARGET_COLUMN].mean() == pytest.approx(0.1)


def test_excludes_identifiers_and_failure_modes(
    sample_dataset: pd.DataFrame,
) -> None:
    splits = split_dataset(sample_dataset)

    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]

    for partition in (splits.train, splits.validation, splits.test):
        assert list(partition.columns) == expected
