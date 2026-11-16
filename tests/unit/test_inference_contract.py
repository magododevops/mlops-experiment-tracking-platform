"""Unit tests for the model inference contract."""

from copy import deepcopy

import pytest

from mlops_experiment_tracking.data.schema import FEATURE_COLUMNS
from mlops_experiment_tracking.inference.contract import (
    SAMPLE_EQUIPMENT,
    classify_failure,
    validate_equipment,
)


def test_valid_equipment() -> None:
    frame = validate_equipment(SAMPLE_EQUIPMENT)

    assert frame.shape == (1, 6)
    assert list(frame.columns) == list(FEATURE_COLUMNS)


def test_rejects_missing_feature() -> None:
    data = deepcopy(SAMPLE_EQUIPMENT)
    del data["Torque [Nm]"]

    with pytest.raises(ValueError, match="missing"):
        validate_equipment(data)


def test_rejects_extra_feature() -> None:
    data = deepcopy(SAMPLE_EQUIPMENT)
    data["Machine failure"] = 1

    with pytest.raises(ValueError, match="unexpected"):
        validate_equipment(data)


def test_rejects_unknown_machine_type() -> None:
    data = deepcopy(SAMPLE_EQUIPMENT)
    data["Type"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Type"):
        validate_equipment(data)


@pytest.mark.parametrize("value", ["1500", True, float("nan")])
def test_rejects_invalid_numeric_value(value: object) -> None:
    data = deepcopy(SAMPLE_EQUIPMENT)
    data["Rotational speed [rpm]"] = value

    with pytest.raises(ValueError):
        validate_equipment(data)


def test_rejects_negative_tool_wear() -> None:
    data = deepcopy(SAMPLE_EQUIPMENT)
    data["Tool wear [min]"] = -1

    with pytest.raises(ValueError):
        validate_equipment(data)


def test_failure_at_threshold() -> None:
    assert classify_failure(0.20, 0.20) == "FAILURE"


def test_normal_below_threshold() -> None:
    assert classify_failure(0.19, 0.20) == "NORMAL"


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        classify_failure(probability, 0.20)


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        classify_failure(0.40, 1.5)
