"""Tests for reproducible MLflow experiment configurations."""

import pytest

from mlops_experiment_tracking.tracking.experiments import (
    build_run_configs,
)


def test_creates_three_baseline_comparisons() -> None:
    configs = build_run_configs(0.20)

    assert len(configs) == 3
    assert [config.name for config in configs] == [
        "dummy-baseline",
        "logistic-default",
        "logistic-selected",
    ]


def test_selected_run_uses_policy_threshold() -> None:
    configs = build_run_configs(0.20)

    assert configs[-1].model_name == "logistic_regression"
    assert configs[-1].threshold == 0.20


def test_avoids_duplicate_default_threshold() -> None:
    configs = build_run_configs(0.50)

    assert len(configs) == 2
    assert len({config.name for config in configs}) == 2


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0, 1.5])
def test_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        build_run_configs(threshold)


def test_run_plan_is_deterministic() -> None:
    assert build_run_configs(0.20) == build_run_configs(0.20)
