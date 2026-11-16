"""Reproducible baseline experiment definitions."""

from dataclasses import dataclass
from typing import Literal

ModelName = Literal["dummy", "logistic_regression"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration of an individual MLflow run."""

    name: str
    model_name: ModelName
    threshold: float


def build_run_configs(
    selected_threshold: float,
) -> tuple[RunConfig, ...]:
    """Create baseline comparisons without duplicate configurations."""
    if not 0 < selected_threshold < 1:
        raise ValueError("Threshold must be between zero and one")

    configs = [
        RunConfig("dummy-baseline", "dummy", 0.50),
        RunConfig("logistic-default", "logistic_regression", 0.50),
    ]

    if selected_threshold != 0.50:
        configs.append(
            RunConfig(
                "logistic-selected",
                "logistic_regression",
                selected_threshold,
            )
        )

    return tuple(configs)
