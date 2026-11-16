"""Project metadata and foundational configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    """Immutable project metadata."""

    name: str
    version: str


def get_project_info() -> ProjectInfo:
    """Return the current project metadata."""
    from mlops_experiment_tracking import __version__

    return ProjectInfo(
        name="mlops-experiment-tracking-platform",
        version=__version__,
    )
