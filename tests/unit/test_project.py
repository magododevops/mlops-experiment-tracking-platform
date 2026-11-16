"""Unit tests for foundational project metadata."""

from mlops_experiment_tracking.project import ProjectInfo, get_project_info


def test_project_info() -> None:
    project = get_project_info()

    assert isinstance(project, ProjectInfo)
    assert project.name == "mlops-experiment-tracking-platform"
    assert project.version == "0.1.0"
