"""Pure validation rules for the training container runtime."""

from mlops_experiment_tracking.pipelines.execution import (
    validate_git_commit,
)


def validate_runtime(
    *,
    uid: int,
    username: str,
    git_executable: str | None,
    revision: str,
    root_readonly: bool,
    data_readonly: bool,
    outputs_writable: bool,
) -> None:
    """Reject an unsafe or incorrectly configured training environment."""
    if uid <= 0:
        raise ValueError("Training must run as a non-root user")

    if username != "trainer":
        raise ValueError("Unexpected training container user")

    if not git_executable:
        raise ValueError("Git executable is missing")

    validate_git_commit(revision)

    if not root_readonly:
        raise ValueError("Container root filesystem must be read-only")

    if not data_readonly:
        raise ValueError("Training data mount must be read-only")

    if not outputs_writable:
        raise ValueError("Pipeline outputs must be writable")
