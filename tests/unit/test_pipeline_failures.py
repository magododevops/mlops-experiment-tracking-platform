"""Unit tests for training failure evidence."""

import json

import pytest

from mlops_experiment_tracking.pipelines.failures import (
    PipelineStageError,
    build_failure_manifest,
    save_failure_manifest,
)

EXECUTION = "a" * 32
STARTED = "2026-11-16T10:00:00+00:00"


@pytest.fixture
def failure() -> dict:
    return build_failure_manifest(
        execution_id=EXECUTION,
        started_at=STARTED,
        stage="training_preflight",
        error_type="PipelineStageError",
        git_commit="b" * 40,
    )


def test_records_failed_execution(failure: dict) -> None:
    assert failure["status"] == "FAILED"
    assert failure["failed_stage"] == "training_preflight"
    assert failure["automatic_retry"] is False


def test_preserves_execution_id(failure: dict) -> None:
    assert failure["pipeline_execution_id"] == EXECUTION


def test_preserves_git_revision(failure: dict) -> None:
    assert failure["git_commit"] == "b" * 40


def test_does_not_store_exception_message(failure: dict) -> None:
    assert "error_message" not in failure


def test_rejects_invalid_execution_id() -> None:
    with pytest.raises(ValueError, match="execution ID"):
        build_failure_manifest(
            execution_id="invalid",
            started_at=STARTED,
            stage="training",
            error_type="RuntimeError",
            git_commit=None,
        )


def test_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError, match="failure stage"):
        build_failure_manifest(
            execution_id=EXECUTION,
            started_at=STARTED,
            stage="../secrets",
            error_type="RuntimeError",
            git_commit=None,
        )


def test_rejects_invalid_error_type() -> None:
    with pytest.raises(ValueError, match="error type"):
        build_failure_manifest(
            execution_id=EXECUTION,
            started_at=STARTED,
            stage="training",
            error_type="Invalid Error",
            git_commit=None,
        )


def test_saves_failure_evidence(tmp_path, failure: dict) -> None:
    saved = save_failure_manifest(
        failure,
        output_root=tmp_path,
    )
    assert json.loads(saved.read_text()) == failure


def test_refuses_to_overwrite_failure(tmp_path, failure: dict) -> None:
    save_failure_manifest(failure, output_root=tmp_path)

    with pytest.raises(FileExistsError):
        save_failure_manifest(failure, output_root=tmp_path)


def test_stage_error_preserves_stage() -> None:
    error = PipelineStageError(
        "training_preflight",
        "Injected failure",
    )
    assert error.stage == "training_preflight"
