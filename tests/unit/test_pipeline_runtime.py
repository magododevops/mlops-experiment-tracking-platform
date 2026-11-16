"""Unit tests for Docker training runtime policies."""

import pytest

from mlops_experiment_tracking.pipelines.runtime import (
    validate_runtime,
)


@pytest.fixture
def valid_runtime() -> dict:
    return {
        "uid": 1000,
        "username": "trainer",
        "git_executable": "/usr/bin/git",
        "revision": "a" * 40,
        "root_readonly": True,
        "data_readonly": True,
        "outputs_writable": True,
    }


def test_accepts_secure_runtime(valid_runtime: dict) -> None:
    validate_runtime(**valid_runtime)


@pytest.mark.parametrize("uid", [0, -1])
def test_rejects_root_or_invalid_uid(
    valid_runtime: dict,
    uid: int,
) -> None:
    valid_runtime["uid"] = uid

    with pytest.raises(ValueError, match="non-root"):
        validate_runtime(**valid_runtime)


def test_rejects_unknown_user(valid_runtime: dict) -> None:
    valid_runtime["username"] = "unknown"

    with pytest.raises(ValueError, match="user"):
        validate_runtime(**valid_runtime)


def test_rejects_missing_git(valid_runtime: dict) -> None:
    valid_runtime["git_executable"] = None

    with pytest.raises(ValueError, match="Git"):
        validate_runtime(**valid_runtime)


def test_rejects_invalid_git_revision(valid_runtime: dict) -> None:
    valid_runtime["revision"] = "not-a-commit"

    with pytest.raises(ValueError, match="Git commit"):
        validate_runtime(**valid_runtime)


def test_rejects_writable_root(valid_runtime: dict) -> None:
    valid_runtime["root_readonly"] = False

    with pytest.raises(ValueError, match="root filesystem"):
        validate_runtime(**valid_runtime)


def test_rejects_writable_dataset(valid_runtime: dict) -> None:
    valid_runtime["data_readonly"] = False

    with pytest.raises(ValueError, match="data mount"):
        validate_runtime(**valid_runtime)


def test_rejects_unwritable_outputs(valid_runtime: dict) -> None:
    valid_runtime["outputs_writable"] = False

    with pytest.raises(ValueError, match="outputs"):
        validate_runtime(**valid_runtime)
