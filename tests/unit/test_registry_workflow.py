"""Tests for explicit local model lifecycle transitions."""

import pytest

from mlops_experiment_tracking.registry.workflow import (
    LifecycleState,
    next_state,
)


def test_approval_does_not_activate() -> None:
    state = next_state(
        LifecycleState(False, False),
        "approve",
    )
    assert state == LifecycleState(True, False)


def test_activation_requires_approval() -> None:
    with pytest.raises(ValueError, match="Approval"):
        next_state(LifecycleState(False, False), "activate")


def test_activation_selects_staging_version() -> None:
    state = next_state(
        LifecycleState(True, False),
        "activate",
    )
    assert state == LifecycleState(True, True)


def test_rollback_retains_approval() -> None:
    state = next_state(
        LifecycleState(True, True),
        "rollback",
    )
    assert state == LifecycleState(True, False)


def test_revoke_requires_rollback() -> None:
    with pytest.raises(ValueError, match="Rollback"):
        next_state(LifecycleState(True, True), "revoke")


def test_revoke_returns_to_candidate() -> None:
    state = next_state(
        LifecycleState(True, False),
        "revoke",
    )
    assert state == LifecycleState(False, False)


def test_duplicate_approval_is_rejected() -> None:
    with pytest.raises(ValueError, match="already approved"):
        next_state(LifecycleState(True, False), "approve")


def test_rollback_requires_active_alias() -> None:
    with pytest.raises(ValueError, match="No active"):
        next_state(LifecycleState(True, False), "rollback")


def test_revoke_requires_approval() -> None:
    with pytest.raises(ValueError, match="not approved"):
        next_state(LifecycleState(False, False), "revoke")


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        next_state(LifecycleState(False, False), "deploy-production")
