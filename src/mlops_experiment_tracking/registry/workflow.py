"""Pure rules for a local demonstration model lifecycle."""

from dataclasses import dataclass

DEMO_ALIAS = "staging-demo"
DEMO_APPROVAL_TAG = "demo_approval"
EXPECTED_VERSION = "1"


@dataclass(frozen=True, slots=True)
class LifecycleState:
    """Current demonstration state of a registered model."""

    approved: bool
    active: bool


def validate_transition(
    state: LifecycleState,
    action: str,
) -> None:
    """Reject invalid lifecycle transitions before mutations."""
    if action == "approve":
        if state.approved:
            raise ValueError("Candidate is already approved")
        if state.active:
            raise ValueError("Active candidate cannot be reapproved")

    elif action == "activate":
        if not state.approved:
            raise ValueError("Approval required before activation")
        if state.active:
            raise ValueError("Candidate is already active")

    elif action == "rollback":
        if not state.active:
            raise ValueError("No active demo alias to roll back")

    elif action == "revoke":
        if state.active:
            raise ValueError("Rollback required before revocation")
        if not state.approved:
            raise ValueError("Candidate is not approved")

    else:
        raise ValueError(f"Unsupported lifecycle action: {action}")


def next_state(
    state: LifecycleState,
    action: str,
) -> LifecycleState:
    """Compute the next state without modifying the Registry."""
    validate_transition(state, action)

    if action == "approve":
        return LifecycleState(approved=True, active=False)

    if action == "activate":
        return LifecycleState(approved=True, active=True)

    if action == "rollback":
        return LifecycleState(approved=True, active=False)

    return LifecycleState(approved=False, active=False)
