"""Manage a local demonstration lifecycle in MLflow Model Registry.

This script does not deploy or approve models for production.
"""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.registry.metadata import MODEL_NAME
from mlops_experiment_tracking.registry.workflow import (
    DEMO_ALIAS,
    DEMO_APPROVAL_TAG,
    EXPECTED_VERSION,
    LifecycleState,
    next_state,
)


def get_demo_alias_version(client: MlflowClient) -> str | None:
    """Resolve the staging alias, returning None only when absent."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, DEMO_ALIAS)
    except MlflowException as error:
        message = str(error)
        missing_alias = f"Registered model alias {DEMO_ALIAS} not found"

        if (
            error.error_code
            in (
                "RESOURCE_DOES_NOT_EXIST",
                "INVALID_PARAMETER_VALUE",
            )
            and missing_alias in message
        ):
            return None

        raise

    return str(version.version)


def read_state(client: MlflowClient) -> LifecycleState:
    """Read and validate the known candidate's demo state."""
    candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")

    if str(candidate.version) != EXPECTED_VERSION:
        raise RuntimeError("Unexpected candidate alias version")

    version = client.get_model_version(MODEL_NAME, EXPECTED_VERSION)
    tags = version.tags

    if tags.get("lifecycle") != "candidate":
        raise RuntimeError("Original lifecycle tag was modified")

    if tags.get("approval") != "not_approved":
        raise RuntimeError("Production approval state was modified")

    raw_approval = tags.get(DEMO_APPROVAL_TAG, "not_approved")

    if raw_approval not in ("not_approved", "approved"):
        raise RuntimeError("Unknown demonstration approval state")

    active_version = get_demo_alias_version(client)

    if active_version not in (None, EXPECTED_VERSION):
        raise RuntimeError("Demo alias points to another version; refusing changes")

    state = LifecycleState(
        approved=raw_approval == "approved",
        active=active_version == EXPECTED_VERSION,
    )

    if state.active and not state.approved:
        raise RuntimeError("Active alias has no demonstration approval")

    return state


def verify_quality_gate() -> None:
    """Run the existing read-only gate before positive transitions."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_promotion.py"],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError("Model promotion quality gate failed")


def timestamp() -> str:
    """Return an auditable UTC timestamp."""
    return datetime.now(UTC).isoformat()


def print_status(state: LifecycleState) -> None:
    """Display current local demonstration state."""
    print()
    print("MODEL DEMO LIFECYCLE")
    print("--------------------")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {EXPECTED_VERSION}")
    print("Candidate alias: candidate")
    print(f"Demo approval: {'APPROVED' if state.approved else 'NOT APPROVED'}")
    print(f"Staging alias: {'ACTIVE' if state.active else 'INACTIVE'}")
    print("Production approval: NOT GRANTED")
    print("Production deployment: NOT PERFORMED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local demonstration model lifecycle.")
    parser.add_argument(
        "action",
        choices=("status", "approve", "activate", "rollback", "revoke"),
    )
    args = parser.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    state = read_state(client)

    if args.action == "status":
        print_status(state)
        return

    updated = next_state(state, args.action)

    if args.action in ("approve", "activate"):
        verify_quality_gate()

    if args.action == "approve":
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            DEMO_APPROVAL_TAG,
            "approved",
        )
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            "demo_approved_at",
            timestamp(),
        )

    elif args.action == "activate":
        client.set_registered_model_alias(
            MODEL_NAME,
            DEMO_ALIAS,
            EXPECTED_VERSION,
        )
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            "demo_activated_at",
            timestamp(),
        )

    elif args.action == "rollback":
        client.delete_registered_model_alias(
            MODEL_NAME,
            DEMO_ALIAS,
        )
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            "demo_rollback_at",
            timestamp(),
        )

    elif args.action == "revoke":
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            DEMO_APPROVAL_TAG,
            "not_approved",
        )
        client.set_model_version_tag(
            MODEL_NAME,
            EXPECTED_VERSION,
            "demo_revoked_at",
            timestamp(),
        )

    actual = read_state(client)

    if actual != updated:
        raise RuntimeError("Registry state differs from expected transition")

    print(f"Action completed: {args.action.upper()}")
    print_status(actual)


if __name__ == "__main__":
    main()
