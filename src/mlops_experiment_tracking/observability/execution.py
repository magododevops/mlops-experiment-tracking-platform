"""Pure validation of training pipeline execution evidence."""

import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from mlops_experiment_tracking.pipelines.execution import (
    EXPECTED_RUN_NAMES,
    validate_git_commit,
)

METRIC_NAMES = (
    "precision",
    "recall",
    "f1",
    "average_precision",
)


def structured_event(
    event: str,
    *,
    execution_id: str,
    **fields: Any,
) -> str:
    """Create a compact, machine-readable operational event."""
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "ERROR" if event == "PIPELINE_UNHEALTHY" else "INFO",
        "event": event,
        "pipeline_execution_id": execution_id,
        **fields,
    }
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate immutable local evidence before querying MLflow."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unexpected manifest schema version")

    if manifest.get("pipeline") != "baseline-training":
        raise ValueError("Unexpected pipeline type")

    if manifest.get("status") != "SUCCEEDED":
        raise ValueError("Pipeline manifest is not successful")

    execution_id = manifest.get("pipeline_execution_id", "")
    if not isinstance(execution_id, str) or not re.fullmatch(r"[0-9a-f]{32}", execution_id):
        raise ValueError("Invalid pipeline execution ID")

    validate_git_commit(manifest.get("git_commit", ""))

    if manifest.get("held_out_test_used") is not False:
        raise ValueError("Held-out test usage is not permitted")

    if manifest.get("model_registry_modified") is not False:
        raise ValueError("Unexpected registry mutation")

    dataset = manifest.get("dataset", {})
    source_hash = dataset.get("source_sha256", "")

    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("Invalid dataset provenance")

    if dataset.get("validation_partition") != "validation":
        raise ValueError("Unexpected evaluation partition")

    mlflow = manifest.get("mlflow", {})
    if mlflow.get("experiment_name") != ("predictive-maintenance-baselines"):
        raise ValueError("Unexpected MLflow experiment")

    runs = mlflow.get("runs", [])
    if len(runs) != len(EXPECTED_RUN_NAMES):
        raise ValueError("Expected exactly three runs")

    if tuple(run["name"] for run in runs) != EXPECTED_RUN_NAMES:
        raise ValueError("Unexpected run names or order")

    ids = [run["run_id"] for run in runs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate run IDs")

    if any(
        not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id) for run_id in ids
    ):
        raise ValueError("Invalid MLflow run ID")


def validate_observed_run(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    execution_id: str,
    git_commit: str,
    dataset_hash: str,
    experiment_id: str,
) -> None:
    """Compare a manifest record with persisted tracking information."""
    for field in ("run_id", "name", "status"):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"Run {field} mismatch")

    if observed.get("status") != "FINISHED":
        raise ValueError("Run is not finished")

    if observed.get("experiment_id") != experiment_id:
        raise ValueError("Run belongs to another experiment")

    tags = observed.get("tags", {})

    expected_tags = {
        "pipeline_execution_id": execution_id,
        "source_git_commit": git_commit,
        "pipeline_type": "baseline-training",
    }

    for key, value in expected_tags.items():
        if tags.get(key) != value:
            raise ValueError(f"Run tag mismatch: {key}")

    params = observed.get("params", {})
    if params.get("source_sha256") != dataset_hash:
        raise ValueError("Run dataset hash mismatch")

    if int(params.get("validation_records", -1)) != 2000:
        raise ValueError("Unexpected validation record count")

    for metric in METRIC_NAMES:
        expected_value = float(expected["metrics"][metric])
        observed_value = float(observed["metrics"][metric])

        if not all(math.isfinite(value) for value in (expected_value, observed_value)):
            raise ValueError(f"Non-finite metric: {metric}")

        if not math.isclose(
            expected_value,
            observed_value,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Run metric mismatch: {metric}")
