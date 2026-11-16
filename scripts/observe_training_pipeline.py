"""Audit an existing training execution without changing MLflow."""

import argparse
import json
import os
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.observability.execution import (
    structured_event,
    validate_manifest,
    validate_observed_run,
)


def observe(manifest_path: Path, tracking_uri: str) -> None:
    """Read a manifest and verify its persisted MLflow runs."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)

    execution_id = manifest["pipeline_execution_id"]
    git_commit = manifest["git_commit"]
    dataset_hash = manifest["dataset"]["source_sha256"]
    mlflow_data = manifest["mlflow"]
    experiment_id = str(mlflow_data["experiment_id"])

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment(experiment_id)
    if experiment.name != mlflow_data["experiment_name"]:
        raise ValueError("MLflow experiment name mismatch")

    print(
        structured_event(
            "PIPELINE_OBSERVATION_STARTED",
            execution_id=execution_id,
            experiment_id=experiment_id,
            expected_runs=len(mlflow_data["runs"]),
        ),
        flush=True,
    )

    for expected in mlflow_data["runs"]:
        run = client.get_run(expected["run_id"])

        observed = {
            "run_id": run.info.run_id,
            "name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "experiment_id": run.info.experiment_id,
            "tags": run.data.tags,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }

        validate_observed_run(
            expected,
            observed,
            execution_id=execution_id,
            git_commit=git_commit,
            dataset_hash=dataset_hash,
            experiment_id=experiment_id,
        )

        print(
            structured_event(
                "RUN_VERIFIED",
                execution_id=execution_id,
                run_id=expected["run_id"],
                model_run=expected["name"],
                f1=observed["metrics"]["f1"],
                recall=observed["metrics"]["recall"],
            ),
            flush=True,
        )

    print(
        structured_event(
            "PIPELINE_HEALTHY",
            execution_id=execution_id,
            verified_runs=len(mlflow_data["runs"]),
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only MLflow training execution observer.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Existing pipeline execution manifest",
    )
    args = parser.parse_args()

    execution_id = "unknown"

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        execution_id = str(manifest.get("pipeline_execution_id", "unknown"))

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

        observe(args.manifest, tracking_uri)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(
            structured_event(
                "PIPELINE_UNHEALTHY",
                execution_id=execution_id,
                error_type=type(error).__name__,
                error_message=str(error),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
