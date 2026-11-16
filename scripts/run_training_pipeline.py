"""Run reproducible baseline experiments and group their MLflow runs.

This command creates new experiment runs but does not register,
approve, or deploy any model.
"""

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.pipelines.execution import (
    EXPECTED_RUN_NAMES,
    extract_run_ids,
    validate_git_commit,
    validate_run_records,
)
from mlops_experiment_tracking.pipelines.failures import (
    PipelineStageError,
    build_failure_manifest,
    save_failure_manifest,
)

EXPERIMENT_NAME = "predictive-maintenance-baselines"


def utc_now() -> str:
    """Return a timezone-aware timestamp."""
    return datetime.now(UTC).isoformat()


def run_command(command: list[str]) -> str:
    """Execute an existing pipeline step and propagate failures."""
    print(f"\nRunning: {' '.join(command)}", flush=True)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout, end="", flush=True)

    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline step failed with exit code {result.returncode}: {command}")

    return result.stdout


def require_clean_repository() -> str:
    """Capture local Git revision or validate a supplied image revision."""
    supplied_commit = os.environ.get("PIPELINE_GIT_COMMIT")

    if supplied_commit is not None:
        return validate_git_commit(supplied_commit)

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    )

    if status.strip():
        raise RuntimeError("Commit or discard source changes before training")

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()

    if branch != "main":
        raise RuntimeError("Expected main branch")

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_mlflow(tracking_uri: str) -> None:
    """Fail before training when tracking is unavailable."""
    health_url = f"{tracking_uri.rstrip('/')}/health"

    with urlopen(health_url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("MLflow health check failed")


def get_run_records(
    client: MlflowClient,
    run_ids: tuple[str, ...],
) -> list[dict]:
    """Retrieve the actual persisted metadata from MLflow."""
    records = []

    for expected_name, run_id in zip(
        EXPECTED_RUN_NAMES,
        run_ids,
        strict=True,
    ):
        run = client.get_run(run_id)
        name = run.data.tags.get("mlflow.runName")

        if name != expected_name:
            raise RuntimeError(f"Run name mismatch: expected {expected_name}, got {name}")

        records.append(
            {
                "name": name,
                "run_id": run_id,
                "status": run.info.status,
                "dataset_sha256": run.data.params["source_sha256"],
                "validation_records": int(run.data.params["validation_records"]),
                "metrics": {
                    "precision": run.data.metrics["precision"],
                    "recall": run.data.metrics["recall"],
                    "f1": run.data.metrics["f1"],
                    "average_precision": run.data.metrics["average_precision"],
                },
            }
        )

    return records


def execute_pipeline(execution_id: str, started_at: str) -> None:
    git_commit = require_clean_repository()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    require_mlflow(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Required experiment not found: {EXPERIMENT_NAME}")

    source = Path("data/raw/ai4i2020.csv")
    if not source.is_file():
        raise FileNotFoundError(source)

    dataset_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    print(f"Pipeline execution ID: {execution_id}")
    print(f"Git commit: {git_commit}")
    print(f"Tracking URI: {tracking_uri}")

    run_command(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/verify_dataset.py",
        ]
    )

    if os.environ.get("PIPELINE_TEST_FAIL_AT") == "before_training":
        raise PipelineStageError("training_preflight", "Controlled failure injection")

    training_output = run_command(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/log_baseline_experiments.py",
        ]
    )

    run_ids = extract_run_ids(training_output)
    records = get_run_records(client, run_ids)

    validate_run_records(records, dataset_hash)

    # Group these runs only after verifying persisted MLflow data.
    for record in records:
        client.set_tag(
            record["run_id"],
            "pipeline_execution_id",
            execution_id,
        )
        client.set_tag(
            record["run_id"],
            "source_git_commit",
            git_commit,
        )
        client.set_tag(
            record["run_id"],
            "pipeline_type",
            "baseline-training",
        )
        client.set_tag(
            record["run_id"],
            "mlflow.source.name",
            "scripts/log_baseline_experiments.py",
        )
        client.set_tag(
            record["run_id"],
            "mlflow.source.type",
            "LOCAL",
        )
        client.set_tag(
            record["run_id"],
            "mlflow.source.git.commit",
            git_commit,
        )

    for record in records:
        stored = client.get_run(record["run_id"])
        if stored.data.tags.get("pipeline_execution_id") != execution_id:
            raise RuntimeError("Failed to verify pipeline tag")

    manifest = {
        "schema_version": 1,
        "pipeline": "baseline-training",
        "pipeline_execution_id": execution_id,
        "status": "SUCCEEDED",
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "git_commit": git_commit,
        "dataset": {
            "name": "ai4i2020",
            "source_sha256": dataset_hash,
            "validation_partition": "validation",
        },
        "mlflow": {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_id": experiment.experiment_id,
            "runs": records,
        },
        "model_registry_modified": False,
        "held_out_test_used": False,
    }

    destination = Path("outputs/pipelines") / execution_id / "manifest.json"

    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("AUTOMATED TRAINING PIPELINE")
    print("---------------------------")
    print(f"Execution ID: {execution_id}")
    print(f"Git commit: {git_commit}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"New MLflow runs: {len(records)}")
    print(f"Manifest: {destination}")

    for record in records:
        print(
            f"- {record['name']}: "
            f"F1={record['metrics']['f1']:.4f}, "
            f"Recall={record['metrics']['recall']:.4f}"
        )

    print("Model Registry: UNCHANGED")
    print("Held-out test dataset: NOT USED")
    print("TRAINING PIPELINE: PASSED")


def main() -> None:
    """Execute training and persist failure evidence when it aborts."""
    execution_id = uuid.uuid4().hex
    started_at = utc_now()

    try:
        execute_pipeline(execution_id, started_at)
    except Exception as error:
        stage = error.stage if isinstance(error, PipelineStageError) else "orchestration"

        # The revision may be available even when preflight fails.
        revision = os.environ.get("PIPELINE_GIT_COMMIT")
        if revision is None:
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = None

        evidence = build_failure_manifest(
            execution_id=execution_id,
            started_at=started_at,
            stage=stage,
            error_type=type(error).__name__,
            git_commit=revision,
        )

        destination = save_failure_manifest(
            evidence,
            output_root=Path("outputs/pipelines"),
        )

        print(
            json.dumps(
                {
                    "event": "PIPELINE_FAILED",
                    "pipeline_execution_id": execution_id,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "failure_manifest": str(destination),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
