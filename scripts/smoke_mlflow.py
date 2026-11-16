"""End-to-end smoke test for the MLflow tracking infrastructure."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "infrastructure-smoke-tests"
RUN_NAME = "tracking-storage-connectivity"


def main() -> None:
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    parsed = urlparse(tracking_uri)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Tracking URI must use HTTP or HTTPS")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    expected_artifact = {
        "purpose": "validate-mlflow-postgresql-s3",
        "status": "passed",
        "sample_count": 10,
    }

    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    with mlflow.start_run(
        run_name=RUN_NAME,
        tags={
            "environment": "local",
            "component": "infrastructure",
            "test_type": "integration",
        },
    ) as run:
        run_id = run.info.run_id

        mlflow.log_param("storage_backend", "s3-compatible")
        mlflow.log_metric("connectivity_check", 1.0)
        mlflow.log_dict(
            expected_artifact,
            "smoke/verification.json",
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    stored_run = client.get_run(run_id)

    assert stored_run.info.status == "FINISHED"
    assert stored_run.data.params["storage_backend"] == "s3-compatible"
    assert stored_run.data.metrics["connectivity_check"] == 1.0

    with TemporaryDirectory() as directory:
        downloaded_path = client.download_artifacts(
            run_id,
            "smoke/verification.json",
            dst_path=directory,
        )

        downloaded = json.loads(Path(downloaded_path).read_text(encoding="utf-8"))

        if downloaded != expected_artifact:
            raise AssertionError("Downloaded artifact differs")

    print()
    print("MLFLOW INTEGRATION RESULT")
    print("-------------------------")
    print(f"Run ID: {run_id}")
    print("PostgreSQL metadata: VERIFIED")
    print("S3 artifact round trip: VERIFIED")
    print("Run status: FINISHED")
    print("Integration smoke test: PASSED")


if __name__ == "__main__":
    main()
