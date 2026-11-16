"""Read, validate, and compare existing MLflow validation runs."""

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.tracking.comparison import (
    EXPECTED_NAMES,
    METRICS,
    build_markdown,
    validate_comparison,
)

EXPERIMENT = "predictive-maintenance-baselines"
ARTIFACT_PATH = "evaluation/validation.json"


def get_record(
    client: MlflowClient,
    experiment_id: str,
    name: str,
    source_hash: str,
) -> dict:
    """Retrieve the newest matching finished run and verify its artifact."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{name}'",
        max_results=100,
        order_by=["attributes.start_time DESC"],
    )

    for run in runs:
        if run.info.status != "FINISHED":
            continue

        if run.data.tags.get("partition") != "validation":
            continue

        if run.data.params.get("source_sha256") != source_hash:
            continue

        with TemporaryDirectory() as directory:
            downloaded_path = client.download_artifacts(
                run.info.run_id,
                ARTIFACT_PATH,
                dst_path=directory,
            )
            artifact = json.loads(Path(downloaded_path).read_text(encoding="utf-8"))

        if artifact["run_name"] != name:
            raise ValueError(f"Artifact run name mismatch: {name}")

        if artifact["dataset_partition"] != "validation":
            raise ValueError("Artifact references the wrong partition")

        if artifact["source_sha256"] != source_hash:
            raise ValueError("Artifact dataset hash mismatch")

        if artifact["model"] != run.data.params["model_name"]:
            raise ValueError("Model parameter and artifact differ")

        threshold = float(run.data.params["threshold"])
        if artifact["threshold"] != threshold:
            raise ValueError("Threshold parameter and artifact differ")

        metrics = {}
        for key in METRICS:
            stored = run.data.metrics[key]
            reported = artifact["metrics"][key]

            if not math.isclose(stored, reported, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Metric mismatch: {name}/{key}")

            metrics[key] = reported

        return {
            "name": name,
            "run_id": run.info.run_id,
            "model": artifact["model"],
            "threshold": threshold,
            "partition": artifact["dataset_partition"],
            "source_sha256": source_hash,
            "validation_rows": artifact["validation_records"],
            "metrics": metrics,
        }

    raise RuntimeError(f"No matching finished run found: {name}")


def write_verified(path: Path, content: bytes) -> None:
    """Create once; reject unexpected changes on subsequent executions."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing report differs: {path}")
        print(f"VERIFIED: {path}")
    else:
        path.write_bytes(content)
        print(f"CREATED: {path}")


def main() -> None:
    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")

    verify_prepared_dataset(raw, processed)
    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT}")

    records = [
        get_record(client, experiment.experiment_id, name, source_hash) for name in EXPECTED_NAMES
    ]

    validate_comparison(records)

    report = {
        "experiment": EXPERIMENT,
        "dataset": "ai4i2020",
        "partition": "validation",
        "source_sha256": source_hash,
        "runs": records,
    }

    output_dir = Path("outputs/comparison")
    json_path = output_dir / "baselines.json"
    markdown_path = output_dir / "baselines.md"

    json_content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    markdown_content = build_markdown(records).encode("utf-8")

    # Check for conflicts before writing either report.
    for path, content in (
        (json_path, json_content),
        (markdown_path, markdown_content),
    ):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Existing report differs: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    write_verified(json_path, json_content)
    write_verified(markdown_path, markdown_content)

    print()
    print(build_markdown(records))
    print("MLFLOW EXPERIMENT COMPARISON: PASSED")
    print("New MLflow runs created: 0")
    print("Held-out test dataset: NOT USED")


if __name__ == "__main__":
    main()
