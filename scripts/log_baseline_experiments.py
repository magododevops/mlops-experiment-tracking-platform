"""Train baselines and register validation experiments in MLflow."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.tracking.experiments import (
    build_run_configs,
)
from mlops_experiment_tracking.training.evaluation import (
    evaluate_binary_classifier,
)
from mlops_experiment_tracking.training.finalization import (
    load_selection,
    verify_selection,
)
from mlops_experiment_tracking.training.models import train_model
from mlops_experiment_tracking.training.thresholds import evaluate_thresholds

EXPERIMENT_NAME = "predictive-maintenance-baselines"


def main() -> None:
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    selection_path = Path("outputs/evaluation/selection.json")

    # Never register experiments using modified or unknown data.
    verify_prepared_dataset(raw, processed)

    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")

    x_validation = validation.loc[:, FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    print("Dataset integrity: PASSED")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")

    # Train each model once and reuse the fitted model.
    models = {name: train_model(name, train) for name in ("dummy", "logistic_regression")}

    probabilities = {
        name: model.predict_proba(x_validation)[:, 1] for name, model in models.items()
    }

    # Recalculate the validation selection before logging it.
    candidates = evaluate_thresholds(
        y_validation,
        pd.Series(probabilities["logistic_regression"]),
    )

    selected = verify_selection(
        load_selection(selection_path),
        candidates,
        source_hash,
        len(validation),
    )

    selected_threshold = float(selected["threshold"])
    configs = build_run_configs(selected_threshold)

    print(f"Selected threshold verified: {selected_threshold:.2f}")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment.experiment_id}")

    for config in configs:
        scores = probabilities[config.model_name]
        predictions = (scores >= config.threshold).astype(int)

        metrics = evaluate_binary_classifier(
            y_validation.to_numpy(),
            predictions,
            scores,
        )

        metric_values = metrics.to_dict()

        artifact = {
            "model": config.model_name,
            "run_name": config.name,
            "threshold": config.threshold,
            "dataset_partition": "validation",
            "source_sha256": source_hash,
            "training_records": len(train),
            "validation_records": len(validation),
            "validation_failures": int(y_validation.sum()),
            "metrics": metric_values,
        }

        with mlflow.start_run(
            run_name=config.name,
            tags={
                "project": "mlops-experiment-tracking-platform",
                "dataset": "ai4i2020",
                "partition": "validation",
                "purpose": "baseline-comparison",
            },
        ) as run:
            run_id = run.info.run_id

            mlflow.log_params(
                {
                    "model_name": config.model_name,
                    "threshold": config.threshold,
                    "random_state": 42,
                    "training_records": len(train),
                    "validation_records": len(validation),
                    "source_sha256": source_hash,
                }
            )
            mlflow.log_metrics(metric_values)
            mlflow.log_dict(
                artifact,
                "evaluation/validation.json",
            )

        # Confirm that the server persisted the expected metadata.
        stored = client.get_run(run_id)

        if stored.info.status != "FINISHED":
            raise RuntimeError(f"Run did not finish: {run_id}")

        if stored.data.params["model_name"] != config.model_name:
            raise RuntimeError("Stored model parameter differs")

        if abs(stored.data.metrics["recall"] - metrics.recall) > 1e-12:
            raise RuntimeError("Stored recall differs")

        # Confirm that the artifact can be retrieved through MLflow.
        with TemporaryDirectory() as directory:
            downloaded_path = client.download_artifacts(
                run_id,
                "evaluation/validation.json",
                dst_path=directory,
            )
            downloaded = json.loads(Path(downloaded_path).read_text(encoding="utf-8"))

        if downloaded != artifact:
            raise RuntimeError("Stored artifact differs")

        print()
        print(f"Run: {config.name}")
        print(f"Run ID: {run_id}")
        print(f"Model: {config.model_name}")
        print(f"Threshold: {config.threshold:.2f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1: {metrics.f1:.4f}")
        print(f"Average Precision: {metrics.average_precision:.4f}")
        print("Metadata: VERIFIED")
        print("Artifact: VERIFIED")

    print()
    print("REAL EXPERIMENT TRACKING: PASSED")
    print("Final test dataset: NOT USED")


if __name__ == "__main__":
    main()
