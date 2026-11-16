"""Read-only promotion gate for the existing MLflow candidate."""

import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.registry.metadata import MODEL_NAME
from mlops_experiment_tracking.registry.promotion import (
    evaluate_promotion,
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

ALIAS = "candidate"


def main() -> None:
    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    output = Path("outputs/registry/promotion_gate.json")

    verify_prepared_dataset(raw, processed)
    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")
    features = validation.loc[:, FEATURE_COLUMNS]
    target = validation[TARGET_COLUMN]

    # Independently recompute the original selection on validation.
    reference_model = train_model("logistic_regression", train)
    reference_scores = reference_model.predict_proba(features)[:, 1]

    candidates = evaluate_thresholds(
        target,
        pd.Series(reference_scores),
    )
    selected = verify_selection(
        load_selection(Path("outputs/evaluation/selection.json")),
        candidates,
        source_hash,
        len(validation),
    )

    threshold = float(selected["threshold"])

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    if str(version.version) != "1":
        raise RuntimeError("Unexpected candidate model version")

    if not version.run_id:
        raise RuntimeError("Registered model has no source Run ID")

    source_run = client.get_run(version.run_id)

    if source_run.info.status != "FINISHED":
        raise RuntimeError("Source training run is not finished")

    if source_run.data.params.get("dataset_sha256") != source_hash:
        raise RuntimeError("Source run dataset hash mismatch")

    if source_run.data.params.get("model_name") != "logistic_regression":
        raise RuntimeError("Unexpected source run model")

    try:
        recorded_threshold = float(source_run.data.params["decision_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Invalid source run threshold") from error

    if recorded_threshold != threshold:
        raise RuntimeError("Source run threshold mismatch")

    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    registered_model = mlflow.sklearn.load_model(model_uri)

    classes = list(registered_model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)

    scores = registered_model.predict_proba(features)[:, positive_index]
    predictions = (scores >= threshold).astype(int)

    metrics = evaluate_binary_classifier(
        target.to_numpy(),
        predictions,
        scores,
    )

    # The registered artifact must reproduce the selected validation metrics.
    for key in ("recall", "f1", "false_positive", "true_positive"):
        expected = selected[key]
        actual = getattr(metrics, key)

        if abs(float(actual) - float(expected)) > 1e-12:
            raise RuntimeError(f"Selected metric mismatch: {key}")

    for key, expected in (
        ("validation_recall", metrics.recall),
        ("validation_f1", metrics.f1),
    ):
        recorded = source_run.data.metrics.get(key)
        if recorded is None or abs(recorded - expected) > 1e-12:
            raise RuntimeError(f"Source run metric mismatch: {key}")

    decision = evaluate_promotion(
        metrics,
        version.tags,
        expected_sha256=source_hash,
        expected_threshold=threshold,
    )

    report = {
        "registered_model": MODEL_NAME,
        "version": str(version.version),
        "alias": ALIAS,
        "source_run_id": version.run_id,
        "dataset_sha256": source_hash,
        "evaluation_partition": "validation",
        "threshold": threshold,
        "metrics": metrics.to_dict(),
        "quality_gate": decision.to_dict(),
        "promotion_performed": False,
        "production_approved": False,
    }

    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")

    # Never replace previously generated evidence silently.
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Existing report differs: {output}")

    if not decision.eligible:
        print("PROMOTION GATE: REJECTED")
        for reason in decision.reasons:
            print(f"- {reason}")
        raise SystemExit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"VERIFIED: {output}")
    else:
        output.write_bytes(content)
        print(f"CREATED: {output}")

    print()
    print("MODEL PROMOTION GATE")
    print("--------------------")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {version.version}")
    print(f"Alias: {ALIAS}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Validation recall: {metrics.recall:.4f}")
    print(f"Validation F1: {metrics.f1:.4f}")
    print(f"False positives: {metrics.false_positive}")
    print("Source run: VERIFIED")
    print("Dataset provenance: VERIFIED")
    print("Registry artifact: VERIFIED")
    print("Promotion gate: ELIGIBLE")
    print("Production approval: NOT GRANTED")
    print("Registry mutations: NONE")
    print("MODEL PROMOTION VALIDATION: PASSED")


if __name__ == "__main__":
    main()
