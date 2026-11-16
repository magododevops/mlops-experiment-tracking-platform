"""Validate and query an existing MLflow model version."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.inference.contract import (
    SAMPLE_EQUIPMENT,
    classify_failure,
    validate_equipment,
)
from mlops_experiment_tracking.registry.metadata import (
    MODEL_NAME,
    CandidateMetadata,
)
from mlops_experiment_tracking.training.finalization import load_selection

VERSION = "1"
CANDIDATE_ALIAS = "candidate"


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform a validated registry-model prediction.")
    parser.add_argument(
        "--input-json",
        type=Path,
        help="JSON file containing one equipment record",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")

    verify_prepared_dataset(raw, processed)

    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    selection = load_selection(Path("outputs/evaluation/selection.json"))

    if selection.get("source_sha256") != source_hash:
        raise ValueError("Selection dataset hash mismatch")

    if selection.get("model") != "logistic_regression":
        raise ValueError("Unexpected selected model")

    if selection.get("status") != "candidate_selected":
        raise ValueError("Selection status mismatch")

    selected = selection["selected"]

    if not isinstance(selected, dict):
        raise ValueError("Invalid selection metadata")

    version = client.get_model_version(MODEL_NAME, VERSION)

    tags = version.tags

    metadata = CandidateMetadata(
        threshold=float(tags["classification_threshold"]),
        dataset_sha256=tags["dataset_sha256"],
        validation_f1=float(tags["validation_f1"]),
        validation_recall=float(tags["validation_recall"]),
    )

    if tags.get("lifecycle") != "candidate":
        raise ValueError("Model is not a candidate")

    if tags.get("approval") != "not_approved":
        raise ValueError("Unexpected model approval state")

    if tags.get("model_algorithm") != "logistic_regression":
        raise ValueError("Unexpected model algorithm")

    if metadata.dataset_sha256 != source_hash:
        raise ValueError("Registry dataset hash mismatch")

    if metadata.threshold != float(selected["threshold"]):
        raise ValueError("Registry threshold differs from selection")

    if metadata.validation_f1 != float(selected["f1"]):
        raise ValueError("Registry F1 differs from selection")

    if metadata.validation_recall != float(selected["recall"]):
        raise ValueError("Registry recall differs from selection")

    if args.input_json:
        equipment = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        equipment = SAMPLE_EQUIPMENT

    if not isinstance(equipment, dict):
        raise ValueError("Input must be a JSON object")

    features = validate_equipment(equipment)

    model_uri = f"models:/{MODEL_NAME}/{VERSION}"
    model = mlflow.sklearn.load_model(model_uri)

    probabilities = model.predict_proba(features)

    if probabilities.shape != (1, 2):
        raise ValueError("Unexpected prediction output shape")

    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(1)

    failure_probability = float(probabilities[0, positive_index])

    prediction = classify_failure(failure_probability, metadata.threshold)

    # Validate both model outputs and registry metadata before
    # assigning the candidate alias.
    registered_model = client.get_registered_model(MODEL_NAME)
    aliases = registered_model.aliases or {}
    existing = aliases.get(CANDIDATE_ALIAS)

    if existing is not None and str(existing) != VERSION:
        raise RuntimeError("Candidate alias points to another version; refusing to overwrite it.")

    if existing is None:
        client.set_registered_model_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS,
            VERSION,
        )
        print("Candidate alias: CREATED")
    else:
        print("Candidate alias: VERIFIED")

    resolved = client.get_model_version_by_alias(MODEL_NAME, CANDIDATE_ALIAS)

    if str(resolved.version) != VERSION:
        raise RuntimeError("Candidate alias resolution mismatch")

    print()
    print("REGISTERED MODEL INFERENCE")
    print("--------------------------")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Model version: {VERSION}")
    print(f"Model URI: {model_uri}")
    print(f"Candidate alias: {CANDIDATE_ALIAS}")
    print(f"Failure probability: {failure_probability:.4%}")
    print(f"Decision threshold: {metadata.threshold:.0%}")
    print(f"Prediction: {prediction}")
    print("Input validation: PASSED")
    print("Registry metadata: VERIFIED")
    print("Production promotion: NOT PERFORMED")
    print("MODEL INFERENCE: PASSED")


if __name__ == "__main__":
    main()
