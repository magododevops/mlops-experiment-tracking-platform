"""Register a verified scikit-learn candidate in MLflow."""

import hashlib
import importlib.metadata
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.registry.metadata import (
    MODEL_NAME,
    CandidateMetadata,
)
from mlops_experiment_tracking.training.finalization import (
    load_selection,
    verify_selection,
)
from mlops_experiment_tracking.training.models import train_model
from mlops_experiment_tracking.training.thresholds import evaluate_thresholds

EXPERIMENT = "predictive-maintenance-registry"


def ensure_not_registered(client: MlflowClient) -> None:
    """Avoid silently generating additional registry versions."""
    try:
        existing = client.get_registered_model(MODEL_NAME)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return
        raise

    raise RuntimeError(
        f"Model already exists: {existing.name}. Inspect its versions instead of registering again."
    )


def main() -> None:
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # This is an intentional one-time registration.
    ensure_not_registered(client)

    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    selection = Path("outputs/evaluation/selection.json")

    verify_prepared_dataset(raw, processed)
    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")

    model = train_model("logistic_regression", train)

    x_validation = validation.loc[:, FEATURE_COLUMNS]
    probabilities = model.predict_proba(x_validation)[:, 1]

    candidates = evaluate_thresholds(
        validation[TARGET_COLUMN],
        pd.Series(probabilities),
    )
    selected = verify_selection(
        load_selection(selection),
        candidates,
        source_hash,
        len(validation),
    )

    metadata = CandidateMetadata(
        threshold=float(selected["threshold"]),
        dataset_sha256=source_hash,
        validation_f1=float(selected["f1"]),
        validation_recall=float(selected["recall"]),
    )

    print("Dataset integrity: PASSED")
    print("Candidate selection: VERIFIED")
    print(f"Threshold: {metadata.threshold:.2f}")

    sample = train.loc[:4, FEATURE_COLUMNS].copy()
    signature = infer_signature(sample, model.predict(sample))

    requirements = [
        f"scikit-learn=={importlib.metadata.version('scikit-learn')}",
        f"pandas=={importlib.metadata.version('pandas')}",
        f"numpy=={importlib.metadata.version('numpy')}",
        f"skops=={importlib.metadata.version('skops')}",
    ]

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(
        run_name="logistic-regression-candidate-v1",
        tags={
            "purpose": "model-registration",
            "partition": "validation",
            "production_ready": "false",
        },
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model_name": "logistic_regression",
                "decision_threshold": metadata.threshold,
                "random_state": 42,
                "dataset_sha256": source_hash,
            }
        )
        mlflow.log_metrics(
            {
                "validation_f1": metadata.validation_f1,
                "validation_recall": metadata.validation_recall,
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="classifier",
            input_example=sample,
            signature=signature,
            serialization_format="skops",
            pip_requirements=requirements,
            registered_model_name=MODEL_NAME,
        )

    # Find the model version associated with this run.
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    matches = [version for version in versions if version.run_id == run_id]

    if len(matches) != 1:
        raise RuntimeError("Expected exactly one model version for the new run")

    version = matches[0]

    for key, value in metadata.tags().items():
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version.version,
            key=key,
            value=value,
        )

    # Reload from S3 through the Model Registry.
    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    restored = mlflow.sklearn.load_model(model_uri)

    restored_scores = restored.predict_proba(x_validation)[:, 1]

    np.testing.assert_allclose(
        probabilities,
        restored_scores,
        rtol=0,
        atol=1e-12,
    )

    original_predictions = (probabilities >= metadata.threshold).astype(int)

    restored_predictions = (restored_scores >= metadata.threshold).astype(int)

    np.testing.assert_array_equal(
        original_predictions,
        restored_predictions,
    )

    stored_version = client.get_model_version(
        MODEL_NAME,
        version.version,
    )

    for key, value in metadata.tags().items():
        if stored_version.tags.get(key) != value:
            raise RuntimeError(f"Registry tag differs: {key}")

    print()
    print("MODEL REGISTRY RESULT")
    print("---------------------")
    print(f"Registered model: {MODEL_NAME}")
    print(f"Version: {version.version}")
    print(f"Source run: {run_id}")
    print(f"MLflow model URI: {model_info.model_uri}")
    print(f"Registry URI: {model_uri}")
    print(f"Threshold metadata: {metadata.threshold:.2f}")
    print("Model reload: VERIFIED")
    print("Probabilities: IDENTICAL")
    print("Thresholded predictions: IDENTICAL")
    print("Version metadata: VERIFIED")
    print("Model status: CANDIDATE")
    print("Production promotion: NOT PERFORMED")
    print("MODEL REGISTRATION: PASSED")


if __name__ == "__main__":
    main()
