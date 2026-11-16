"""Unit tests for structured pipeline observability."""

from copy import deepcopy

import pytest

from mlops_experiment_tracking.observability.execution import (
    structured_event,
    validate_manifest,
    validate_observed_run,
)

EXECUTION_ID = "a" * 32
COMMIT = "b" * 40
HASH = "c" * 64
EXPERIMENT_ID = "2"


@pytest.fixture
def manifest() -> dict:
    names = (
        "dummy-baseline",
        "logistic-default",
        "logistic-selected",
    )

    runs = [
        {
            "name": name,
            "run_id": str(index + 1) * 32,
            "status": "FINISHED",
            "metrics": {
                "precision": 0.4,
                "recall": 0.44,
                "f1": 0.42,
                "average_precision": 0.405,
            },
        }
        for index, name in enumerate(names)
    ]

    return {
        "schema_version": 1,
        "pipeline": "baseline-training",
        "pipeline_execution_id": EXECUTION_ID,
        "status": "SUCCEEDED",
        "git_commit": COMMIT,
        "dataset": {
            "source_sha256": HASH,
            "validation_partition": "validation",
        },
        "mlflow": {
            "experiment_name": "predictive-maintenance-baselines",
            "experiment_id": EXPERIMENT_ID,
            "runs": runs,
        },
        "held_out_test_used": False,
        "model_registry_modified": False,
    }


@pytest.fixture
def observed(manifest: dict) -> dict:
    run = manifest["mlflow"]["runs"][0]
    return {
        **deepcopy(run),
        "experiment_id": EXPERIMENT_ID,
        "tags": {
            "pipeline_execution_id": EXECUTION_ID,
            "source_git_commit": COMMIT,
            "pipeline_type": "baseline-training",
        },
        "params": {
            "source_sha256": HASH,
            "validation_records": "2000",
        },
    }


def check_run(manifest: dict, observed: dict) -> None:
    validate_observed_run(
        manifest["mlflow"]["runs"][0],
        observed,
        execution_id=EXECUTION_ID,
        git_commit=COMMIT,
        dataset_hash=HASH,
        experiment_id=EXPERIMENT_ID,
    )


def test_valid_manifest(manifest: dict) -> None:
    validate_manifest(manifest)


def test_rejects_failed_manifest(manifest: dict) -> None:
    manifest["status"] = "FAILED"
    with pytest.raises(ValueError, match="not successful"):
        validate_manifest(manifest)


def test_rejects_invalid_commit(manifest: dict) -> None:
    manifest["git_commit"] = "unknown"
    with pytest.raises(ValueError, match="Git commit"):
        validate_manifest(manifest)


def test_rejects_test_partition(manifest: dict) -> None:
    manifest["dataset"]["validation_partition"] = "test"
    with pytest.raises(ValueError, match="partition"):
        validate_manifest(manifest)


def test_rejects_registry_mutation(manifest: dict) -> None:
    manifest["model_registry_modified"] = True
    with pytest.raises(ValueError, match="registry"):
        validate_manifest(manifest)


def test_valid_observed_run(manifest: dict, observed: dict) -> None:
    check_run(manifest, observed)


def test_rejects_wrong_execution_tag(manifest: dict, observed: dict) -> None:
    observed["tags"]["pipeline_execution_id"] = "wrong"
    with pytest.raises(ValueError, match="tag mismatch"):
        check_run(manifest, observed)


def test_rejects_metric_drift(manifest: dict, observed: dict) -> None:
    observed["metrics"]["f1"] = 0.1
    with pytest.raises(ValueError, match="metric mismatch"):
        check_run(manifest, observed)


def test_rejects_wrong_experiment(manifest: dict, observed: dict) -> None:
    observed["experiment_id"] = "999"
    with pytest.raises(ValueError, match="another experiment"):
        check_run(manifest, observed)


def test_structured_event_contains_context() -> None:
    import json

    event = json.loads(
        structured_event(
            "RUN_VERIFIED",
            execution_id=EXECUTION_ID,
            run_id="1" * 32,
        )
    )

    assert event["level"] == "INFO"
    assert event["event"] == "RUN_VERIFIED"
    assert event["pipeline_execution_id"] == EXECUTION_ID
    assert event["run_id"] == "1" * 32
    assert event["timestamp"]
