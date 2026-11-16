"""Evaluate a locked candidate once on the reserved test partition."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
)
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
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


def main() -> None:
    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    selection_path = Path("outputs/evaluation/selection.json")
    output = Path("outputs/evaluation/final_test.json")

    # Verify all data before evaluating the locked candidate.
    verify_prepared_dataset(raw, processed)

    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")

    model = train_model("logistic_regression", train)

    # Recompute validation metrics to verify the selected configuration.
    validation_scores = pd.Series(model.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1])

    candidates = evaluate_thresholds(
        validation[TARGET_COLUMN],
        validation_scores,
    )

    selection_report = load_selection(selection_path)

    selected = verify_selection(
        selection_report,
        candidates,
        source_hash,
        len(validation),
    )

    threshold = float(selected["threshold"])

    print("Candidate integrity: PASSED")
    print("Locked model: logistic_regression")
    print(f"Locked threshold: {threshold:.2f}")

    # The test partition is evaluated only after candidate verification.
    test = pd.read_csv(processed / "test.csv")
    test_features = test.loc[:, FEATURE_COLUMNS]
    test_target = test[TARGET_COLUMN]

    test_scores = model.predict_proba(test_features)[:, 1]
    test_predictions = (test_scores >= threshold).astype(int)

    metrics = evaluate_binary_classifier(
        test_target.to_numpy(),
        test_predictions,
        test_scores,
    )

    report = {
        "model": "logistic_regression",
        "threshold": threshold,
        "evaluation_partition": "test",
        "selection_partition": "validation",
        "source_sha256": source_hash,
        "random_state": 42,
        "training_rows": len(train),
        "test_rows": len(test),
        "test_failures": int(test_target.sum()),
        "metrics": metrics.to_dict(),
        "limitations": [
            "Synthetic dataset",
            "No real-world production validation",
            "No operational cost model",
            "No external temporal validation",
        ],
        "note": (
            "Final held-out evaluation. Do not tune the model or threshold using these results."
        ),
    }

    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError(
                f"Final report differs: {output}. Do not overwrite the held-out evaluation."
            )
        print(f"VERIFIED: {output}")
    else:
        output.write_bytes(content)
        print(f"CREATED: {output}")

    print()
    print("FINAL HELD-OUT TEST EVALUATION")
    print("------------------------------")
    print(f"Test records:      {len(test)}")
    print(f"Actual failures:   {int(test_target.sum())}")
    print(f"Accuracy:          {metrics.accuracy:.4f}")
    print(f"Precision:         {metrics.precision:.4f}")
    print(f"Recall:            {metrics.recall:.4f}")
    print(f"F1-score:          {metrics.f1:.4f}")
    print(f"Average Precision: {metrics.average_precision:.4f}")
    print()
    print(f"True negatives:    {metrics.true_negative}")
    print(f"False positives:   {metrics.false_positive}")
    print(f"False negatives:   {metrics.false_negative}")
    print(f"True positives:    {metrics.true_positive}")
    print()
    print("Final evaluation: COMPLETED")
    print("Production readiness: NOT ASSESSED")


if __name__ == "__main__":
    main()
