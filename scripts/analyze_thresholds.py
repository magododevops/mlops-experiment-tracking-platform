"""Analyze fixed decision thresholds on validation data only."""

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
from mlops_experiment_tracking.training.models import train_model
from mlops_experiment_tracking.training.thresholds import evaluate_thresholds


def main() -> None:
    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    output = Path("outputs/evaluation/thresholds.json")

    verify_prepared_dataset(raw, processed)

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")

    model = train_model("logistic_regression", train)

    scores = model.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1]

    probabilities = pd.Series(scores)
    actual = validation[TARGET_COLUMN]

    results = evaluate_thresholds(actual, probabilities)

    print()
    print("Threshold | Precision | Recall | F1     | TP | FP | FN")
    print("-" * 63)

    for result in results:
        print(
            f"{result['threshold']:>9.2f} | "
            f"{result['precision']:>9.4f} | "
            f"{result['recall']:>6.4f} | "
            f"{result['f1']:>6.4f} | "
            f"{result['true_positive']:>2} | "
            f"{result['false_positive']:>2} | "
            f"{result['false_negative']:>2}"
        )

    report = {
        "model": "logistic_regression",
        "dataset": "validation",
        "thresholds": results,
        "note": "Exploratory comparison; no threshold promoted.",
    }

    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError(f"Existing threshold analysis differs: {output}")
        print(f"\nVERIFIED: {output}")
    else:
        output.write_bytes(content)
        print(f"\nCREATED: {output}")

    print("Threshold analysis: PASSED")


if __name__ == "__main__":
    main()
