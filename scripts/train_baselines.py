"""CLI for fitting predictive-maintenance baseline models."""

import argparse
from pathlib import Path

import pandas as pd

from mlops_experiment_tracking.data.integrity import verify_prepared_dataset
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.training.models import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run predictive-maintenance baselines.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/ai4i2020.csv"),
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/seed-42"),
    )
    args = parser.parse_args()

    # Reject altered data before any model fitting.
    verify_prepared_dataset(args.raw, args.processed)

    train = pd.read_csv(args.processed / "train.csv")
    validation = pd.read_csv(args.processed / "validation.csv")

    x_validation = validation.loc[:, FEATURE_COLUMNS]

    print("Dataset integrity: PASSED")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")
    print(f"Training failures: {int(train[TARGET_COLUMN].sum())}")
    print()

    for name in ("dummy", "logistic_regression"):
        model = train_model(name, train)

        predictions = model.predict(x_validation)
        probabilities = model.predict_proba(x_validation)[:, 1]

        print(f"Model: {name}")
        print(f"Predictions generated: {len(predictions)}")
        print(f"Predicted failures: {int(predictions.sum())}")
        print(
            f"First five failure probabilities: {[round(float(p), 4) for p in probabilities[:5]]}"
        )
        print()

    print("Baseline training: PASSED")
    print("Validation metrics will be implemented in M3.2")


if __name__ == "__main__":
    main()
