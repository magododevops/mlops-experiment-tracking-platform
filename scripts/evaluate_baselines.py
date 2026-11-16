"""Evaluate baseline classifiers on the validation partition only."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from mlops_experiment_tracking.data.integrity import verify_prepared_dataset
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from mlops_experiment_tracking.training.evaluation import (
    evaluate_binary_classifier,
)
from mlops_experiment_tracking.training.models import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictive maintenance baselines.")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/baselines.json"),
    )
    args = parser.parse_args()

    verify_prepared_dataset(args.raw, args.processed)

    train = pd.read_csv(args.processed / "train.csv")
    validation = pd.read_csv(args.processed / "validation.csv")

    features = validation.loc[:, FEATURE_COLUMNS]
    target = validation[TARGET_COLUMN]

    report = {
        "dataset_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "random_state": 42,
        "evaluation_partition": "validation",
        "validation_rows": len(validation),
        "validation_failures": int(target.sum()),
        "models": {},
    }

    for name in ("dummy", "logistic_regression"):
        model = train_model(name, train)

        predictions = model.predict(features)
        positive_class_index = list(model.named_steps["classifier"].classes_).index(1)
        scores = model.predict_proba(features)[:, positive_class_index]

        metrics = evaluate_binary_classifier(
            target.to_numpy(),
            predictions,
            scores,
        )
        report["models"][name] = metrics.to_dict()

        print(f"\nModel: {name}")
        print(f"Accuracy:          {metrics.accuracy:.4f}")
        print(f"Precision:         {metrics.precision:.4f}")
        print(f"Recall:            {metrics.recall:.4f}")
        print(f"F1-score:          {metrics.f1:.4f}")
        print(f"Average Precision: {metrics.average_precision:.4f}")
        print(
            "Confusion matrix (TN, FP, FN, TP): "
            f"{metrics.true_negative}, "
            f"{metrics.false_positive}, "
            f"{metrics.false_negative}, "
            f"{metrics.true_positive}"
        )

    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        if args.output.read_bytes() != content:
            raise FileExistsError(f"Existing evaluation differs: {args.output}")
        print(f"\nVERIFIED: {args.output}")
    else:
        args.output.write_bytes(content)
        print(f"\nCREATED: {args.output}")

    print("Baseline evaluation: PASSED")


if __name__ == "__main__":
    main()
