"""Apply an experimental model quality gate on validation data."""

import hashlib
import json
from dataclasses import asdict
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
from mlops_experiment_tracking.training.selection import (
    SelectionPolicy,
    select_candidate,
)
from mlops_experiment_tracking.training.thresholds import (
    evaluate_thresholds,
)


def main() -> None:
    raw = Path("data/raw/ai4i2020.csv")
    processed = Path("data/processed/seed-42")
    output = Path("outputs/evaluation/selection.json")

    verify_prepared_dataset(raw, processed)

    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_csv(processed / "validation.csv")

    model = train_model("logistic_regression", train)

    probabilities = pd.Series(model.predict_proba(validation.loc[:, FEATURE_COLUMNS])[:, 1])

    candidates = evaluate_thresholds(
        validation[TARGET_COLUMN],
        probabilities,
    )

    policy = SelectionPolicy()
    selected = select_candidate(candidates, policy)

    report = {
        "model": "logistic_regression",
        "partition": "validation",
        "status": "candidate_selected",
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "validation_rows": len(validation),
        "policy": asdict(policy),
        "selection_criteria": [
            "highest_f1",
            "highest_recall",
            "lowest_threshold",
        ],
        "candidate_count": len(candidates),
        "selected": selected,
        "note": (
            "Experimental selection only. No final test "
            "evaluation or production promotion performed."
        ),
    }

    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError(f"Existing report differs: {output}")
        print(f"VERIFIED: {output}")
    else:
        output.write_bytes(content)
        print(f"CREATED: {output}")

    print()
    print("MODEL SELECTION RESULT")
    print("----------------------")
    print("Model: logistic_regression")
    print(f"Threshold: {selected['threshold']:.2f}")
    print(f"Recall: {selected['recall']:.4f}")
    print(f"Precision: {selected['precision']:.4f}")
    print(f"F1-score: {selected['f1']:.4f}")
    print(f"False positives: {selected['false_positive']}")
    print("Quality gate: PASSED")
    print("Final test set: NOT USED")


if __name__ == "__main__":
    main()
