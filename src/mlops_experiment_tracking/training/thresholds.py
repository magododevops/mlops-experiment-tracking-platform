"""Reusable decision threshold evaluation."""

import pandas as pd

from mlops_experiment_tracking.training.evaluation import evaluate_binary_classifier

THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)


def evaluate_thresholds(
    actual: pd.Series,
    probabilities: pd.Series,
) -> list[dict]:
    """Evaluate predefined thresholds without refitting."""
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        metrics = evaluate_binary_classifier(
            actual.to_numpy(),
            predictions.to_numpy(),
            probabilities.to_numpy(),
        )

        results.append(
            {
                "threshold": threshold,
                **metrics.to_dict(),
            }
        )

    return results
