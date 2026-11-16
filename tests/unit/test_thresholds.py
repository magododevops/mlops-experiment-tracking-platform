"""Tests for decision threshold analysis."""

import pandas as pd

from mlops_experiment_tracking.training.thresholds import (
    THRESHOLDS,
    evaluate_thresholds,
)


def test_evaluates_all_predefined_thresholds() -> None:
    actual = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.10, 0.40, 0.30, 0.90])

    results = evaluate_thresholds(actual, scores)

    assert len(results) == len(THRESHOLDS)
    assert [r["threshold"] for r in results] == list(THRESHOLDS)


def test_lower_threshold_cannot_reduce_recall() -> None:
    actual = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.10, 0.40, 0.30, 0.90])

    results = evaluate_thresholds(actual, scores)
    recalls = [result["recall"] for result in results]

    assert recalls == sorted(recalls, reverse=True)


def test_threshold_changes_alert_count() -> None:
    actual = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.10, 0.40, 0.30, 0.90])

    results = evaluate_thresholds(actual, scores)

    low = results[0]
    high = results[-1]

    low_alerts = low["true_positive"] + low["false_positive"]
    high_alerts = high["true_positive"] + high["false_positive"]

    assert low_alerts >= high_alerts
    assert high["true_positive"] == 1


def test_probability_ranking_metrics_do_not_change() -> None:
    actual = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.10, 0.40, 0.30, 0.90])

    results = evaluate_thresholds(actual, scores)

    ap_values = {r["average_precision"] for r in results}
    assert len(ap_values) == 1
