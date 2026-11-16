"""Pure validation and reporting logic for tracked experiments."""

import math
from typing import Any

EXPECTED_NAMES = (
    "dummy-baseline",
    "logistic-default",
    "logistic-selected",
)

METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "average_precision",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)


def validate_comparison(records: list[dict[str, Any]]) -> None:
    """Reject incomparable or inconsistent experiment records."""
    if len(records) != len(EXPECTED_NAMES):
        raise ValueError("Expected exactly three experiment runs")

    names = [record["name"] for record in records]
    if names != list(EXPECTED_NAMES):
        raise ValueError("Missing, duplicated, or unordered run names")

    if len({record["source_sha256"] for record in records}) != 1:
        raise ValueError("Runs have different dataset hashes")

    if any(record["partition"] != "validation" for record in records):
        raise ValueError("Only validation runs can be compared")

    if any(record["validation_rows"] != 2000 for record in records):
        raise ValueError("Unexpected validation dataset size")

    for record in records:
        metrics = record["metrics"]

        for key in METRICS:
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid metric type: {key}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid metric: {key}")

        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
        ):
            if metrics[key] > 1:
                raise ValueError(f"Metric outside [0, 1]: {key}")

        for key in (
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ):
            if not float(metrics[key]).is_integer():
                raise ValueError(f"Count must be integral: {key}")

        total = sum(
            metrics[key]
            for key in (
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            )
        )
        if total != record["validation_rows"]:
            raise ValueError("Confusion matrix does not match dataset")

    if records[0]["model"] != "dummy":
        raise ValueError("Invalid dummy model")

    if any(record["model"] != "logistic_regression" for record in records[1:]):
        raise ValueError("Invalid logistic model")

    if records[0]["threshold"] != 0.5:
        raise ValueError("Unexpected dummy threshold")

    if records[1]["threshold"] != 0.5:
        raise ValueError("Unexpected default threshold")

    if not 0 < records[2]["threshold"] < 1:
        raise ValueError("Unexpected selected threshold")

    # Threshold changes predictions, not probability ranking.
    if not math.isclose(
        records[1]["metrics"]["average_precision"],
        records[2]["metrics"]["average_precision"],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Logistic runs have different ranking metrics")


def build_markdown(records: list[dict[str, Any]]) -> str:
    """Generate a readable comparison without changing model policy."""
    validate_comparison(records)

    lines = [
        "# MLflow Baseline Comparison",
        "",
        "Dataset: AI4I 2020",
        "",
        "Partition: validation (2,000 records)",
        "",
        "This report is observational, not a production approval.",
        "",
        "| Run | Threshold | Precision | Recall | F1 | AP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for record in records:
        m = record["metrics"]
        lines.append(
            f"| {record['name']} "
            f"| {record['threshold']:.2f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['average_precision']:.4f} "
            f"| {int(m['false_positive'])} "
            f"| {int(m['false_negative'])} |"
        )

    lines.extend(
        [
            "",
            "## Run provenance",
            "",
            "| Run | MLflow Run ID |",
            "|---|---|",
        ]
    )

    for record in records:
        lines.append(f"| {record['name']} | `{record['run_id']}` |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Higher recall detects more failures, potentially with more false alarms.",
            "",
            "The selected threshold was established before the held-out test evaluation.",
            "",
            "Do not use this report to retune against the test dataset.",
            "",
        ]
    )

    return "\n".join(lines)
