"""CLI entry point for dataset acquisition and inspection."""

import argparse
from pathlib import Path

from mlops_experiment_tracking.data.ingestion import (
    download_dataset,
    load_dataset,
)
from mlops_experiment_tracking.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    select_features,
)

DEFAULT_PATH = Path("data/raw/ai4i2020.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate the AI4I dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PATH,
        help="CSV destination path",
    )

    args = parser.parse_args()

    if args.output.exists():
        print(f"Using existing dataset: {args.output}")
    else:
        digest = download_dataset(args.output)
        print(f"Downloaded dataset. SHA-256: {digest}")

    frame = load_dataset(args.output)
    features = select_features(frame)

    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Features: {list(FEATURE_COLUMNS)}")
    print(f"Selected features: {features.shape[1]}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Failures: {int(frame[TARGET_COLUMN].sum())}")
    print("Data validation: PASSED")


if __name__ == "__main__":
    main()
