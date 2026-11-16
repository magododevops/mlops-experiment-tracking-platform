"""CLI verification of raw and prepared dataset integrity."""

import argparse
from pathlib import Path

from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
    verify_raw_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset integrity and reproducibility.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/ai4i2020.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/seed-42"),
    )
    args = parser.parse_args()

    digest = verify_raw_dataset(args.input)
    print(f"Raw dataset SHA-256: {digest}")
    print("Raw dataset integrity: PASSED")

    verify_prepared_dataset(args.input, args.output)
    print("Processed dataset integrity: PASSED")
    print("Reproducibility verification: PASSED")


if __name__ == "__main__":
    main()
