"""Prepare deterministic ML datasets and a provenance manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from mlops_experiment_tracking.data.ingestion import load_dataset
from mlops_experiment_tracking.data.integrity import (
    verify_prepared_dataset,
    verify_raw_dataset,
)
from mlops_experiment_tracking.data.profiling import build_profile
from mlops_experiment_tracking.data.schema import TARGET_COLUMN
from mlops_experiment_tracking.data.splitting import (
    RANDOM_STATE,
    split_dataset,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_verified(path: Path, content: bytes) -> None:
    """Write new output or verify an existing identical output."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Existing output differs: {path}. Choose another output directory."
            )
        print(f"VERIFIED: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f"CREATED: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible ML dataset partitions.")
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

    verify_raw_dataset(args.input)
    frame = load_dataset(args.input)
    partitions = split_dataset(frame, random_state=RANDOM_STATE)

    datasets = {
        "train": partitions.train,
        "validation": partitions.validation,
        "test": partitions.test,
    }

    manifest = {
        "source_sha256": sha256(args.input.read_bytes()),
        "random_state": RANDOM_STATE,
        "split_strategy": "stratified_random_60_20_20",
        "target": TARGET_COLUMN,
        "partitions": {},
    }

    output_files: dict[str, bytes] = {}

    for name, dataset in datasets.items():
        content = dataset.to_csv(index=False).encode("utf-8")
        output_files[f"{name}.csv"] = content

        manifest["partitions"][name] = {
            "rows": len(dataset),
            "failures": int(dataset[TARGET_COLUMN].sum()),
            "failure_rate": round(float(dataset[TARGET_COLUMN].mean()), 6),
            "sha256": sha256(content),
            "source_row_indices_sha256": sha256(",".join(map(str, dataset.index)).encode("utf-8")),
        }

    profile = build_profile(partitions.train)

    output_files["profile.json"] = (json.dumps(profile, indent=2) + "\n").encode("utf-8")

    output_files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    # Validate all preexisting outputs before creating anything.
    for filename, content in output_files.items():
        path = args.output / filename
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Output conflict: {path}")

    for filename, content in output_files.items():
        write_verified(args.output / filename, content)

    print()
    verify_prepared_dataset(args.input, args.output)
    print("Dataset preparation PASSED")
    for name, dataset in datasets.items():
        failures = int(dataset[TARGET_COLUMN].sum())
        print(f"{name}: {len(dataset)} rows | {failures} failures")


if __name__ == "__main__":
    main()
