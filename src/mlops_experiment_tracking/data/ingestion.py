"""Secure and reproducible acquisition of the UCI dataset."""

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from mlops_experiment_tracking.data.schema import validate_dataset

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
DATASET_FILE = "ai4i2020.csv"

MAX_DOWNLOAD_BYTES = 10_000_000
MAX_CSV_BYTES = 5_000_000


def download_dataset(destination: Path) -> str:
    """Download, validate, and persist the official CSV.

    Returns the SHA-256 digest of the raw CSV.
    """
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; refusing to overwrite")

    destination.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        DATASET_URL,
        headers={"User-Agent": "mlops-experiment-tracking/0.1"},
    )

    with urlopen(request, timeout=30) as response:
        archive_bytes = response.read(MAX_DOWNLOAD_BYTES + 1)

    if len(archive_bytes) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Dataset archive exceeds size limit")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        info = archive.getinfo(DATASET_FILE)

        if info.file_size > MAX_CSV_BYTES:
            raise ValueError("CSV exceeds size limit")

        csv_bytes = archive.read(info)

    frame = pd.read_csv(io.BytesIO(csv_bytes))
    validate_dataset(frame)

    digest = hashlib.sha256(csv_bytes).hexdigest()

    destination.write_bytes(csv_bytes)

    manifest = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "source": DATASET_URL,
        "doi": "10.24432/C5HS5C",
        "license": "CC BY 4.0",
        "sha256": digest,
        "rows": len(frame),
        "columns": len(frame.columns),
    }

    manifest_path = destination.with_suffix(".manifest.json")

    if manifest_path.exists():
        destination.unlink()
        raise FileExistsError(f"{manifest_path} already exists; refusing to overwrite")

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return digest


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a previously downloaded raw dataset."""
    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame
