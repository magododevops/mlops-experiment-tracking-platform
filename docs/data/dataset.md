# Dataset Provenance and Contracts

## Source

- Name: AI4I 2020 Predictive Maintenance Dataset
- Provider: UCI Machine Learning Repository
- DOI: https://doi.org/10.24432/C5HS5C
- License: Creative Commons Attribution 4.0
- Data type: Synthetic industrial predictive maintenance observations

## Acquisition

Run:

```bash
uv run python scripts/download_dataset.py
```

The acquisition process validates the input and stores:

- `data/raw/ai4i2020.csv`
- `data/raw/ai4i2020.manifest.json`

Both files are excluded from Git. The manifest records the
source and SHA-256 digest of the downloaded CSV.

## Prediction Target

`Machine failure` (0 = no failure, 1 = failure).

## Feature Contract

Only these fields are eligible as model inputs:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

## Excluded Columns

- UDI: record identifier
- Product ID: product identifier
- TWF, HDF, PWF, OSF, RNF: failure-mode labels
- Machine failure: prediction target

The failure-mode labels must not be used as predictors,
because they encode information directly related to
the prediction target.

## Limitations

This dataset is synthetic. It is useful for demonstrating
reproducible ML workflows but does not establish real-world
predictive maintenance performance.

## Planned Validation

- Train/validation/test splits
- Class imbalance analysis
- Exploratory data analysis
- Baseline evaluation

## Data Quality Rules

The ingestion contract rejects:

- Missing or unexpected columns
- Null and non-finite numeric values
- Duplicate record and product identifiers
- Invalid categorical equipment types
- Nonbinary failure labels
- Nonpositive temperatures or rotational speed
- Negative torque or tool wear values

## Reproducibility and Integrity

The preparation uses a fixed seed of 42 with stratified splits:

- Training: 60%
- Validation: 20%
- Test: 20%

Run:

```bash
make verify-data
```

The verifier checks raw-file integrity and regenerates the
expected partitions to compare content, hashes, statistics,
and manifest metadata.

The SHA-256 manifests detect changes relative to the recorded
data, but they are not cryptographic signatures or proof of
upstream authenticity.
