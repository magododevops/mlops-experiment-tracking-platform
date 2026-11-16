"""Input data contracts for AI4I predictive maintenance."""

import pandas as pd

TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

IDENTIFIER_COLUMNS = (
    "UDI",
    "Product ID",
)

FAILURE_MODE_COLUMNS = (
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)

EXPECTED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
)

NUMERIC_COLUMNS = tuple(column for column in FEATURE_COLUMNS if column != "Type")


class DataValidationError(ValueError):
    """Raised when the dataset violates its input contract."""


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate the raw dataset before any ML transformation."""
    if frame.empty:
        raise DataValidationError("Dataset must not be empty")

    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(EXPECTED_COLUMNS)

    if missing or unexpected:
        raise DataValidationError(
            f"Schema mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    if frame.isna().any().any():
        raise DataValidationError("Dataset contains missing values")

    if frame["UDI"].duplicated().any():
        raise DataValidationError("Duplicate record identifiers detected")

    if frame["Product ID"].duplicated().any():
        raise DataValidationError("Duplicate product identifiers detected")

    if not pd.api.types.is_integer_dtype(frame["UDI"]):
        raise DataValidationError("UDI must contain integer values")

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise DataValidationError(f"{column} must be numeric")

    if not frame.loc[:, NUMERIC_COLUMNS].apply(lambda column: column.map(pd.notna).all()).all():
        raise DataValidationError("Invalid numeric values detected")

    if not frame["Type"].isin(("L", "M", "H")).all():
        raise DataValidationError("Unexpected equipment type")

    target = frame[TARGET_COLUMN]

    if not pd.api.types.is_integer_dtype(target):
        raise DataValidationError("Target must contain integer values")

    if not target.isin((0, 1)).all():
        raise DataValidationError("Target must be binary")

    for column in FAILURE_MODE_COLUMNS:
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise DataValidationError(f"Failure mode {column} must contain integer values")
        if not frame[column].isin((0, 1)).all():
            raise DataValidationError(f"Failure mode {column} must be binary")


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select inference-time features, excluding leakage columns."""
    validate_dataset(frame)
    return frame.loc[:, FEATURE_COLUMNS].copy()
