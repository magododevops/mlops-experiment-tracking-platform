"""Train-fitted feature transformations for predictive maintenance."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_experiment_tracking.data.schema import NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted and reusable feature transformer."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                list(NUMERIC_COLUMNS),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Type"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
