"""
Module: Data Validation
-----------------------
Role: Single authority for schema/domain validation before training.
Input: pandas.DataFrame (cleaned).
Output: Boolean (True if valid) or raises ValueError.

Educational Goal:
- Why this module exists in an MLOps system:
    Fail-fast validation prevents silent errors that corrupt downstream
    model training and predictions.
- Responsibility (separation of concerns):
    This module enforces schema contracts and data-quality invariants.
- Pipeline contract (inputs and outputs):
    Accepts a DataFrame and required columns list, raises exceptions on
    critical failures, and returns True if valid.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    binary_cols: list = None,
    valid_furnishing_values: list = None,
    target_column: str = None,
) -> bool:
    """
    Validates the cleaned DataFrame before training or inference.

    Inputs:
        df: Cleaned pd.DataFrame from clean_data.py
        required_columns: Column names that must be present
        binary_cols: Columns that must only contain 0 or 1
        valid_furnishing_values: Allowed values for furnishingstatus
        target_column: Name of the target column to check for positive values.
                       Defaults to "price" if not provided.
    Outputs:
        True if all checks pass, raises ValueError or TypeError otherwise
    """
    logger.info("Running data validation checks...")

    # 1. None / type guard — reject before any attribute access
    if df is None:
        raise TypeError("[validate] Input DataFrame is None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[validate] Expected pd.DataFrame, got {type(df).__name__}."
        )

    # 2. Empty DataFrame
    if df.empty:
        raise ValueError("[validate] DataFrame is empty.")

    # 3. Required columns
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[validate] Missing required columns: {missing_cols}"
        )

    # Reject unexpected columns to keep training contract stable
    unexpected_cols = [c for c in df.columns if c not in required_columns]
    if unexpected_cols:
        raise ValueError(
            f"[validate] Unexpected columns present: {unexpected_cols}"
        )

    # 4. Missing values guard — whole-DataFrame pattern
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        raise ValueError(
            f"[validate] NaN values found after cleaning:\n"
            f"{null_counts[null_counts > 0]}"
        )

    # 5. Dtype validation — numeric columns must be numeric dtype
    _binary = set(binary_cols) if binary_cols else set()
    _target = target_column or "price"
    _cat = {"furnishingstatus"}
    for col in required_columns:
        if col in df.columns and col not in _binary and col not in _cat:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(
                    f"[validate] Column '{col}' expected numeric dtype, "
                    f"got {df[col].dtype}."
                )

    # 6. Target column must be strictly positive (parametric, not hardcoded)
    if _target in df.columns:
        if (df[_target] <= 0).any():
            raise ValueError(
                f"[validate] Target column '{_target}' contains "
                "non-positive values."
            )

    # 7. Binary columns must be 0 or 1
    _binary_cols = binary_cols if binary_cols is not None else []
    for col in _binary_cols:
        if col in df.columns:
            bad = df[~df[col].isin([0, 1])][col]
            if not bad.empty:
                raise ValueError(
                    f"[validate] Column '{col}' has values outside {{0,1}}: "
                    f"{bad.unique()}"
                )

    # 8. furnishingstatus domain check
    if "furnishingstatus" in df.columns:
        _valid = set(valid_furnishing_values) if valid_furnishing_values else set()
        unknown = set(df["furnishingstatus"].unique()) - _valid
        if unknown:
            raise ValueError(
                f"[validate] Unknown furnishingstatus values: {unknown}. "
                f"Expected: {_valid}"
            )

    logger.info(
        "All checks passed — %d rows, %d columns.", df.shape[0], df.shape[1]
    )
    return True
