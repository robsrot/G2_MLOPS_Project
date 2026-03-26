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
) -> bool:
    """
    Validates the cleaned DataFrame before training.

    Inputs:
        df: Cleaned pd.DataFrame from clean_data.py
        required_columns: Column names that must be present
    Outputs:
        True if all checks pass, raises ValueError otherwise
    """
    logger.info("Running data validation checks...")

    # 1. Empty DataFrame
    if df.empty:
        raise ValueError("[validate] DataFrame is empty.")

    # 2. Required columns
    # Strict schema check: every required column must exist.
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[validate] Missing required columns: {missing_cols}"
        )

    # Also reject drifted/extra columns to keep training contract stable.
    unexpected_cols = [c for c in df.columns if c not in required_columns]
    if unexpected_cols:
        raise ValueError(
            f"[validate] Unexpected columns present: {unexpected_cols}"
        )

    # 3. No remaining NaN values
    # Cleaned data should be NaN-free for model reliability.
    null_counts = df[required_columns].isna().sum()
    if null_counts.any():
        raise ValueError(
            f"[validate] NaN values found after cleaning:\n"
            f"{null_counts[null_counts > 0]}"
        )

    # 4. Target column (price) must be positive
    if "price" in df.columns:
        if (df["price"] <= 0).any():
            raise ValueError(
                "[validate] 'price' contains non-positive values."
            )

    # 5. Binary columns must be 0 or 1
    # Binary domain constraints prevent silent encoding mistakes.
    _binary_cols = binary_cols if binary_cols is not None else []
    for col in _binary_cols:
        if col in df.columns:
            bad = df[~df[col].isin([0, 1])][col]
            if not bad.empty:
                raise ValueError(
                    f"[validate] Column '{col}' has values outside {{0,1}}: "
                    f"{bad.unique()}"
                )

    # 6. furnishingstatus must be a known category
    if "furnishingstatus" in df.columns:
        _valid_furnishing = set(valid_furnishing_values) if valid_furnishing_values is not None else set()
        unknown = set(df["furnishingstatus"].unique()) - _valid_furnishing
        if unknown:
            raise ValueError(
                f"[validate] Unknown furnishingstatus values: {unknown}. "
                f"Expected: {_valid_furnishing}"
            )

    logger.info(
        "All checks passed — %d rows, %d columns.", df.shape[0], df.shape[1]
    )
    return True
