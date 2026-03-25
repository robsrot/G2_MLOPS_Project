"""
Module: Data Cleaning
---------------------
Role: Deterministic preprocessing and feature preparation.
Input: pandas.DataFrame (data/raw).
Output: pandas.DataFrame (data/processed).

What this module does:
    1. Drops duplicate rows
    2. Fails fast on missing values (or optionally drops them)
    3. Encodes binary yes/no columns to 0/1 integers
    4. Applies log to the 'area' feature to reduce right-skewness
    5. Returns the cleaned DataFrame with the target column intact

Educational Goal:
- Why this module exists in an MLOps system:
    Data quality issues (missing values, outliers, inconsistencies)
    cause many production ML failures.
- Responsibility (separation of concerns):
    This module owns data-cleaning transformations that must happen
    before feature engineering.
- Pipeline contract (inputs and outputs):
    Accepts raw DataFrame and returns a cleaned DataFrame ready for
    validation and feature extraction.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(
    df_raw: pd.DataFrame,
    binary_cols: list = None,
    log_transform_cols: list = None,
    drop_missing_rows: bool = False,
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Cleans the raw Housing DataFrame.

    Inputs:
        df_raw: Raw pd.DataFrame straight from load_data.py
        drop_missing_rows: If True, drop rows containing NaN values.
                           If False (default), fail fast with ValueError.
        allow_duplicates: If True, skip the deduplication step.
                          Set True for API inference batches where identical
                          records are intentional and each must receive a
                          prediction (default False preserves training behaviour).
    Outputs:
        df_clean: pd.DataFrame ready for schema/domain validation
                  and downstream model training
    """
    logger.info("Starting cleaning — shape: %s", df_raw.shape)

    # Work on a copy so upstream callers keep their original frame unchanged.
    df = df_raw.copy()

    # 1. Drop duplicates (skipped for API inference batches via allow_duplicates)
    if not allow_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d duplicate rows.", dropped)

    # 2. Missing value check
    # Compute per-column NaN counts once for reporting and branching.
    missing = df.isna().sum()
    if missing.any():
        missing_nonzero = missing[missing > 0]
        if not drop_missing_rows:
            raise ValueError(
                "[clean_data] Missing values detected. "
                "Fix upstream data or call clean_dataframe(..., "
                "drop_missing_rows=True).\n"
                f"{missing_nonzero}"
            )
        logger.warning(
            "WARNING — dropping rows with missing values:\n%s", missing_nonzero
        )
        df = df.dropna()
        logger.info("Rows after dropping NaNs: %d", len(df))
    else:
        logger.info("No missing values found.")

    # 3. Binary encoding: "yes" -> 1, "no" -> 0
    # Notebook cells 117 (Model 5) — applied once before fold loop
    # Encode only columns that are present to keep function schema-tolerant.
    _binary_cols = binary_cols if binary_cols is not None else []
    for col in _binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
            # Cast to int so downstream validators can check dtype
            df[col] = df[col].astype(int)

    logger.info("Binary-encoded columns: %s", _binary_cols)

    # 4. Log-transform specified features to reduce right-skewness
    # Notebook cell 118 (Model 5):
    # X_cv_5["area"] = np.log1p(X_cv_5["area"])
    _log_transform_cols = log_transform_cols if log_transform_cols is not None else []
    for col in _log_transform_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            logger.info("Applied log1p transform to '%s'.", col)

    logger.info("Cleaning complete — shape: %s", df.shape)
    return df
