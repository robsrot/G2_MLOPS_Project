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

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be
imported from config.yml in a later session
"""

import numpy as np
import pandas as pd

from src.schema import BINARY_COLS


def clean_dataframe(
    df_raw: pd.DataFrame,
    drop_missing_rows: bool = False,
) -> pd.DataFrame:
    """
    Cleans the raw Housing DataFrame.

    Inputs:
        df_raw: Raw pd.DataFrame straight from load_data.py
        drop_missing_rows: If True, drop rows containing NaN values.
                           If False (default), fail fast with ValueError.
    Outputs:
        df_clean: pd.DataFrame ready for schema/domain validation
                  and downstream model training
    """
    print(f"[clean_data] Starting cleaning — shape: {df_raw.shape}")

    # Work on a copy so upstream callers keep their original frame unchanged.
    df = df_raw.copy()

    # 1. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[clean_data] Dropped {dropped} duplicate rows.")

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
        print(
            "[clean_data] WARNING — dropping rows with missing values:\n"
            f"{missing_nonzero}"
        )
        df = df.dropna()
        print(f"[clean_data] Rows after dropping NaNs: {len(df)}")
    else:
        print("[clean_data] No missing values found.")

    # 3. Binary encoding: "yes" -> 1, "no" -> 0
    # Notebook cells 117 (Model 5) — applied once before fold loop
    # Encode only columns that are present to keep function schema-tolerant.
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0})
            # Cast to int so downstream validators can check dtype
            df[col] = df[col].astype(int)

    print(f"[clean_data] Binary-encoded columns: {BINARY_COLS}")

    # 4. Log-transform 'area' feature to reduce right-skewness
    # Notebook cell 118 (Model 5):
    # X_cv_5["area"] = np.log1p(X_cv_5["area"])
    if "area" in df.columns:
        df["area"] = np.log1p(df["area"])
        print("[clean_data] Applied log1p transform to 'area'.")

    print(f"[clean_data] Cleaning complete — shape: {df.shape}")
    return df