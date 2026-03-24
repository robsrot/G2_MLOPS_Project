"""
Module: Feature Engineering
----------------------------
Role: Build the scikit-learn ColumnTransformer that encodes and scales
      features inside the Pipeline. By wrapping all transforms here, the
      same preprocessing is guaranteed at training AND inference time.

What this module does:
    - StandardScaler on numeric columns
        (area, bedrooms, bathrooms, stories, parking)
    - OneHotEncoder (drop="first") on furnishingstatus

Educational Goal:
- Why this module exists in an MLOps system:
    Feature-engineering bugs can cause data leakage (using test-set
    statistics), inflate validation metrics, and fail in production.
- Responsibility (separation of concerns):
    This module builds unfitted transformation recipes
    (ColumnTransformer) that are fit only on training data in
    the Pipeline.
- Pipeline contract (inputs and outputs):
    Returns a ColumnTransformer recipe configured with
    student-specified transformations, preventing leakage by deferring
    fit to train.py.
"""

import logging
from typing import Optional, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def get_feature_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Builds and returns an *unfitted* ColumnTransformer.
    The Pipeline in train.py calls .fit_transform() on this object,
    which means it is always fit on training data only — no leakage.

    Inputs:
        numeric_cols: Columns to StandardScale.
                           Defaults to NUMERIC_COLS defined above.
        categorical_cols: Columns to OneHotEncode (drop="first").
                           Defaults to CATEGORICAL_COLS defined above.
        binary_cols: Already 0/1 integer columns — passed through
                           unchanged (remainder="passthrough" handles them,
                           and listing them explicitly keeps
                           the contract clear).
                           Defaults to BINARY_COLS defined above.
    Outputs:
        preprocessor : sklearn ColumnTransformer (unfitted)
    """
    logger.info("Building feature recipe from configuration")

    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []
    binary_cols = binary_cols or []

    # OneHotEncoder
    try:
        # sklearn >= 1.2
        ohe = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
    except TypeError:
        # Backward compatibility for older sklearn versions.
        ohe = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="ignore"
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",    StandardScaler(), numeric_cols),
            ("cat",    ohe,             categorical_cols),
            # Binary columns (already 0/1) are passed through as-is
            ("binary", "passthrough",   binary_cols),
        ],
        # Drop any unlisted columns to keep feature contract explicit.
        remainder="drop",
    )

    logger.info(
        "ColumnTransformer built. numeric: %s | categorical: %s | binary: %s",
        numeric_cols,
        categorical_cols,
        binary_cols,
    )
    return preprocessor
