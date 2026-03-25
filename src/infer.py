"""Inference utilities.

This module executes predictions with a fitted pipeline and returns a
standardized DataFrame contract for downstream consumers.

Educational Goal:
- Why this module exists in an MLOps system:
    Production inference must use the exact same preprocessing as
    training to prevent train/serve skew.
- Responsibility (separation of concerns):
    This module owns the contract for applying trained models to
    new data.
- Pipeline contract (inputs and outputs):
    Accepts a fitted Pipeline and new data, then returns predictions in
    a standardized format for downstream systems.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_inference(pipeline: Any, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Generates price predictions for new data.

    Inputs:
        pipeline: Fitted sklearn Pipeline loaded from models/model.joblib.
                   Must be the Pipeline returned by train.py — it expects
                   the same column format as X_train (cleaned, binary-encoded,
                   area log1p-transformed, furnishingstatus still raw strings).
        X_infer: pd.DataFrame — new data in the same format as X_train.
                   Must contain all feature columns (no target column).

    Outputs:
        predictions: pd.DataFrame with one column "prediction" containing
                      predicted house prices on the original scale.
                      Index is preserved from X_infer.
    """
    if not hasattr(pipeline, "predict") or not callable(pipeline.predict):
        raise TypeError(
            "[infer] 'pipeline' must implement "
            "a callable predict() method."
        )

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError("[infer] 'X_infer' must be a pandas DataFrame.")

    if X_infer.empty:
        raise ValueError("[infer] 'X_infer' must not be empty.")

    logger.info("Running inference on %d rows...", len(X_infer))

    # Model is trained on log1p(price), so predictions are on log scale.
    y_pred_log = pipeline.predict(X_infer)
    # Convert back to original price units for user-facing outputs.
    y_pred = np.expm1(y_pred_log)

    # Preserve input index to keep traceability to source rows.
    predictions = pd.DataFrame(
        {"prediction": y_pred},
        index=X_infer.index,
    )

    logger.info(
        "Done. Predicted price range: %.0f - %.0f",
        predictions["prediction"].min(),
        predictions["prediction"].max(),
    )

    return predictions
