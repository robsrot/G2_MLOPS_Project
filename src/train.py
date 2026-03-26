"""
Module: Model Training
----------------------
Role: Faithfully implement Model 5 from the HousingPricesPrediction notebook.

Model 5:
  - Uses the FULL dataset (no prior train/test split)
  - KFold(n_splits=5, shuffle=True, random_state=42) is the evaluation strategy
  - Inside every fold: OHE + StandardScaler fit on fold-train only (no leakage)
    - Fits LinearRegression on log1p(price), predicts,
        then applies expm1 to score
  - Reports mean metrics across all folds
    - Finally refits on all rows so the saved model has seen every data point

Input: Full cleaned DataFrame
    (all rows, X and y together via df + target_column)
Output: Fitted sklearn Pipeline ready for serialisation + CV metrics dict

Educational Goal:
- Why this module exists in an MLOps system:
    Model training logic must be reproducible, version-controlled, and
    isolated from experimentation code.
- Responsibility (separation of concerns):
    This module owns model instantiation and training, always using a
    Pipeline so preprocessing and model are serialized together.
- Pipeline contract (inputs and outputs):
    Accepts training data and an unfitted preprocessor, then returns a
    fitted Pipeline ready for deployment.
"""

import logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    target_column: str,
    preprocessor=None,            # ← add this
    n_folds: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    fit_intercept: bool = True,
    numeric_cols: list = None,
    categorical_cols: list = None,
    binary_cols: list = None,
) -> tuple:
    """
    Runs n_folds-fold CV on the full dataset (Model 5),
    then refits on all rows.

    Inputs:
        df: Full cleaned pd.DataFrame from clean_data.py.
                        Must contain all feature columns AND the target column.
                        Binary cols already 0/1,
                        area already log1p-transformed,
                        furnishingstatus still as raw strings for the OHE step.
        target_column: Name of the target column ("price").

    Outputs:
        pipeline: sklearn Pipeline fitted on ALL rows in df.
                        Steps: ("preprocess", ColumnTransformer)
                               ("model",      LinearRegression)
                        pipeline.predict() returns log(price).
                        Apply np.expm1() in infer.py to recover price scale.
        cv_results: dict with keys r2, adjusted_r2, mae, rmse
                        (mean across n_folds folds, on original price scale)
                        These are the numbers comparable to the notebook.
    """
    logger.info(
        "Model 5: K-Fold CV on full dataset + final refit on all rows"
    )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("[train] 'df' must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("[train] Input DataFrame is empty.")

    if not isinstance(target_column, str) or not target_column.strip():
        raise TypeError("[train] 'target_column' must be a non-empty string.")

    if target_column not in df.columns:
        raise KeyError(
            f"[train] Target column '{target_column}' not found in DataFrame."
        )

    if len(df) < n_folds:
        raise ValueError(
            f"[train] At least {n_folds} rows are required for "
            f"{n_folds}-fold cross-validation."
        )

    if df[target_column].isna().any():
        raise ValueError("[train] Target column contains missing values.")

    # Split into features and target once; folds index into these views.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "[train] Feature matrix is empty after dropping target."
        )

    if len(X) != len(y):
        raise ValueError("[train] Feature/target row count mismatch.")

    n_total = len(df)

    # K-Fold CV
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Collect per-fold metrics, then average at the end.
    r2_scores, mae_scores, rmse_scores = [], [], []
    all_y_true, all_y_pred = [], []

    logger.info("Running %d-fold CV on %d rows...", n_folds, n_total)

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        X_fold_train = X.iloc[train_idx].copy()
        X_fold_test = X.iloc[test_idx].copy()
        y_fold_train = y.iloc[train_idx]
        y_fold_test = y.iloc[test_idx]

        # Use sklearn.base.clone() to deep-copy the unfitted preprocessor for
        # each fold — this is the idiomatic sklearn way to reset an estimator
        # to its unfitted state without re-instantiating it from scratch.
        _fold_preprocessor = (
            get_feature_preprocessor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                binary_cols=binary_cols,
            )
            if preprocessor is None
            else clone(preprocessor)
        )
        fold_pipeline = Pipeline([
            ("preprocess", _fold_preprocessor),
            ("model", LinearRegression(fit_intercept=fit_intercept)),
        ])

        # model.fit(X_train, np.log1p(y_train))
        fold_pipeline.fit(X_fold_train, np.log1p(y_fold_train))

        # y_pred = np.expm1(model.predict(X_test))
        y_pred = np.expm1(fold_pipeline.predict(X_fold_test))

        # Score on original price scale for business interpretability.
        fold_r2 = metrics.r2_score(y_fold_test, y_pred)
        fold_mae = metrics.mean_absolute_error(y_fold_test, y_pred)
        fold_rmse = np.sqrt(metrics.mean_squared_error(y_fold_test, y_pred))

        r2_scores.append(fold_r2)
        mae_scores.append(fold_mae)
        rmse_scores.append(fold_rmse)
        all_y_true.extend(y_fold_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        logger.info(
            "Fold %d: R²=%.3f  MAE=%.0f  RMSE=%.0f",
            fold_num, fold_r2, fold_mae, fold_rmse,
        )

    # Mean CV metrics
    mean_r2 = float(np.mean(r2_scores))
    mean_mae = float(np.mean(mae_scores))
    mean_rmse = float(np.mean(rmse_scores))

    # Final refit on all rows:
    # After CV we retrain on 100% of data so the deployed model is as
    # strong as possible. This is the Pipeline that gets saved to disk.
    logger.info("Refitting final Pipeline on ALL rows...")

    # For the final refit we use the passed-in preprocessor directly (no clone
    # needed here — this is the one that gets saved into the Pipeline artifact).
    _final_preprocessor = (
        get_feature_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
        )
        if preprocessor is None
        else preprocessor
    )
    final_pipeline = Pipeline([
        ("preprocess", _final_preprocessor),
        ("model", LinearRegression(fit_intercept=fit_intercept)),
    ])
    final_pipeline.fit(X, np.log1p(y))

    # Adjusted R² uses transformed feature count after preprocessing.
    transformed_feature_count = final_pipeline.named_steps[
        "preprocess"
    ].transform(X.iloc[[0]]).shape[1]
    adj_r2 = float(
        1 - (1 - mean_r2) * (n_total - 1)
        / (n_total - transformed_feature_count - 1)
    )

    # Keep both summary metrics and full prediction arrays for plotting.
    cv_results = {
        "r2": mean_r2,
        "adjusted_r2": adj_r2,
        "mae": mean_mae,
        "rmse": mean_rmse,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
    }

    logger.info(
        "CV mean over all %d folds: "
        "R²=%.3f  Adj-R²=%.3f  MAE=%.0f  RMSE=%.0f",
        n_folds,
        mean_r2, adj_r2, mean_mae, mean_rmse,
    )
    logger.info(
        "Final pipeline fitted on %d rows (%d input columns). "
        "pipeline.predict() returns log(price) - use np.expm1() "
        "to recover original price scale.",
        n_total, X.shape[1],
    )

    return final_pipeline, cv_results
