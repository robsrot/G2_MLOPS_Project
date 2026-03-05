"""
Module: Evaluation
------------------
Role: Evaluate CV results and optionally save evaluation artifacts.

This module now separates pure evaluation from side effects:
    - evaluate_model(): reads cv_results and returns metrics (no file I/O)
    - save_evaluation_plots(): explicit file-writing function

      Plots are saved as PNG files — open the reports/ folder in your
      file explorer to view them. They cannot be shown in a terminal.

Input:  cv_results dict from train_model() (contains metrics + predictions)
Output: Metrics dict, and optional PNG files when explicitly requested

Educational Goal:
- Why this module exists in an MLOps system:
    Evaluation metrics must be consistent across experiments and aligned
    with business objectives.
- Responsibility (separation of concerns):
    This module owns metric computation logic, making it easy to swap
    metrics without touching training code.
- Pipeline contract (inputs and outputs):
    Accepts a fitted model and test data, then returns a metric used for
    monitoring and alerting.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be
imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_CV_KEYS = {
    "r2",
    "adjusted_r2",
    "mae",
    "rmse",
    "all_y_true",
    "all_y_pred",
}


def _validate_cv_results_payload(cv_results: dict) -> None:
    """Validate required contract for CV result payload.

    The evaluation module expects scalar metrics plus aligned prediction
    arrays. This check fails fast before metric reporting or plot writing.
    """
    # Contract check first so downstream code can assume dict operations.
    if not isinstance(cv_results, dict):
        raise TypeError("[evaluate] 'cv_results' must be a dict.")

    missing_keys = REQUIRED_CV_KEYS - set(cv_results.keys())
    if missing_keys:
        raise ValueError(
            f"[evaluate] Missing keys in cv_results: {sorted(missing_keys)}"
        )

    # Ensure reported metrics are scalar numerics (not arrays/objects).
    metric_keys = ["r2", "adjusted_r2", "mae", "rmse"]
    for key in metric_keys:
        value = cv_results[key]
        if not isinstance(value, (int, float)):
            raise TypeError(
                "[evaluate] Metric "
                f"'{key}' must be numeric, got {type(value)}."
            )

    # Prediction arrays must be aligned for valid residual analysis.
    y_true = cv_results["all_y_true"]
    y_pred = cv_results["all_y_pred"]
    if len(y_true) != len(y_pred):
        raise ValueError(
            "[evaluate] all_y_true and all_y_pred must have the same length."
        )
    if len(y_true) == 0:
        raise ValueError("[evaluate] Prediction arrays must not be empty.")


def evaluate_model(
    cv_results: dict,
) -> dict:
    """
    Returns CV metrics and prints them.

    Inputs:
        cv_results: dict returned by train_model() containing:
                    r2, adjusted_r2, mae, rmse - mean CV metrics
                    all_y_true - actual prices
                    all_y_pred - predicted prices

    Outputs:
        metrics: dict with r2, adjusted_r2, mae, rmse
    """
    _validate_cv_results_payload(cv_results)

    r2 = cv_results["r2"]
    adj = cv_results["adjusted_r2"]
    mae = cv_results["mae"]
    rmse = cv_results["rmse"]

    # Print metrics table
    print(
        "[evaluate] Model 5 CV results"
        "(mean over 5 folds, original price scale):\n"
        f"R² = {r2:.3f}\n"
        f"Adjusted R² = {adj:.3f}\n"
        f"MAE = {mae:,.0f}\n"
        f"RMSE = {rmse:,.0f}\n"
    )

    return {"r2": r2, "adjusted_r2": adj, "mae": mae, "rmse": rmse}


def save_evaluation_plots(
    cv_results: dict,
    reports_dir: Path = Path("reports"),
) -> None:
    """
    Explicitly saves evaluation plots to reports/.

    Inputs:
        cv_results: dict from train_model() with all_y_true/all_y_pred
        reports_dir: Path where PNG plots are saved (default: reports/)
    """
    _validate_cv_results_payload(cv_results)

    reports_dir = Path(reports_dir)
    # Explicitly create reports destination for first-run friendliness.
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true = cv_results["all_y_true"]
    y_pred = cv_results["all_y_pred"]

    # Plot 1: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, color="#44489a", alpha=0.6)
    ax.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle="--", color="#636bb8",
    )
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Actual vs Predicted — Model 5 K-Fold CV")
    plt.tight_layout()
    path1 = reports_dir / "actual_vs_predicted.png"
    plt.savefig(path1, dpi=120)
    plt.close()
    print(f"[evaluate] Plot saved → {path1}")

    # Plot 2: Residuals panel
    residuals = pd.Series(y_true) - pd.Series(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, color="#44489a", alpha=0.6)
    axes[0].axhline(0, color="#636bb8", linestyle="--")
    axes[0].set_xlabel("Predicted price")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(
        residuals,
        bins=30,
        edgecolor="black",
        color="#44489a",
        alpha=0.85,
    )
    axes[1].set_title("Distribution of Residuals")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")

    # Simple autocorrelation-style visual: residual(t-1) vs residual(t).
    lagged = residuals.shift(1)
    axes[2].scatter(lagged, residuals, color="#44489a", alpha=0.6)
    axes[2].axhline(0, color="#636bb8", linestyle="--")
    axes[2].axvline(0, color="#636bb8", linestyle="--")
    axes[2].set_title("Residuals vs Lagged Residuals")
    axes[2].set_xlabel("Previous Residual")
    axes[2].set_ylabel("Current Residual")
    axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    path2 = reports_dir / "residuals.png"
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"[evaluate] Plot saved → {path2}")
