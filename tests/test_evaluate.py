"""Comprehensive pytest suite for src.evaluate.

- Covers metric pass-through contracts and plot artifact writing
- Uses small in-memory CV result payloads for deterministic, fast tests
"""

from pathlib import Path

import numpy as np
import pytest

from src.evaluate import evaluate_model, save_evaluation_plots


@pytest.fixture
def toy_cv_results() -> dict:
    """Small, valid cv_results payload matching train_model contract."""
    return {
        "r2": 0.50,
        "adjusted_r2": 0.45,
        "mae": 12345.0,
        "rmse": 23456.0,
        "all_y_true": [200000, 250000, 300000, 275000],
        "all_y_pred": [210000, 240000, 290000, 280000],
    }


def test_evaluate_model_returns_metrics_dict(toy_cv_results: dict):
    """evaluate_model returns exactly the scalar metric subset."""
    metrics = evaluate_model(toy_cv_results)

    assert set(metrics.keys()) == {"r2", "adjusted_r2", "mae", "rmse"}
    assert metrics["r2"] == toy_cv_results["r2"]
    assert metrics["adjusted_r2"] == toy_cv_results["adjusted_r2"]
    assert metrics["mae"] == toy_cv_results["mae"]
    assert metrics["rmse"] == toy_cv_results["rmse"]


def test_evaluate_model_missing_metric_key_raises(toy_cv_results: dict):
    """Missing required keys should fail fast with ValueError."""
    # Enforces payload contract strictness so downstream reporting cannot
    # silently proceed with incomplete metrics.
    broken = dict(toy_cv_results)
    broken.pop("rmse")

    with pytest.raises(ValueError):
        evaluate_model(broken)


def test_evaluate_model_rejects_non_dict_cv_results() -> None:
    """cv_results must be a dictionary payload."""
    with pytest.raises(TypeError):
        evaluate_model([1, 2, 3])


def test_save_evaluation_plots_rejects_mismatched_lengths(
    toy_cv_results: dict,
    tmp_path: Path,
):
    """y_true/y_pred length mismatch should fail before plotting."""
    # Misaligned arrays invalidate residual analysis; this must fail early
    # to avoid writing misleading diagnostics.
    broken = dict(toy_cv_results)
    broken["all_y_pred"] = broken["all_y_pred"][:-1]

    with pytest.raises(ValueError):
        save_evaluation_plots(broken, reports_dir=tmp_path)


def test_save_evaluation_plots_writes_pngs(
    toy_cv_results: dict,
    tmp_path: Path,
):
    """save_evaluation_plots writes both expected PNG artifacts."""
    # Confirms the reporting contract that stakeholders rely on: both
    # diagnostic plots must be emitted in successful runs.
    save_evaluation_plots(toy_cv_results, reports_dir=tmp_path)

    assert (tmp_path / "actual_vs_predicted.png").exists()
    assert (tmp_path / "residuals.png").exists()


def test_save_evaluation_plots_creates_reports_dir(
    toy_cv_results: dict,
    tmp_path: Path,
):
    """Non-existing output directory is created automatically."""
    reports_dir = tmp_path / "nested" / "reports"
    assert not reports_dir.exists()

    save_evaluation_plots(toy_cv_results, reports_dir=reports_dir)

    assert reports_dir.exists()
    assert (reports_dir / "actual_vs_predicted.png").exists()
    assert (reports_dir / "residuals.png").exists()


def test_save_evaluation_plots_accepts_numpy_arrays(tmp_path: Path):
    """y_true/y_pred payload can be numpy arrays, not only Python lists."""
    cv_results = {
        "r2": 0.0,
        "adjusted_r2": 0.0,
        "mae": 1.0,
        "rmse": 1.0,
        "all_y_true": np.array([1.0, 2.0, 3.0]),
        "all_y_pred": np.array([1.1, 1.9, 3.2]),
    }
    save_evaluation_plots(cv_results, reports_dir=tmp_path)

    assert (tmp_path / "actual_vs_predicted.png").exists()
    assert (tmp_path / "residuals.png").exists()
