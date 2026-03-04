"""Comprehensive pytest suite for src.train."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.clean_data import clean_dataframe
from src.train import train_model


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def train_df() -> pd.DataFrame:
    raw_df = pd.read_csv(MOCK_CSV_PATH)

    # Expand small fixture to ensure stable 5-fold CV with enough rows.
    frames = []
    for offset in range(5):
        part = raw_df.copy()
        part["area"] = part["area"] + (offset * 10)
        part["price"] = part["price"] + (offset * 1000)
        frames.append(part)

    expanded = pd.concat(frames, ignore_index=True)
    return clean_dataframe(expanded)


# --- Core training contracts ---


def test_train_model_returns_pipeline_and_cv_results(train_df: pd.DataFrame):
    """train_model returns a fitted Pipeline and full CV result payload."""
    pipeline, cv_results = train_model(train_df, target_column="price")

    assert isinstance(pipeline, Pipeline)
    assert {
        "r2",
        "adjusted_r2",
        "mae",
        "rmse",
        "all_y_true",
        "all_y_pred",
    }.issubset(cv_results.keys())
    assert len(cv_results["all_y_true"]) == len(train_df)
    assert len(cv_results["all_y_pred"]) == len(train_df)

    # Core metric outputs should be finite and non-negative where applicable.
    assert np.isfinite(cv_results["mae"])
    assert np.isfinite(cv_results["rmse"])
    assert np.isfinite(cv_results["r2"])
    assert np.isfinite(cv_results["adjusted_r2"])
    assert cv_results["mae"] >= 0
    assert cv_results["rmse"] >= 0


def test_trained_pipeline_predicts_log_prices(train_df: pd.DataFrame):
    """Returned pipeline can predict on feature-only frame."""
    pipeline, _ = train_model(train_df, target_column="price")
    X = train_df.drop(columns=["price"]).head(3)

    y_pred_log = pipeline.predict(X)
    assert len(y_pred_log) == 3
    assert np.isfinite(y_pred_log).all()


def test_train_model_missing_target_column_raises(train_df: pd.DataFrame):
    """Missing target column fails fast."""
    with pytest.raises(KeyError):
        train_model(train_df, target_column="not_a_real_target")


def test_train_model_model_roundtrip(tmp_path: Path, train_df: pd.DataFrame):
    """Saved and reloaded trained pipeline remains usable for inference."""
    pipeline, _ = train_model(train_df, target_column="price")

    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)

    # Reloaded artifact should preserve inference contract.
    X = train_df.drop(columns=["price"]).head(5)
    preds = reloaded.predict(X)

    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


def test_train_model_raises_on_empty_dataframe() -> None:
    """Empty DataFrame should fail fast before training starts."""
    empty = pd.DataFrame(columns=["price", "area"])

    with pytest.raises(ValueError):
        train_model(empty, target_column="price")


def test_train_model_raises_when_rows_less_than_folds() -> None:
    """5-fold CV requires at least 5 rows."""
    tiny = pd.DataFrame(
        {
            "price": [100000, 120000, 130000, 140000],
            "area": [1000, 1200, 1300, 1400],
            "bedrooms": [2, 2, 3, 3],
            "bathrooms": [1, 1, 2, 2],
            "stories": [1, 1, 2, 2],
            "mainroad": [1, 1, 0, 1],
            "guestroom": [0, 0, 0, 1],
            "basement": [0, 0, 1, 0],
            "hotwaterheating": [0, 0, 0, 0],
            "airconditioning": [1, 0, 1, 0],
            "parking": [1, 1, 2, 2],
            "prefarea": [0, 1, 0, 1],
            "furnishingstatus": [
                "furnished",
                "semi-furnished",
                "unfurnished",
                "furnished",
            ],
        }
    )

    with pytest.raises(ValueError):
        train_model(tiny, target_column="price")
