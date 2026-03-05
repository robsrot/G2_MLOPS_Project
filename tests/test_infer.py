"""Comprehensive pytest suite for src.infer."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.infer import run_inference


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Create test data without external dependencies."""
    raw_df = pd.read_csv(MOCK_CSV_PATH)

    # Expand the tiny fixture so tests exercise inference contracts on more
    # than a trivial number of rows and avoid brittle sample-specific behavior.
    frames = []
    for offset in range(5):
        part = raw_df.copy()
        part["area"] = part["area"] + (offset * 10)
        part["price"] = part["price"] + (offset * 1000)
        frames.append(part)

    expanded = pd.concat(frames, ignore_index=True)

    # Mirror the cleaned feature contract expected by inference pipelines.
    # This keeps failures focused on infer.py behavior, not preprocessing gaps.
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea"]:
        expanded[col] = (expanded[col] == "yes").astype(float)

    # Deterministic numeric encoding keeps this test independent from OHE logic
    # so we can validate inference input/output contracts in isolation.
    furnishing_map = {"furnished": 0.0,
                      "semi-furnished": 1.0, "unfurnished": 2.0}
    expanded["furnishingstatus"] = expanded["furnishingstatus"]\
        .map(furnishing_map)

    # Normalize dtypes to remove ambiguity around sklearn numeric handling.
    for col in ["bedrooms", "bathrooms", "stories", "parking"]:
        expanded[col] = expanded[col].astype(float)

    return expanded


@pytest.fixture
def mock_pipeline() -> Pipeline:
    """Create a fitted pipeline without depending on train.py.

    This simulates what train_model() produces: a sklearn Pipeline
    that expects log1p(price) as target and returns log-scale predictions.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # Use a compact but realistic schema so contract tests stay fast while
    # still representing production-like feature shapes.
    X_dummy = pd.DataFrame({
        "area": [1000.0, 1200.0, 1500.0, 1800.0, 2000.0],
        "bedrooms": [2.0, 3.0, 3.0, 4.0, 4.0],
        "bathrooms": [1.0, 2.0, 2.0, 3.0, 3.0],
        "stories": [1.0, 2.0, 2.0, 2.0, 3.0],
        "mainroad": [1.0, 1.0, 0.0, 1.0, 1.0],  # yes=1, no=0
        "guestroom": [0.0, 0.0, 1.0, 0.0, 1.0],
        "basement": [0.0, 1.0, 0.0, 1.0, 0.0],
        "hotwaterheating": [0.0, 0.0, 0.0, 1.0, 0.0],
        "airconditioning": [1.0, 1.0, 0.0, 1.0, 1.0],
        "parking": [1.0, 2.0, 0.0, 2.0, 2.0],
        "prefarea": [0.0, 1.0, 0.0, 1.0, 0.0],
        "furnishingstatus": [0.0, 1.0, 2.0, 0.0, 1.0]
    })
    # Mimics project behavior where training target is log-transformed and
    # inference converts back to original scale.
    y_dummy = np.log1p([100000, 120000, 150000, 180000, 200000])

    pipeline.fit(X_dummy, y_dummy)
    return pipeline


def test_run_inference_returns_prediction_dataframe(
    train_df: pd.DataFrame, mock_pipeline: Pipeline
):
    """Happy path: inference returns finite, positive predictions."""
    # Enforces the production contract that target values are never available
    # at prediction time.
    X = train_df.drop(columns=["price"]).head(4)

    predictions = run_inference(mock_pipeline, X)

    assert list(predictions.columns) == ["prediction"]
    assert len(predictions) == len(X)
    assert np.isfinite(predictions["prediction"]).all()
    assert (predictions["prediction"] > 0).all()


def test_run_inference_preserves_input_index(
    train_df: pd.DataFrame, mock_pipeline: Pipeline
):
    """Prediction frame should preserve input DataFrame index."""
    X = train_df.drop(columns=["price"]).iloc[[1, 3, 5]]

    predictions = run_inference(mock_pipeline, X)

    # Preserving index is important for row-level traceability to source data.
    assert predictions.index.equals(X.index)


def test_run_inference_fails_on_missing_required_feature(
    train_df: pd.DataFrame, mock_pipeline: Pipeline
):
    """Missing required model feature should fail fast during transform."""
    # Fail-fast here prevents silent schema drift in upstream data pipelines.
    X_missing = train_df.drop(columns=["price", "area"]).head(3)

    with pytest.raises(ValueError):
        run_inference(mock_pipeline, X_missing)


def test_run_inference_raises_when_pipeline_has_no_predict(
    train_df: pd.DataFrame,
):
    """Pipeline contract should require callable predict()."""
    X = train_df.drop(columns=["price"]).head(2)

    with pytest.raises(TypeError):
        run_inference(object(), X)


def test_run_inference_raises_when_input_not_dataframe(
    train_df: pd.DataFrame, mock_pipeline: Pipeline
):
    """Inference input contract requires a pandas DataFrame."""
    X = train_df.drop(columns=["price"]).head(2).to_numpy()

    with pytest.raises(TypeError):
        run_inference(mock_pipeline, X)
