"""Comprehensive pytest suite for src.train."""

from pathlib import Path
import importlib
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_feature_preprocessor(
    numeric_cols=None,
    categorical_cols=None,
    binary_cols=None,
) -> ColumnTransformer:
    # Keeps test preprocessing aligned with training feature expectations.
    # Accepts the new keyword parameters but falls back to hardcoded columns
    # for test isolation from config drift.
    numeric_columns = numeric_cols or [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    categorical_columns = categorical_cols or ["furnishingstatus"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


_features_stub = types.ModuleType("src.features")
# Keep training tests independent from feature-module implementation drift.
_features_stub.get_feature_preprocessor = _get_feature_preprocessor


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


def _clean_dataframe_for_test(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    binary_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    binary_map = {"yes": 1, "no": 0, "Yes": 1, "No": 0}
    # Normalize fixture variants so failures reflect training logic,
    # not label casing.
    for column in binary_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(binary_map)

    if "area" in cleaned.columns:
        # Mirrors expected training-space scale for heavy-tailed area values.
        cleaned["area"] = np.log1p(
            pd.to_numeric(cleaned["area"], errors="coerce")
        )

    return cleaned


@pytest.fixture
def train_model_fn(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.features", _features_stub)
    import src.train as train_module

    # Force src.train to bind to the patched dependency each test run.
    train_module = importlib.reload(train_module)
    return train_module.train_model


@pytest.fixture
def train_df() -> pd.DataFrame:
    raw_df = pd.read_csv(MOCK_CSV_PATH)

    # Replication keeps the fixture lightweight while satisfying CV minimums.
    frames = []
    for offset in range(5):
        part = raw_df.copy()
        part["area"] = part["area"] + (offset * 10)
        part["price"] = part["price"] + (offset * 1000)
        frames.append(part)

    expanded = pd.concat(frames, ignore_index=True)
    return _clean_dataframe_for_test(expanded)


# --- Core training contracts ---


def test_train_model_returns_pipeline_and_cv_results(
    train_df: pd.DataFrame,
    train_model_fn,
):
    """train_model returns a fitted Pipeline and full CV result payload."""
    pipeline, cv_results = train_model_fn(
        train_df,
        target_column="price",
        numeric_cols=["area", "bedrooms", "bathrooms", "stories", "parking"],
        categorical_cols=["furnishingstatus"],
        binary_cols=["mainroad", "guestroom", "basement",
                     "hotwaterheating", "airconditioning", "prefarea"],
    )

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

    # NaN/inf metrics usually signal fold leakage or invalid transforms.
    assert np.isfinite(cv_results["mae"])
    assert np.isfinite(cv_results["rmse"])
    assert np.isfinite(cv_results["r2"])
    assert np.isfinite(cv_results["adjusted_r2"])
    assert cv_results["mae"] >= 0
    assert cv_results["rmse"] >= 0


def test_trained_pipeline_predicts_log_prices(
    train_df: pd.DataFrame,
    train_model_fn,
):
    """Returned pipeline can predict on feature-only frame."""
    pipeline, _ = train_model_fn(
        train_df,
        target_column="price",
        numeric_cols=["area", "bedrooms", "bathrooms", "stories", "parking"],
        categorical_cols=["furnishingstatus"],
        binary_cols=["mainroad", "guestroom", "basement",
                     "hotwaterheating", "airconditioning", "prefarea"],
    )
    # Ensures inference path works when target is absent, as in production.
    X = train_df.drop(columns=["price"]).head(3)

    y_pred_log = pipeline.predict(X)
    assert len(y_pred_log) == 3
    assert np.isfinite(y_pred_log).all()


def test_train_model_missing_target_column_raises(
    train_df: pd.DataFrame,
    train_model_fn,
):
    """Missing target column fails fast."""
    # Avoids silent training on the wrong label.
    with pytest.raises(KeyError):
        train_model_fn(train_df, target_column="not_a_real_target")


def test_train_model_model_roundtrip(
    tmp_path: Path,
    train_df: pd.DataFrame,
    train_model_fn,
):
    """Saved and reloaded trained pipeline remains usable for inference."""
    pipeline, _ = train_model_fn(
        train_df,
        target_column="price",
        numeric_cols=["area", "bedrooms", "bathrooms", "stories", "parking"],
        categorical_cols=["furnishingstatus"],
        binary_cols=["mainroad", "guestroom", "basement",
                     "hotwaterheating", "airconditioning", "prefarea"],
    )

    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)

    # Catches serialization issues that only appear outside
    # the training process.
    X = train_df.drop(columns=["price"]).head(5)
    preds = reloaded.predict(X)

    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


def test_train_model_raises_on_empty_dataframe(train_model_fn) -> None:
    """Empty DataFrame should fail fast before training starts."""
    # Early guard prevents opaque downstream estimator errors.
    empty = pd.DataFrame(columns=["price", "area"])

    with pytest.raises(ValueError):
        train_model_fn(empty, target_column="price")


def test_train_model_raises_when_rows_less_than_folds(train_model_fn) -> None:
    """5-fold CV requires at least 5 rows."""
    # Makes CV preconditions explicit at the API boundary.
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
        train_model_fn(tiny, target_column="price")


def test_train_model_raises_when_df_is_not_dataframe(train_model_fn) -> None:
    """Input must be a pandas DataFrame."""
    # Enforces predictable column semantics for downstream transforms.
    with pytest.raises(TypeError):
        train_model_fn([{"price": 100000}], target_column="price")


def test_train_model_raises_when_target_column_is_invalid(
    train_df: pd.DataFrame,
    train_model_fn,
) -> None:
    """Target column name must be a non-empty string."""
    # Keeps column lookup behavior unambiguous.
    with pytest.raises(TypeError):
        train_model_fn(train_df, target_column=None)

    with pytest.raises(TypeError):
        train_model_fn(train_df, target_column="   ")


def test_train_model_raises_when_target_has_missing_values(
    train_df: pd.DataFrame,
    train_model_fn,
) -> None:
    """Missing target values should fail before CV starts."""
    # Prevents fold metrics from being computed on ill-defined labels.
    bad = train_df.copy()
    bad.loc[0, "price"] = np.nan

    with pytest.raises(ValueError):
        train_model_fn(bad, target_column="price")


def test_train_model_raises_when_no_feature_columns(train_model_fn) -> None:
    """Dropping target that leaves no features should raise ValueError."""
    # Training without predictors should fail with a clear contract error.
    only_target = pd.DataFrame({"price": [100000, 120000, 130000, 140000]})

    with pytest.raises(ValueError):
        train_model_fn(only_target, target_column="price")
