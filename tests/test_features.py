"""Comprehensive pytest suite for src.features."""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
from src.features import get_feature_preprocessor


NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_COLS = ["furnishingstatus"]
BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

# Keep this explicit so tests fail if schema contracts drift and
# preprocessing starts expecting a different feature surface.
REQUIRED_COLUMNS = (
    ["price"] + NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
)

VALID_FURNISHING = {"furnished", "semi-furnished", "unfurnished"}

schema_stub = types.ModuleType("src.schema")
schema_stub.NUMERIC_COLS = NUMERIC_COLS
schema_stub.CATEGORICAL_COLS = CATEGORICAL_COLS
schema_stub.BINARY_COLS = BINARY_COLS
schema_stub.REQUIRED_COLUMNS = REQUIRED_COLUMNS
schema_stub.VALID_FURNISHING = VALID_FURNISHING
sys.modules.setdefault("src.schema", schema_stub)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [13300000, 12250000, 12250000, 12215000],
            "area": [7420, 8960, 9960, 7500],
            "bedrooms": [4, 4, 3, 4],
            "bathrooms": [2, 4, 2, 2],
            "stories": [3, 4, 2, 2],
            "parking": [2, 3, 2, 3],
            "furnishingstatus": [
                "furnished", "semi-furnished", "unfurnished", "furnished"],
            "mainroad": [1, 1, 1, 1],
            "guestroom": [0, 0, 0, 0],
            "basement": [0, 0, 1, 0],
            "hotwaterheating": [0, 0, 0, 0],
            "airconditioning": [1, 1, 0, 1],
            "prefarea": [1, 0, 0, 1],
        }
    )


def test_get_feature_preprocessor_returns_column_transformer():
    """Factory returns an unfitted ColumnTransformer."""
    preprocessor = get_feature_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_can_fit_transform(clean_df: pd.DataFrame):
    """Happy path: preprocessor fits and transforms feature matrix."""
    # Ensures preprocessing contract matches train-time usage where the
    # target is never part of the feature matrix.
    X = clean_df.drop(columns=["price"])
    preprocessor = get_feature_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed).all()


def test_preprocessor_transform_before_fit_raises(clean_df: pd.DataFrame):
    """Transform before fit should raise a not-fitted error."""
    X = clean_df.drop(columns=["price"])
    preprocessor = get_feature_preprocessor()

    # This protects pipeline callers from using stale/uninitialized
    # preprocessing state, which would produce invalid features.
    with pytest.raises(NotFittedError):
        preprocessor.transform(X)


def test_preprocessor_custom_columns_contract(clean_df: pd.DataFrame):
    """Custom column lists are respected and produce expected width."""
    X = clean_df.drop(columns=["price"])

    preprocessor = get_feature_preprocessor(
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        binary_cols=BINARY_COLS,
    )
    transformed = preprocessor.fit_transform(X)

    expected_min_cols = len(NUMERIC_COLS) + len(BINARY_COLS)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= expected_min_cols


def test_preprocessor_fails_on_missing_required_feature(
    clean_df: pd.DataFrame,
):
    """Missing configured input feature should raise during fit."""
    # Verifies fail-fast behavior for schema drift, preventing silent model
    # degradation when upstream data drops an expected column.
    X = clean_df.drop(columns=["price"]).drop(columns=["area"])
    preprocessor = get_feature_preprocessor()

    with pytest.raises(ValueError):
        preprocessor.fit_transform(X)


def test_preprocessor_idempotent_transform(clean_df: pd.DataFrame):
    """After fit, repeated transforms of same data are identical."""
    X = clean_df.drop(columns=["price"])
    preprocessor = get_feature_preprocessor()
    preprocessor.fit(X)

    out1 = preprocessor.transform(X)
    out2 = preprocessor.transform(X)

    np.testing.assert_allclose(out1, out2)


def test_preprocessor_ohe_backward_compatibility_branch(monkeypatch:
                                                        pytest.MonkeyPatch):
    """Covers the except TypeError fallback for older sklearn APIs."""
    call_count = {"n": 0}

    def fake_one_hot_encoder(*args, **kwargs):
        call_count["n"] += 1
        if "sparse_output" in kwargs:
            raise TypeError("unexpected keyword argument 'sparse_output'")
        if "sparse" in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return OneHotEncoder(*args, **kwargs)

    monkeypatch.setattr("src.features.OneHotEncoder", fake_one_hot_encoder)

    preprocessor = get_feature_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    assert call_count["n"] >= 2
