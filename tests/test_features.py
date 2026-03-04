"""Comprehensive pytest suite for src.features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from src.clean_data import clean_dataframe
from src.features import get_feature_preprocessor
from src.schema import BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def clean_df() -> pd.DataFrame:
    raw_df = pd.read_csv(MOCK_CSV_PATH)
    return clean_dataframe(raw_df)


def test_get_feature_preprocessor_returns_column_transformer():
    """Factory returns an unfitted ColumnTransformer."""
    preprocessor = get_feature_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_can_fit_transform(clean_df: pd.DataFrame):
    """Happy path: preprocessor fits and transforms feature matrix."""
    # Model input excludes target.
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

    # sklearn should guard against transform-before-fit usage.
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
    # Remove a required numeric feature from the design matrix.
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
