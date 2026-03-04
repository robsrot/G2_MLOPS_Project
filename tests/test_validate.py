"""Comprehensive pytest suite for src.validate."""

from pathlib import Path

import pandas as pd
import pytest

from src.clean_data import clean_dataframe
from src.schema import REQUIRED_COLUMNS
from src.validate import validate_dataframe


TEST_DIR = Path(_file_).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def clean_df() -> pd.DataFrame:
    raw_df = pd.read_csv(MOCK_CSV_PATH)
    return clean_dataframe(raw_df)


def test_validate_dataframe_passes_clean_data(clean_df: pd.DataFrame):
    """Happy path: cleaned data passes schema/domain checks."""
    assert validate_dataframe(clean_df, REQUIRED_COLUMNS) is True


def test_validate_dataframe_fails_on_missing_column(clean_df: pd.DataFrame):
    """Missing required column should fail fast."""
    bad = clean_df.drop(columns=["price"])
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_invalid_binary(clean_df: pd.DataFrame):
    """Binary columns must contain only {0,1}."""
    bad = clean_df.copy()
    # Introduce invalid encoded value.
    bad.loc[0, "mainroad"] = 2
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_invalid_category(clean_df: pd.DataFrame):
    """Unknown furnishingstatus category should fail validation."""
    bad = clean_df.copy()
    # Inject out-of-domain category.
    bad.loc[0, "furnishingstatus"] = "unknown"
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_non_positive_price(
    clean_df: pd.DataFrame,
):
    """Target price must be strictly positive."""
    bad = clean_df.copy()
    bad.loc[0, "price"] = 0

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_nan_after_clean(clean_df: pd.DataFrame):
    """Any NaN in required columns should be rejected."""
    bad = clean_df.copy()
    bad.loc[0, "area"] = pd.NA

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_unexpected_column(
    clean_df: pd.DataFrame,
):
    """Unexpected columns should fail strict schema validation."""
    bad = clean_df.copy()
    # Add drifted column not present in canonical schema.
    bad["unexpected_feature"] = 1

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)