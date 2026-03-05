"""Comprehensive pytest suite for src.validate."""

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Monkeypatch missing module dependency for isolated unit-test execution.
schema_module = types.ModuleType("src.schema")
schema_module.BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
schema_module.VALID_FURNISHING = {
    "furnished",
    "semi-furnished",
    "unfurnished",
}
sys.modules["src.schema"] = schema_module
validate_dataframe = importlib.import_module(
    "src.validate"
).validate_dataframe


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"
REQUIRED_COLUMNS = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]
BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]


@pytest.fixture
def clean_df() -> pd.DataFrame:
    df = pd.read_csv(MOCK_CSV_PATH)
    binary_map = {"yes": 1, "no": 0}
    for col in BINARY_COLS:
        df[col] = df[col].map(binary_map)
    return df


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
