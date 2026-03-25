"""Comprehensive pytest suite for src.validate."""

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

# Decouples validate tests from schema module import-time availability.
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
    # Keep fixture encoding aligned with validator expectations.
    for col in BINARY_COLS:
        df[col] = df[col].map(binary_map)
    return df


def test_validate_dataframe_passes_clean_data(clean_df: pd.DataFrame):
    """Happy path: cleaned data passes schema/domain checks."""
    # Baseline proves strict checks still accept valid production-shaped data.
    assert validate_dataframe(
        clean_df,
        REQUIRED_COLUMNS,
        binary_cols=BINARY_COLS,
        valid_furnishing_values=["furnished", "semi-furnished", "unfurnished"],
    ) is True


def test_validate_dataframe_fails_on_missing_column(clean_df: pd.DataFrame):
    """Missing required column should fail fast."""
    # Missing core fields should stop the pipeline before training/inference.
    bad = clean_df.drop(columns=["price"])
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_invalid_binary(clean_df: pd.DataFrame):
    """Binary columns must contain only {0,1}."""
    bad = clean_df.copy()
    # Single out-of-domain value is enough to reject the row set.
    bad.loc[0, "mainroad"] = 2
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_invalid_category(clean_df: pd.DataFrame):
    """Unknown furnishingstatus category should fail validation."""
    bad = clean_df.copy()
    # Guards against category drift at inference time.
    bad.loc[0, "furnishingstatus"] = "unknown"
    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_non_positive_price(
    clean_df: pd.DataFrame,
):
    """Target price must be strictly positive."""
    # Rejects implausible targets that can destabilize model behavior.
    bad = clean_df.copy()
    bad.loc[0, "price"] = 0

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_nan_after_clean(clean_df: pd.DataFrame):
    """Any NaN in required columns should be rejected."""
    # Ensures cleaning guarantees are enforced at validation boundary.
    bad = clean_df.copy()
    bad.loc[0, "area"] = pd.NA

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)


def test_validate_dataframe_fails_on_unexpected_column(
    clean_df: pd.DataFrame,
):
    """Unexpected columns should fail strict schema validation."""
    bad = clean_df.copy()
    # Flags upstream schema drift instead of silently tolerating it.
    bad["unexpected_feature"] = 1

    with pytest.raises(ValueError):
        validate_dataframe(bad, REQUIRED_COLUMNS)
