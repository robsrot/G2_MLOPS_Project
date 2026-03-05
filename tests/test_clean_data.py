"""Comprehensive pytest suite for src.clean_data."""

import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture(scope="module")
def clean_dataframe():
    # Isolate this test module from schema-side changes so failures here
    # reflect clean_data behavior, not unrelated schema refactors.
    fake_schema = types.ModuleType("src.schema")
    fake_schema.BINARY_COLS = BINARY_COLS
    sys.modules["src.schema"] = fake_schema

    import src.clean_data as clean_data_module

    importlib.reload(clean_data_module)
    return clean_data_module.clean_dataframe


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.read_csv(MOCK_CSV_PATH)


def test_clean_dataframe_transforms_area_and_binaries(
    clean_dataframe,
    raw_df: pd.DataFrame,
):
    """Happy path: area is log-transformed and binary cols map to {0,1}."""
    df_clean = clean_dataframe(raw_df)

    # Guard against silent preprocessing drift that would break train/serve
    # consistency with the notebook/model assumptions.
    assert "area" in df_clean.columns
    assert np.allclose(
        df_clean["area"].values,
        np.log1p(raw_df["area"].values),
    )

    # Binary domain must remain strict; any value outside {0,1} can corrupt
    # downstream validation and feature preprocessing contracts.
    for col in BINARY_COLS:
        assert set(df_clean[col].unique()).issubset({0, 1})


def test_clean_dataframe_raises_on_missing_by_default(
    clean_dataframe,
    raw_df: pd.DataFrame,
):
    """Fail-fast behavior: missing values raise unless explicitly allowed."""
    df_bad = raw_df.copy()
    # One missing value is enough to verify that the default policy protects
    # model training from silently accepting incomplete data.
    df_bad.loc[0, "area"] = np.nan

    with pytest.raises(ValueError):
        clean_dataframe(df_bad)


def test_clean_dataframe_can_drop_missing(
    clean_dataframe,
    raw_df: pd.DataFrame,
):
    """Optional behavior: drop_missing_rows=True removes NaN rows."""
    df_bad = raw_df.copy()
    # Mirrors the strict-path setup to prove behavior flips only because of
    # explicit caller intent, not accidental side effects.
    df_bad.loc[0, "area"] = np.nan

    df_clean = clean_dataframe(df_bad, drop_missing_rows=True)
    assert len(df_clean) == len(df_bad) - 1
    assert not df_clean.isna().any().any()


def test_clean_dataframe_drops_exact_duplicates(
    clean_dataframe,
    raw_df: pd.DataFrame,
):
    """Duplicate rows are removed during cleaning."""
    # Duplicate removal is a data-quality invariant; this protects against
    # regressions that would skew metrics by overweighting repeated records.
    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)

    df_clean = clean_dataframe(duplicated)
    df_reference = clean_dataframe(raw_df)

    assert len(df_clean) == len(df_reference)
