"""Comprehensive pytest suite for src.clean_data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe
from src.schema import BINARY_COLS


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.read_csv(MOCK_CSV_PATH)


def test_clean_dataframe_transforms_area_and_binaries(raw_df: pd.DataFrame):
    """Happy path: area is log-transformed and binary cols map to {0,1}."""
    df_clean = clean_dataframe(raw_df)

    assert "area" in df_clean.columns
    assert np.allclose(
        df_clean["area"].values,
        np.log1p(raw_df["area"].values),
    )

    for col in BINARY_COLS:
        assert set(df_clean[col].unique()).issubset({0, 1})


def test_clean_dataframe_raises_on_missing_by_default(raw_df: pd.DataFrame):
    """Fail-fast behavior: missing values raise unless explicitly allowed."""
    df_bad = raw_df.copy()
    # Inject one NaN to trigger missing-data branch.
    df_bad.loc[0, "area"] = np.nan

    with pytest.raises(ValueError):
        clean_dataframe(df_bad)


def test_clean_dataframe_can_drop_missing(raw_df: pd.DataFrame):
    """Optional behavior: drop_missing_rows=True removes NaN rows."""
    df_bad = raw_df.copy()
    # Same setup as previous test, but with permissive option enabled.
    df_bad.loc[0, "area"] = np.nan

    df_clean = clean_dataframe(df_bad, drop_missing_rows=True)
    assert len(df_clean) == len(df_bad) - 1
    assert not df_clean.isna().any().any()


def test_clean_dataframe_drops_exact_duplicates(raw_df: pd.DataFrame):
    """Duplicate rows are removed during cleaning."""
    # Append one known duplicate row.
    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)

    df_clean = clean_dataframe(duplicated)
    df_reference = clean_dataframe(raw_df)

    assert len(df_clean) == len(df_reference)