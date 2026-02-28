"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature prep.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NAs, duplicates, unused columns, and fix column names."""
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    df_cleaned = df_cleaned.drop(['ID'], axis=1)
    df_cleaned = df_cleaned.rename(columns={'rx ds': 'rx_ds'})
    return df_cleaned