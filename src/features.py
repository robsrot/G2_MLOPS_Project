"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

"""
Module: Feature Engineering
----------------------------
Role: Creates new features from cleaned data.
Input: pandas.DataFrame (Clean).
Output: pandas.DataFrame (Features ready for modelling).
"""
import pandas as pd


# =============================================================================
# OPTION A – Notebook-Logik (aktiv)
# =============================================================================

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket rx_ds, create binary sum features, one-hot encode."""
    df_features = df.copy()

    # Percentile-based bucketing for rx_ds
    df_features['rx_ds_bucket'] = pd.qcut(
        df_features['rx_ds'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    # Sum of all binary features
    binary_cols = [col for col in df_features.columns
                   if col not in ['OD', 'rx_ds', 'rx_ds_bucket']]
    df_features['binary_sum'] = df_features[binary_cols].sum(axis=1)

    # Ratio feature
    df_features['rx_ds_to_binary_sum'] = (
        df_features['rx_ds'] / df_features['binary_sum']
    )

    # One-hot encode rx_ds_bucket
    df_one_hot = pd.get_dummies(df_features['rx_ds_bucket'], prefix='rx_ds_bucket')
    df_features = pd.concat([df_features, df_one_hot], axis=1)
    df_features.drop(['rx_ds_bucket'], axis=1, inplace=True)

    return df_features


# =============================================================================
# OPTION B – MLOps-Style ColumnTransformer (noch nicht aktiv, ignorieren)
# =============================================================================

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# import numpy as np
#
#
# def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Step 1: Add all derived columns to the DataFrame before ColumnTransformer.
#     - binary_sum: sum of all binary indicator columns
#     - rx_ds_to_binary_sum: ratio of rx_ds to binary_sum (needs binary_sum first!)
#     - rx_ds_bucket: quartile bucket of rx_ds (needed for OHE in next step)
#     """
#     df = df.copy()
#     binary_cols = [col for col in df.columns if col not in ['OD', 'rx_ds']]
#
#     df['binary_sum'] = df[binary_cols].sum(axis=1)
#     df['rx_ds_to_binary_sum'] = df['rx_ds'] / df['binary_sum']
#     df['rx_ds_bucket'] = pd.qcut(df['rx_ds'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
#
#     return df
#
#
# def build_feature_transformer(binary_cols: list, rx_ds_col: str = 'rx_ds'):
#     """
#     Returns a scikit-learn Pipeline that:
#     1. Adds derived features (binary_sum, ratio, bucket) via FunctionTransformer
#     2. Applies ColumnTransformer:
#        - OHE on rx_ds_bucket
#        - StandardScaler on continuous features (rx_ds, binary_sum, ratio)
#          → important for Logistic Regression (equal feature scales)
#        - Passthrough on binary columns (already 0/1, no scaling needed)
#     """
#     continuous_cols = [rx_ds_col, 'binary_sum', 'rx_ds_to_binary_sum']
#
#     col_transformer = ColumnTransformer(transformers=[
#         ('ohe_bucket',
#             OneHotEncoder(sparse=False, handle_unknown='ignore'),
#             ['rx_ds_bucket']),
#         ('scale_continuous',
#             StandardScaler(),
#             continuous_cols),
#         ('passthrough_binary',
#             'passthrough',
#             binary_cols),
#     ])
#
#     full_pipeline = Pipeline(steps=[
#         ('derive_features', FunctionTransformer(_add_derived_features)),
#         ('transform',       col_transformer),
#     ])
#
#     return full_pipeline


#Um Option B zu aktivieren musst du später nur die # entfernen und in train.py build_feature_transformer() statt engineer() aufrufen.