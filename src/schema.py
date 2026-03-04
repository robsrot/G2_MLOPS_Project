"""
Shared schema/constants for the housing pipeline.
"""

# Canonical column order expected after cleaning.
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

# Binary yes/no columns after encoding should contain only 0/1.
BINARY_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

# Numeric columns fed into scaler.
NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# Categorical columns fed into OHE.
CATEGORICAL_COLS = ["furnishingstatus"]

# Allowed domain values for furnishingstatus.
VALID_FURNISHING = {"furnished", "semi-furnished", "unfurnished"}
