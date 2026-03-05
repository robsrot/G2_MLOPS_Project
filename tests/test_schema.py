"""Comprehensive pytest suite for src.schema.

- Validates schema constants are defined, unique, and internally coherent
- Keeps checks behavioral (contract-focused), not implementation-copied
"""

from src import schema


def test_schema_constants_are_non_empty():
    """All primary schema lists should be populated."""
    # Empty schemas would silently accept any data structure
    assert len(schema.REQUIRED_COLUMNS) > 0
    assert len(schema.BINARY_COLS) > 0
    assert len(schema.NUMERIC_COLS) > 0
    assert len(schema.CATEGORICAL_COLS) > 0


def test_schema_columns_are_unique():
    """No duplicate entries should exist inside column groups."""
    # Duplicates would double-count features and bias model predictions
    assert len(schema.REQUIRED_COLUMNS) == len(set(schema.REQUIRED_COLUMNS))
    assert len(schema.BINARY_COLS) == len(set(schema.BINARY_COLS))


def test_schema_groups_are_subsets_of_required_columns():
    """Feature groups should all be represented in REQUIRED_COLUMNS."""
    # Feature engineering will fail if referencing columns not in ingested data
    required = set(schema.REQUIRED_COLUMNS)

    assert set(schema.BINARY_COLS).issubset(required)
    assert set(schema.NUMERIC_COLS).issubset(required)
    assert set(schema.CATEGORICAL_COLS).issubset(required)


def test_valid_furnishing_categories_non_empty_strings():
    """Furnishing categories should be defined as meaningful strings."""
    # Empty strings bypass categorical validation and create
    # meaningless features
    assert len(schema.VALID_FURNISHING) > 0
    assert all(
        isinstance(item, str) and item.strip()
        for item in schema.VALID_FURNISHING
    )
