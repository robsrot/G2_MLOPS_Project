"""Basic file I/O helpers for CSV data and model artifacts.

This module intentionally stays minimal:
- straightforward read/write helpers used across pipeline stages
Educational Goal:
- Why this module exists in an MLOps system:
    Centralize I/O operations to reduce file handling errors and ensure
    consistent serialization across pipeline stages
- Responsibility (separation of concerns):
    This module owns all interactions with the
    filesystem (CSV and model persistence)
- Pipeline contract (inputs and outputs):
    Provides reusable save/load functions that guarantee
    reproducibility and compatibility
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(
    filepath: Path,
    read_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a CSV file from disk into a DataFrame."""
    filepath = Path(filepath)

    # Keep safe defaults while allowing caller overrides.
    default_read_options = {"encoding": "utf-8"}

    if read_options is None:
        read_options = default_read_options
    else:
        read_options = {**default_read_options, **read_options}

    if not filepath.exists():
        raise FileNotFoundError(f"[utils] CSV file not found: {filepath}")

    if filepath.suffix.lower() != ".csv":
        raise ValueError(f"[utils] Expected a .csv file, got: {filepath}")

    return pd.read_csv(filepath, **read_options)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to CSV."""
    filepath = Path(filepath)
    # Create the output directory if it does not exist.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    write_options = {"index": False, "encoding": "utf-8"}

    df.to_csv(filepath, **write_options)
    logger.info("CSV saved to %s", filepath)


def save_model(model: Any, filepath: Path) -> None:
    """Serialize a model artifact to disk with joblib."""
    filepath = Path(filepath)
    # Mirror CSV behavior for model artifact folders.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)


def load_model(filepath: Path) -> Any:
    """Load a serialized model artifact from disk."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"[utils] Model file not found: {filepath}")

    logger.info("Loading model from %s", filepath)
    return joblib.load(filepath)
