"""
Module: Data Loader
-------------------
Role: Fetch raw data explicitly, then load it with fail-fast
    file/read integrity checks.
Input: Path to CSV.
Output: pandas.DataFrame.

Educational Goal:
- Why this module exists in an MLOps system:
    Isolates data ingestion logic from transformation logic, making
    pipelines testable and source-agnostic.
- Responsibility (separation of concerns):
    This module owns the contract with upstream data providers
    (files, databases, APIs).
- Pipeline contract (inputs and outputs):
    Returns a raw DataFrame that subsequent modules can clean and
    transform predictably.
"""

# Standard library
import logging
import shutil
from pathlib import Path

# Third-party
import pandas as pd

# Local
from src.utils import load_csv, save_csv

logger = logging.getLogger(__name__)


def _create_dummy_housing_data(raw_data_path: Path) -> pd.DataFrame:
    """Create and persist deterministic fallback data for scaffolding."""
    # Intentional loud warning so dummy usage is visible in logs.
    logger.warning(
        "LOUD WARNING: CREATING DUMMY DATASET FOR SCAFFOLDING ONLY."
        " UPDATE SETTINGS."
    )

    # Deterministic tiny dataset that matches the expected Housing schema.
    dummy_data = pd.DataFrame({
        "area": [2000, 3000, 4000],
        "bedrooms": [2, 3, 4],
        "bathrooms": [1, 2, 2],
        "stories": [1, 2, 2],
        "mainroad": ["yes", "yes", "no"],
        "guestroom": ["no", "no", "yes"],
        "basement": ["no", "yes", "no"],
        "hotwaterheating": ["no", "no", "no"],
        "airconditioning": ["yes", "yes", "no"],
        "parking": [1, 2, 2],
        "prefarea": ["yes", "no", "no"],
        "furnishingstatus": ["furnished", "semi-furnished", "unfurnished"],
        "price": [500000.0, 750000.0, 900000.0],
    })

    # Persist fallback to keep downstream pipeline steps file-based
    # and reproducible.
    raw_data_path = Path(raw_data_path)
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(dummy_data, raw_data_path)
    logger.info("Created dummy CSV at %s", raw_data_path)
    return dummy_data


def fetch_raw_data_from_kaggle(
    destination: Path,
    overwrite: bool = False,
    kaggle_dataset: str = "yasserh/housing-prices-dataset",
    kaggle_filename: str = "Housing.csv",
) -> Path:
    """
    Explicitly downloads Housing.csv from Kaggle into destination.

    Requires either:
      - A ~/.kaggle/kaggle.json API token, OR
      - Environment variables KAGGLE_USERNAME and KAGGLE_KEY set.

    Obtain token at: https://www.kaggle.com/settings
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "[load_data] kagglehub is not installed.\n"
            "Run:  pip install kagglehub\n"
            "Then add your Kaggle API token to ~/.kaggle/kaggle.json\n"
            "  (download it from https://www.kaggle.com/settings → API)"
        )

    destination = Path(destination)

    # Reuse local file unless overwrite is explicitly requested.
    if destination.exists() and not overwrite:
        return destination

    logger.info("Downloading '%s' from Kaggle...", kaggle_dataset)

    # kagglehub.dataset_download returns the path to the local cache folder
    cache_dir = Path(kagglehub.dataset_download(kaggle_dataset))

    # Find the CSV inside the downloaded folder
    csv_candidates = sorted(cache_dir.rglob(kaggle_filename))
    if len(csv_candidates) != 1:
        available_files = sorted(
            str(path.relative_to(cache_dir))
            for path in cache_dir.rglob("*")
        )
        raise FileNotFoundError(
            "[load_data] Expected exactly one "
            f"'{kaggle_filename}' in dataset cache, found "
            f"{len(csv_candidates)} at {cache_dir}. "
            f"Files present: {available_files}"
        )

    # Copy the discovered dataset CSV into our project path.
    source_csv = csv_candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, destination)
    logger.info("Saved to %s", destination)
    return destination


def ensure_raw_data_exists(
    raw_data_path: Path,
    fetch_if_missing: bool = False,
    kaggle_dataset: str = "yasserh/housing-prices-dataset",
    kaggle_filename: str = "Housing.csv",
) -> Path:
    """
    Ensures the raw CSV file exists on disk.

    If fetch_if_missing is True, this function may fetch from Kaggle.
    """
    raw_data_path = Path(raw_data_path)

    # Fast path: file already exists.
    if raw_data_path.exists():
        return raw_data_path

    if not fetch_if_missing:
        raise FileNotFoundError(
            f"[load_data] Raw data file not found: {raw_data_path}. "
            "Call ensure_raw_data_exists(..., "
            "fetch_if_missing=True) to fetch explicitly."
        )

    # Explicitly fetch only when caller opted in.
    return fetch_raw_data_from_kaggle(
        raw_data_path,
        kaggle_dataset=kaggle_dataset,
        kaggle_filename=kaggle_filename,
    )


def load_raw_data(
    raw_data_path: Path,
    use_dummy_on_failure: bool = True,
    kaggle_dataset: str = "yasserh/housing-prices-dataset",
    kaggle_filename: str = "Housing.csv",
) -> pd.DataFrame:
    """
    Loads raw CSV from disk with fail-fast file/read validation.

    Note:
        Schema/domain checks are intentionally handled in validate.py.

    Inputs:
        raw_data_path: Path to the CSV file (e.g. data/raw/Housing.csv)
        use_dummy_on_failure: If True, writes and returns deterministic
            fallback data whenever local load/fetch fails.
    Outputs:
        df: pd.DataFrame with all original columns untouched
    """
    raw_data_path = Path(raw_data_path)

    if raw_data_path.is_dir():
        raise IsADirectoryError(
            "[load_data] Expected a CSV file path, "
            f"got directory: {raw_data_path}"
        )

    logger.info("Loading from: %s", raw_data_path)
    try:
        # Primary path: load local CSV as-is.
        df = load_csv(raw_data_path)
    except FileNotFoundError as exc:
        # Missing local file: optionally try remote fetch before fallback.
        logger.warning("Missing file: %s", exc)
        if not use_dummy_on_failure:
            raise

        try:
            logger.info("Attempting Kaggle fetch before dummy fallback...")
            ensure_raw_data_exists(
                raw_data_path,
                fetch_if_missing=True,
                kaggle_dataset=kaggle_dataset,
                kaggle_filename=kaggle_filename,
            )
            df = load_csv(raw_data_path)
        except Exception as fetch_exc:
            # Last-resort behavior: synthesize a valid dummy dataset.
            logger.warning("Fetch/load failed: %s", fetch_exc)
            return _create_dummy_housing_data(raw_data_path)
    except pd.errors.EmptyDataError as exc:
        # Empty file exists; optionally substitute deterministic fallback.
        if not use_dummy_on_failure:
            raise ValueError(
                f"[load_data] CSV is empty: {raw_data_path}"
            ) from exc
        logger.warning("CSV is empty: %s", raw_data_path)
        return _create_dummy_housing_data(raw_data_path)
    except pd.errors.ParserError as exc:
        # Corrupted CSV syntax; optionally substitute deterministic fallback.
        if not use_dummy_on_failure:
            raise ValueError(
                f"[load_data] CSV parse error in file: {raw_data_path}"
            ) from exc
        logger.warning("CSV parse error in file: %s", raw_data_path)
        return _create_dummy_housing_data(raw_data_path)
    except ValueError as exc:
        # Utility-level format validation failures (e.g., wrong extension).
        if not use_dummy_on_failure:
            raise
        logger.warning("Invalid CSV format: %s", exc)
        return _create_dummy_housing_data(raw_data_path)

    # Header-only CSVs can parse but still be unusable for modeling.
    if df.empty:
        if not use_dummy_on_failure:
            raise ValueError(
                f"[load_data] CSV has no data rows: {raw_data_path}"
            )
        logger.warning("CSV has no data rows: %s", raw_data_path)
        return _create_dummy_housing_data(raw_data_path)

    logger.info("Loaded %d rows x %d columns.", df.shape[0], df.shape[1])
    logger.info("Columns: %s", df.columns.tolist())
    return df
