"""Integration tests for src/main.py pipeline orchestration.

Patching strategy: monkeypatch src.main._load_config to return a complete
in-memory config dict with all paths redirected to pytest's tmp_path.

This replaces the old approach of patching module-level constants
(RAW_DATA_PATH, PROCESSED_DIR, etc.) that no longer exist after the
Step 5 config-driven rewrite — all paths now come from config.yaml at
runtime inside main().
"""

import os

os.environ.setdefault("WANDB_MODE", "disabled")

from pathlib import Path

import pandas as pd
import pytest
import src.main as main_module

TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def large_mock_csv(tmp_path: Path) -> Path:
    """Create a large enough mock CSV for full pipeline testing.

    Requires at least 5 rows for 5-fold CV. We replicate the 5-row template
    15 times (75 rows) with varied area/price to avoid duplicate-dropping.
    """
    csv_path = tmp_path / "mock_housing_large.csv"
    template_df = pd.read_csv(MOCK_CSV_PATH)
    frames = []
    for i in range(15):
        df_copy = template_df.copy()
        df_copy["area"] = df_copy["area"] + (i * 50)
        df_copy["price"] = df_copy["price"] + (i * 5000)
        frames.append(df_copy)
    large_df = pd.concat(frames, ignore_index=True)
    large_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def inference_csv(tmp_path: Path) -> Path:
    """Create a 20-row inference CSV (no price column) from the mock template.

    Area is varied across copies so all 20 rows are unique after dedup.
    """
    template_df = pd.read_csv(MOCK_CSV_PATH)
    infer_df = template_df.drop(columns=["price"])
    frames = []
    for i in range(4):
        copy = infer_df.copy()
        copy["area"] = copy["area"] + (i * 100)
        frames.append(copy)
    result = pd.concat(frames, ignore_index=True)
    csv_path = tmp_path / "housing_inference.csv"
    result.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def _make_test_config(tmp_path: Path, raw_csv: Path, infer_csv: Path) -> dict:
    """Return a complete config dict with all paths redirected into tmp_path.

    Using absolute path strings is intentional: _resolve_path() in main.py
    does (project_root / relative).resolve(), and on pathlib joining an
    absolute string replaces the base, so the absolute tmp_path paths win.

    W&B is disabled via log_to_wandb=False so no network calls are made.
    """
    return {
        "paths": {
            "raw_data":             str(raw_csv),
            "processed_data":       str(tmp_path / "processed" / "clean.csv"),
            "model_artifact":       str(tmp_path / "models" / "model.joblib"),
            "inference_data":       str(infer_csv),
            "predictions_artifact": str(tmp_path / "reports" / "predictions.csv"),
            "log_file":             str(tmp_path / "logs" / "pipeline.log"),
        },
        "logging": {"level": "INFO", "format": "text"},
        "problem": {"type": "regression", "target_column": "price"},
        "split": {"n_folds": 5, "random_state": 42},
        "training": {
            "regression": {
                "model_type": "linear_regression",
                "random_state": 42,
                "fit_intercept": True,
            }
        },
        "data": {
            "fetch_if_missing": True,
            "use_dummy_on_failure": True,
            "kaggle_dataset": "yasserh/housing-prices-dataset",
            "kaggle_filename": "Housing.csv",
            "drop_missing_rows": False,
            "allow_duplicates": False,
        },
        "features": {
            "log_transform_cols": ["area"],
            "binary_cols": [
                "mainroad", "guestroom", "basement",
                "hotwaterheating", "airconditioning", "prefarea",
            ],
            "categorical_onehot": ["furnishingstatus"],
            "numeric_passthrough": ["bedrooms", "bathrooms", "stories", "parking"],
            "n_bins": 5,
            "valid_furnishing_values": ["furnished", "semi-furnished", "unfurnished"],
        },
        "validation": {
            "numeric_non_negative_cols": [
                "price", "area", "bedrooms", "bathrooms", "stories", "parking"
            ]
        },
        "evaluation": {
            "primary_metric": "rmse",
            "secondary_metrics": ["mae", "r2", "adj_r2"],
            "save_plots": True,
        },
        "run": {"log_to_wandb": False, "save_predictions": True},
        "wandb": {
            "project": "test",
            "job_type": "test",
            "group": "test",
            "tags": [],
            "notes": "test",
            "model_artifact_name":  "test-model",
            "model_registry_name":  "test-predictor",
            "model_alias":          "prod",
            "log_dataset":    False,
            "log_model":      False,
            "log_predictions": False,
            "log_plots":      False,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_main_runs_with_mock_data(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
    inference_csv: Path,
):
    """Happy path: main() writes all expected artifacts to configured dirs."""
    monkeypatch.setattr(
        main_module,
        "_load_config",
        lambda _path: _make_test_config(tmp_path, large_mock_csv, inference_csv),
    )

    main_module.main()

    # Processed CSV
    processed_csv = tmp_path / "processed" / "clean.csv"
    assert processed_csv.exists()
    df_processed = pd.read_csv(processed_csv)
    assert len(df_processed) > 0
    assert "price" in df_processed.columns

    # Model artifact — one file at the exact configured path
    assert (tmp_path / "models" / "model.joblib").exists()

    # Evaluation plots
    assert (tmp_path / "reports" / "actual_vs_predicted.png").exists()
    assert (tmp_path / "reports" / "residuals.png").exists()

    # Predictions artifact — new location comes from paths.predictions_artifact
    predictions_csv = tmp_path / "reports" / "predictions.csv"
    assert predictions_csv.exists()
    df_predictions = pd.read_csv(predictions_csv)
    assert "prediction" in df_predictions.columns
    assert len(df_predictions) > 0
    assert (df_predictions["prediction"] > 0).all()


def test_main_creates_all_output_directories(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
    inference_csv: Path,
):
    """Verify main() creates all required output directories."""
    monkeypatch.setattr(
        main_module,
        "_load_config",
        lambda _path: _make_test_config(tmp_path, large_mock_csv, inference_csv),
    )

    main_module.main()

    assert (tmp_path / "processed").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_main_raises_on_missing_raw_data_when_fetch_disabled(
    monkeypatch,
    tmp_path: Path,
    inference_csv: Path,
):
    """If load_raw_data raises, main() propagates the error without swallowing it.

    In the new pipeline load_raw_data() is the ingestion entry point;
    ensure_raw_data_exists() is no longer called from main(). We patch
    load_raw_data directly to simulate a complete ingestion failure.
    """
    monkeypatch.setattr(
        main_module,
        "_load_config",
        lambda _path: _make_test_config(
            tmp_path, tmp_path / "missing.csv", inference_csv
        ),
    )

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("mock missing")

    monkeypatch.setattr(main_module, "load_raw_data", _raise_not_found)

    with pytest.raises(FileNotFoundError, match="mock missing"):
        main_module.main()


def test_main_preserves_data_split_contract(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
    inference_csv: Path,
):
    """Training and inference use entirely separate datasets.

    The prediction count must equal the number of unique rows in the
    inference CSV — confirming that (a) inference runs on the dedicated
    file, not on a sample from training data, and (b) every unique
    inference row receives a positive predicted price.
    """
    monkeypatch.setattr(
        main_module,
        "_load_config",
        lambda _path: _make_test_config(tmp_path, large_mock_csv, inference_csv),
    )

    main_module.main()

    predictions_csv = tmp_path / "reports" / "predictions.csv"
    assert predictions_csv.exists()

    df_predictions = pd.read_csv(predictions_csv)

    # Expected count: unique rows after clean_dataframe's deduplication step.
    # Reading the inference CSV and deduplicating mirrors what main() does.
    expected_count = len(pd.read_csv(inference_csv).drop_duplicates())
    assert len(df_predictions) == expected_count

    assert (df_predictions["prediction"] > 0).all()
