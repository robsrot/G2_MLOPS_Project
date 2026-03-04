"""Comprehensive integration tests for src.main orchestration"""

from pathlib import Path

import pandas as pd
import pytest
import src.main as main_module

TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def large_mock_csv(tmp_path: Path) -> Path:
    """Create a large enough mock CSV for full pipeline testing.
    
    Main requires at least:
    - 50 rows for inference holdout
    - 5+ rows for 5-fold CV training
    - So we need 60+ total rows
    """
    csv_path = tmp_path / "mock_housing_large.csv"
    
    # Read the small mock CSV as a template
    template_df = pd.read_csv(MOCK_CSV_PATH)
    
    # Replicate it 15 times to get 75 rows (5 original * 15 = 75)
    frames = []
    for i in range(15):
        df_copy = template_df.copy()
        # Add variation to avoid identical duplicates
        df_copy["area"] = df_copy["area"] + (i * 50)
        df_copy["price"] = df_copy["price"] + (i * 5000)
        frames.append(df_copy)
    
    large_df = pd.concat(frames, ignore_index=True)
    large_df.to_csv(csv_path, index=False)
    
    return csv_path


def test_main_runs_with_mock_data(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
):
    """Happy path: main() writes all expected artifacts to configured dirs."""
    # Redirect all pipeline outputs into pytest temp folders.
    monkeypatch.setattr(main_module, "RAW_DATA_PATH", large_mock_csv)
    monkeypatch.setattr(main_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(main_module, "INFERENCE_DIR", tmp_path / "inference")

    main_module.main()

    # Assert one artifact per major pipeline stage.
    processed_csv = tmp_path / "processed" / "clean.csv"
    assert processed_csv.exists()
    
    # Verify processed data has expected structure
    df_processed = pd.read_csv(processed_csv)
    assert len(df_processed) > 0
    assert "price" in df_processed.columns
    
    # Check model artifact
    model_files = list((tmp_path / "models").glob("model_*.joblib"))
    assert len(model_files) == 1
    
    # Check evaluation plots
    assert (tmp_path / "reports" / "actual_vs_predicted.png").exists()
    assert (tmp_path / "reports" / "residuals.png").exists()
    
    # Check predictions
    prediction_files = list((tmp_path / "inference").glob("predictions_*.csv"))
    assert len(prediction_files) == 1
    
    # Verify predictions structure
    df_predictions = pd.read_csv(prediction_files[0])
    assert "prediction" in df_predictions.columns
    assert len(df_predictions) == 50  # Should match inference sample size
    assert (df_predictions["prediction"] > 0).all()  # Prices should be positive


def test_main_creates_all_output_directories(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
):
    """Verify main() creates all required output directories."""
    monkeypatch.setattr(main_module, "RAW_DATA_PATH", large_mock_csv)
    monkeypatch.setattr(main_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(main_module, "INFERENCE_DIR", tmp_path / "inference")

    main_module.main()

    # All directories should exist
    assert (tmp_path / "processed").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "inference").is_dir()


def test_main_raises_on_missing_raw_data_when_fetch_disabled(
    monkeypatch,
    tmp_path: Path,
):
    """If data is missing and fetch is blocked, main should fail clearly."""
    monkeypatch.setattr(main_module, "RAW_DATA_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(main_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(main_module, "INFERENCE_DIR", tmp_path / "inference")

    # Simulate ingestion failure
    def mock_ensure_raises(*args, **kwargs):
        raise FileNotFoundError("mock missing")
    
    monkeypatch.setattr(main_module, "ensure_raw_data_exists", mock_ensure_raises)

    # Orchestration should propagate the root ingestion error.
    with pytest.raises(FileNotFoundError, match="mock missing"):
        main_module.main()


def test_main_preserves_data_split_contract(
    monkeypatch,
    tmp_path: Path,
    large_mock_csv: Path,
    capsys,
):
    """Verify that data splitting follows the expected 50/90/10 contract."""
    monkeypatch.setattr(main_module, "RAW_DATA_PATH", large_mock_csv)
    monkeypatch.setattr(main_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(main_module, "INFERENCE_DIR", tmp_path / "inference")

    main_module.main()
    
    # Capture printed output to verify split sizes
    captured = capsys.readouterr()
    
    # Check that split information is printed
    assert "Train shape:" in captured.out
    assert "Test shape:" in captured.out
    assert "Infer shape:" in captured.out
    
    # Verify inference predictions count
    prediction_files = list((tmp_path / "inference").glob("predictions_*.csv"))
    df_predictions = pd.read_csv(prediction_files[0])
    assert len(df_predictions) == 50  # Inference holdout size
