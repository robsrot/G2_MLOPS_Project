"""Comprehensive integration tests for src.main orchestration."""

from pathlib import Path

import pytest
import src.main as main_module

TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def mock_csv_path() -> Path:
    return MOCK_CSV_PATH


def test_main_runs_with_mock_data(
    monkeypatch,
    tmp_path: Path,
    mock_csv_path: Path,
):
    """Happy path: main() writes all expected artifacts to configured dirs."""
    # Redirect all pipeline outputs into pytest temp folders.
    monkeypatch.setattr(main_module, "RAW_DATA_PATH", mock_csv_path)
    monkeypatch.setattr(main_module, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(main_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(main_module, "INFERENCE_DIR", tmp_path / "inference")

    main_module.main()

    # Assert one artifact per major pipeline stage.
    assert (tmp_path / "processed" / "clean.csv").exists()
    assert any((tmp_path / "models").glob("model_*.joblib"))
    assert (tmp_path / "reports" / "actual_vs_predicted.png").exists()
    assert (tmp_path / "reports" / "residuals.png").exists()
    assert any((tmp_path / "inference").glob("predictions_*.csv"))


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

    # Simulate ingestion failure regardless of source configuration.
    monkeypatch.setattr(
        main_module,
        "ensure_raw_data_exists",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("mock missing")
        ),
    )

    # Orchestration should propagate the root ingestion error.
    with pytest.raises(FileNotFoundError):
        main_module.main()
