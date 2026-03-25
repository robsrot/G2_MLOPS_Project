"""Comprehensive pytest suite for src.utils."""

from pathlib import Path

import pandas as pd
import pytest

import src.utils as utils


TEST_DIR = Path(__file__).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


def test_load_csv_reads_mock_csv() -> None:
    """Happy path: load_csv returns a non-empty DataFrame."""
    df = utils.load_csv(MOCK_CSV_PATH)

    # Guards basic loader contract used by downstream training tests.
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "price" in df.columns


def test_load_csv_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing CSV should raise FileNotFoundError."""
    # Clear file-not-found errors simplify pipeline troubleshooting.
    with pytest.raises(FileNotFoundError):
        utils.load_csv(tmp_path / "missing.csv")


def test_load_csv_raises_for_non_csv(tmp_path: Path) -> None:
    """Non-CSV extension should raise ValueError."""
    # Prevents silent parsing of arbitrary text inputs as tabular data.
    text_path = tmp_path / "not_csv.txt"
    text_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        utils.load_csv(text_path)


def test_load_csv_allows_overriding_read_options(tmp_path: Path) -> None:
    """Caller can override pandas read options when needed."""
    # Non-default delimiter confirms call-time parsing knobs are honored.
    semicolon_csv = tmp_path / "data.csv"
    semicolon_csv.write_text("price;area\n100;1000\n", encoding="utf-8")

    df = utils.load_csv(semicolon_csv, read_options={"sep": ";"})

    assert list(df.columns) == ["price", "area"]
    assert len(df) == 1


def test_save_csv_creates_parent_directories_and_writes_file(
    tmp_path: Path,
) -> None:
    """save_csv should create parent directories and persist data."""
    # Mirrors real pipeline outputs where report paths may not pre-exist.
    output_path = tmp_path / "nested" / "out.csv"
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    utils.save_csv(frame, output_path)

    assert output_path.exists()
    loaded = pd.read_csv(output_path)
    assert loaded.equals(frame)


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    """Model artifact should round-trip through joblib serialization."""
    model = {"name": "demo", "version": 1}
    model_path = tmp_path / "model.joblib"

    utils.save_model(model, model_path)
    loaded = utils.load_model(model_path)

    # Catches subtle serialization drift between save/load helpers.
    assert loaded == model


def test_load_model_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing model path should raise FileNotFoundError."""
    # Inference startup should fail loudly when artifacts are absent.
    with pytest.raises(FileNotFoundError):
        utils.load_model(tmp_path / "missing.joblib")


def test_save_model_allows_non_joblib_extension(tmp_path: Path) -> None:
    """Non-joblib extension should still save successfully."""
    # Keeps helper tolerant of project-specific naming conventions.
    model = {"k": "v"}
    model_path = tmp_path / "model.pkl"

    utils.save_model(model, model_path)

    assert model_path.exists()
