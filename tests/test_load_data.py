"""Comprehensive pytest suite for src.load_data."""

import builtins
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

import src.load_data as load_data


TEST_DIR = Path(_file_).resolve().parent
MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture
def mock_csv_path() -> Path:
    return MOCK_CSV_PATH


def test_load_raw_data_reads_mock_csv(mock_csv_path: Path):
    """Happy path: load_raw_data returns a non-empty DataFrame."""
    df = load_data.load_raw_data(mock_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "price" in df.columns


def test_load_raw_data_raises_for_missing_file(tmp_path: Path):
    """Missing file should fail fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data.load_raw_data(
            tmp_path / "missing.csv",
            use_dummy_on_failure=False,
        )


def test_load_raw_data_raises_for_non_csv(tmp_path: Path):
    """Unsupported extension should fail fast with ValueError."""
    # Minimal non-CSV file to trigger extension validation in utils.load_csv.
    text_path = tmp_path / "not_csv.txt"
    text_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data.load_raw_data(text_path, use_dummy_on_failure=False)


def test_load_raw_data_raises_for_directory_path(tmp_path: Path):
    """Directory path should fail fast with IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        load_data.load_raw_data(tmp_path)


def test_load_raw_data_raises_for_empty_csv(tmp_path: Path):
    """Empty CSV should be rejected explicitly."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data.load_raw_data(empty_csv, use_dummy_on_failure=False)


def test_load_raw_data_raises_for_header_only_csv(tmp_path: Path):
    """CSV with headers but no rows should be rejected."""
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("price,area\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data.load_raw_data(header_only, use_dummy_on_failure=False)


def test_load_raw_data_raises_for_parser_error(tmp_path: Path):
    """Malformed CSV syntax should raise a parse-related ValueError."""
    # Unclosed quote creates a parser-level CSV failure.
    broken_csv = tmp_path / "broken.csv"
    broken_csv.write_text('price,area\n"100,200\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_data.load_raw_data(broken_csv, use_dummy_on_failure=False)


def test_load_raw_data_returns_dummy_when_all_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """If load and fetch both fail, return and save dummy data."""
    missing_path = tmp_path / "raw" / "missing.csv"

    def fake_fetch(path: Path, fetch_if_missing: bool = False):
        # Simulate unavailable remote source.
        raise RuntimeError("simulated fetch failure")

    # Force the explicit fetch call in load_raw_data to fail.
    monkeypatch.setattr(load_data, "ensure_raw_data_exists", fake_fetch)

    df = load_data.load_raw_data(missing_path, use_dummy_on_failure=True)

    # Fallback contract: deterministic tiny table with expected schema.
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {
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
        "price",
    }
    assert missing_path.exists()


def test_ensure_raw_data_exists_returns_existing_path(mock_csv_path: Path):
    """Existing path should be returned unchanged."""
    resolved = load_data.ensure_raw_data_exists(
        mock_csv_path,
        fetch_if_missing=True,
    )
    assert Path(resolved) == mock_csv_path


def test_ensure_raw_data_exists_raises_when_missing_and_no_fetch(
    tmp_path: Path,
):
    """Missing path without fetch should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data.ensure_raw_data_exists(
            tmp_path / "missing.csv",
            fetch_if_missing=False,
        )


def test_ensure_raw_data_exists_calls_fetch_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Missing path with fetch=True should delegate to fetch function."""
    destination = tmp_path / "downloaded.csv"

    def fake_fetch(path: Path) -> Path:
        # Stub creates the expected downloaded artifact.
        path.write_text("price\n123\n", encoding="utf-8")
        return path

    monkeypatch.setattr(load_data, "fetch_raw_data_from_kaggle", fake_fetch)

    resolved = load_data.ensure_raw_data_exists(
        destination,
        fetch_if_missing=True,
    )

    assert resolved == destination
    assert destination.exists()


def test_fetch_raw_data_from_kaggle_raises_if_kagglehub_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Missing kagglehub dependency should fail with clear ImportError."""
    destination = tmp_path / "Housing.csv"
    original_import = builtins._import_

    def fake_import(name, *args, **kwargs):
        if name == "kagglehub":
            raise ImportError("mock missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "_import_", fake_import)

    with pytest.raises(ImportError):
        load_data.fetch_raw_data_from_kaggle(destination)


def test_fetch_raw_data_from_kaggle_returns_existing_when_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """If destination exists and overwrite=False, no download is attempted."""
    destination = tmp_path / "Housing.csv"
    destination.write_text("price\n123\n", encoding="utf-8")

    # Fail immediately if code wrongly attempts dataset_download.
    fake_kagglehub = types.SimpleNamespace(
        dataset_download=lambda _args, *_kwargs: (_ for _ in ()).throw(
            AssertionError("dataset_download should not be called")
        )
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    resolved = load_data.fetch_raw_data_from_kaggle(
        destination,
        overwrite=False,
    )
    assert resolved == destination


def test_fetch_raw_data_from_kaggle_raises_when_csv_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """If cache lacks Housing.csv, function should raise FileNotFoundError."""
    cache_dir = tmp_path / "kaggle_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "other.txt").write_text("x", encoding="utf-8")

    fake_kagglehub = types.SimpleNamespace(
        dataset_download=lambda _args, *_kwargs: str(cache_dir)
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    with pytest.raises(FileNotFoundError):
        load_data.fetch_raw_data_from_kaggle(
            tmp_path / "out.csv",
            overwrite=True,
        )


def test_fetch_raw_data_from_kaggle_copies_single_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Happy path: cache CSV is copied to destination path."""
    cache_dir = tmp_path / "kaggle_cache"
    source_dir = cache_dir / "dataset"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_csv = source_dir / "Housing.csv"
    source_csv.write_text("price,area\n100,1000\n", encoding="utf-8")

    fake_kagglehub = types.SimpleNamespace(
        dataset_download=lambda _args, *_kwargs: str(cache_dir)
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    destination = tmp_path / "raw" / "Housing.csv"
    resolved = load_data.fetch_raw_data_from_kaggle(
        destination,
        overwrite=True,
    )

    assert resolved == destination
    assert destination.exists()
    assert destination.read_text(
        encoding="utf-8"
    ) == source_csv.read_text(encoding="utf-8")
