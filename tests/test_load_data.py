"""Comprehensive pytest suite for src.load_data."""

import builtins
import importlib
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
import joblib


def _fake_load_csv(path: Path, read_options=None, **kwargs) -> pd.DataFrame:
    """Fake load_csv that accepts read_options dict and other kwargs."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected .csv file, got: {path}")

    # Unpack read_options if provided
    if read_options is None:
        read_options = {}

    return pd.read_csv(path, **read_options, **kwargs)


def _fake_save_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _fake_save_model(model, filepath: Path) -> None:
    """Fake save_model that actually saves using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def _fake_load_model(filepath: Path):
    """Fake load_model that loads from joblib."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    return joblib.load(filepath)


_fake_utils_module = types.ModuleType("src.utils")
_fake_utils_module.load_csv = _fake_load_csv
_fake_utils_module.save_csv = _fake_save_csv
_fake_utils_module.save_model = _fake_save_model
_fake_utils_module.load_model = _fake_load_model
# Replace src.utils at import time so this suite isolates load_data behavior
# from utility implementation changes and file-system side effects.
# We intentionally patch before importing src.load_data because Python binds
# imported symbols at module import time; patching later would miss that path.
sys.modules["src.utils"] = _fake_utils_module

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_data = importlib.import_module("src.load_data")


try:
    TEST_DIR = Path(__file__).resolve().parent
except NameError:
    TEST_DIR = Path.cwd()

MOCK_CSV_PATH = TEST_DIR / "mock_data" / "housing_small.csv"


@pytest.fixture(autouse=True)
def patch_paths_if_no__file_(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Patch path globals when _file_ is unavailable in the runtime."""
    if "__file__" not in globals():
        # Keeps tests resilient in interactive/embedded runners where
        # __file__ is absent, preventing path-resolution false negatives.
        monkeypatch.setattr(sys.modules[__name__], "TEST_DIR", tmp_path)
        monkeypatch.setattr(
            sys.modules[__name__],
            "MOCK_CSV_PATH",
            tmp_path / "mock_data" / "housing_small.csv",
        )


@pytest.fixture
def mock_csv_path() -> Path:
    return MOCK_CSV_PATH


def test_load_raw_data_reads_mock_csv(mock_csv_path: Path):
    """Happy path: load_raw_data returns a non-empty DataFrame."""
    # Baseline contract test: if this fails, downstream tests are less
    # informative because core ingestion is already broken.
    df = load_data.load_raw_data(mock_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
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
    # Guards against accidental ingestion of wrong file types that would
    # otherwise fail later with less actionable model-stage errors.
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
    # Empty files can appear from interrupted exports; failing here avoids
    # training on structurally valid but content-free inputs.
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data.load_raw_data(empty_csv, use_dummy_on_failure=False)


def test_load_raw_data_raises_for_header_only_csv(tmp_path: Path):
    """CSV with headers but no rows should be rejected."""
    # Header-only files are a subtle production failure mode; this ensures
    # the loader treats them as invalid data, not successful ingestion.
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("price,area\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data.load_raw_data(header_only, use_dummy_on_failure=False)


def test_load_raw_data_raises_for_parser_error(tmp_path: Path):
    """Malformed CSV syntax should raise a parse-related ValueError."""
    # Corrupted CSVs appear in real data handoffs; this ensures the loader
    # fails with a clear error instead of silently producing bad data.
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
        # Forces last-resort branch so fallback behavior remains explicit
        # and test-covered even when remote dependencies are down.
        raise RuntimeError("simulated fetch failure")

    # Exercises the deterministic dummy-data contract end to end.
    monkeypatch.setattr(load_data, "ensure_raw_data_exists", fake_fetch)

    df = load_data.load_raw_data(missing_path, use_dummy_on_failure=True)

    # Fallback contract: deterministic tiny table with expected schema.
    # Determinism matters so tests and demos remain reproducible even when
    # external data sources are unavailable.
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
    # Guards against unnecessary fetch/copy work when data already exists.
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
        # Returning a real file proves delegation behavior
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
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        # Simulates missing optional dependency to validate user-facing
        # remediation guidance in the raised ImportError path.
        if name == "kagglehub":
            raise ImportError("mock missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        load_data.fetch_raw_data_from_kaggle(destination)


def test_fetch_raw_data_from_kaggle_returns_existing_when_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """If destination exists and overwrite=False, no download is attempted."""
    destination = tmp_path / "Housing.csv"
    destination.write_text("price\n123\n", encoding="utf-8")

    # Protects idempotency contract: existing files must be reused unless the
    # caller explicitly opts into overwrite.
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
    # Ensures strict dataset integrity: wrong cache contents must not be
    # accepted silently because that would propagate invalid training data.
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

    # Byte-level equality check guarantees we validate copy semantics rather
    # than only file existence, which can hide partial/corrupted writes.
    assert resolved == destination
    assert destination.exists()
    assert destination.read_text(
        encoding="utf-8"
    ) == source_csv.read_text(encoding="utf-8")
