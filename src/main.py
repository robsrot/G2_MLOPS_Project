"""Main pipeline entry point.

This module orchestrates the end-to-end workflow in a readable order:
load data, clean data, validate schema/domain rules, train the model,
evaluate model quality, and save deployable artifacts.

Educational Goal:
- Why this module exists in an MLOps system:
    Orchestration scripts provide a reproducible, auditable record of
    the exact steps taken to produce a model.
- Responsibility (separation of concerns):
    This module owns the high-level pipeline flow, calling functions
    from other modules in the correct sequence.
- Pipeline contract (inputs and outputs):
    Reads configuration, executes end-to-end pipeline, and writes
    artifacts to standardized paths.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be
imported from config.yml in a later session
"""

from datetime import datetime
from pathlib import Path

from src.load_data import ensure_raw_data_exists, load_raw_data
from src.clean_data import clean_dataframe
from src.schema import REQUIRED_COLUMNS
from src.validate import validate_dataframe
from src.train import train_model
from src.evaluate import evaluate_model, save_evaluation_plots
from src.infer import run_inference
from src.utils import save_csv, save_model


# Paths and configuration
RAW_DATA_PATH = Path("data/raw/Housing.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
INFERENCE_DIR = Path("data/inference")
TARGET_COLUMN = "price"


def main():
    """Run the complete training/evaluation/inference artifact pipeline."""

    # Timestamped run_id keeps artifacts from different runs separate.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 0. Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ensure data exists (download on-demand when missing).
    ensure_raw_data_exists(RAW_DATA_PATH, fetch_if_missing=True)

    # 2. Load raw table from disk (or fallback path inside loader).
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 3. Clean
    df_clean = clean_dataframe(df_raw)

    # 4. Save processed CSV
    processed_path = PROCESSED_DIR / "clean.csv"
    save_csv(df_clean, processed_path)
    print(f"[main] Processed data saved → {processed_path}")

    # 5. Enforce schema/domain assumptions before model training.
    validate_dataframe(df_clean, required_columns=REQUIRED_COLUMNS)

    # 6. Train (Model 5): 5-fold CV + refit on all cleaned rows.
    pipeline, cv_results = train_model(df_clean, target_column=TARGET_COLUMN)

    # 7. Save model
    # To load: pipeline = joblib.load("models/model_<timestamp>.joblib")
    model_path = MODELS_DIR / f"model_{run_id}.joblib"
    save_model(pipeline, model_path)
    print(f"[main] Model saved → {model_path}")

    # 8. Evaluate
    metrics = evaluate_model(cv_results)
    print(f"[main] Final metrics: {metrics}")

    # 9. Save evaluation plots into reports
    save_evaluation_plots(cv_results, reports_dir=REPORTS_DIR)

    # 10. Example inference on first 5 rows to produce deployable output.
    X_sample = df_clean.drop(columns=[TARGET_COLUMN]).head(10)
    predictions = run_inference(pipeline, X_sample)

    predictions_path = INFERENCE_DIR / f"predictions_{run_id}.csv"
    save_csv(predictions, predictions_path)
    print(f"[main] Predictions saved → {predictions_path}")
    print(predictions)

    print(f"\n[main] Run complete  |  run_id: {run_id}")
    print("[main] Artifacts saved:")
    print(f"{processed_path}")
    print(f"{model_path}")
    print(f"{REPORTS_DIR / 'actual_vs_predicted.png'}")
    print(f"{REPORTS_DIR / 'residuals.png'}")
    print(f"{predictions_path}")


if __name__ == "__main__":
    main()
