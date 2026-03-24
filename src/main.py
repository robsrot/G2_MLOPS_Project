"""Main pipeline entry point.

Reads all settings from config.yaml, secrets from .env, configures logging,
integrates Weights & Biases, and orchestrates the full pipeline:
load → clean → validate → train → evaluate → save model → infer.

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
"""

# Standard library
import logging
from pathlib import Path

# Third-party
import wandb
import yaml
from dotenv import load_dotenv

# Local
from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model, save_evaluation_plots
from src.infer import run_inference
from src.load_data import ensure_raw_data_exists, load_raw_data
from src.logger import configure_logging
from src.train import train_model
from src.utils import load_csv, save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Load and parse config.yaml. Raises if file missing or not valid YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config.yaml must be a YAML mapping, got: {type(cfg)}"
        )
    return cfg


def _require_section(cfg: dict, key: str) -> dict:
    """Return cfg[key] if it exists and is a dict. Raises KeyError with clear
      message if not."""
    if key not in cfg:
        raise KeyError(f"Required section '{key}' missing from config.yaml")
    if not isinstance(cfg[key], dict):
        raise KeyError(
            f"Config section '{key}' must be a mapping, got: {type(cfg[key])}"
        )
    return cfg[key]


def _require_str(section: dict, key: str) -> str:
    """Return section[key] if it exists and is a non-empty string. Raises if
      missing or empty."""
    if key not in section:
        raise KeyError(f"Required key '{key}' missing from config section")
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise KeyError(
            f"Config key '{key}' must be a non-empty string, got: {value!r}"
        )
    return value


def _resolve_path(project_root: Path, relative: str) -> Path:
    """Join project_root with a relative path string from config. Returns
      absolute Path."""
    return (project_root / relative).resolve()


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _wandb_get_str(cfg: dict, key: str, default: str = "") -> str:
    """Safely get a string value from cfg['wandb'][key]. Returns default if
      missing."""
    value = cfg.get("wandb", {}).get(key, default)
    return value if isinstance(value, str) else default


def _wandb_get_bool(cfg: dict, key: str, default: bool = False) -> bool:
    """Safely get a bool value from cfg['wandb'][key]. Returns default if
      missing."""
    value = cfg.get("wandb", {}).get(key, default)
    return bool(value) if isinstance(value, bool) else default


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete training/evaluation/inference artifact pipeline."""

    # 1. Resolve project root (one level up from src/)
    project_root = Path(__file__).resolve().parents[1]

    # 2. Load secrets from .env — override=False keeps env vars already set
    #    by CI or the shell (e.g. WANDB_API_KEY set in GitHub Actions secrets)
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # 3. Load config
    cfg = _load_config(project_root / "config.yaml")

    # 4. Extract sections
    paths_cfg = _require_section(cfg, "paths")
    data_cfg = _require_section(cfg, "data")
    logging_cfg = _require_section(cfg, "logging")
    problem_cfg = _require_section(cfg, "problem")
    split_cfg = _require_section(cfg, "split")
    training_cfg = _require_section(cfg, "training")
    features_cfg = _require_section(cfg, "features")
    validation_cfg = _require_section(cfg, "validation")
    evaluation_cfg = _require_section(cfg, "evaluation")
    run_cfg = _require_section(cfg, "run")

    # 5. Resolve all paths from config (all absolute, relative to project root)
    raw_data_path = _resolve_path(
        project_root, _require_str(paths_cfg, "raw_data")
        )
    processed_data_path = _resolve_path(
        project_root, _require_str(paths_cfg, "processed_data")
        )
    model_artifact_path = _resolve_path(
        project_root, _require_str(paths_cfg, "model_artifact")
        )
    inference_data_path = _resolve_path(
        project_root, _require_str(paths_cfg, "inference_data")
        )
    predictions_artifact_path = _resolve_path(
        project_root, _require_str(paths_cfg, "predictions_artifact")
        )
    log_file_path = _resolve_path(
        project_root, _require_str(paths_cfg, "log_file")
        )
    reports_dir = predictions_artifact_path.parent  # reports/

    # 6. Set up logging (must happen before any logger.* calls below)
    configure_logging(
        log_level=logging_cfg.get("level", "INFO"),
        log_file=log_file_path,
    )

    # 7. Startup banner
    logger.info("Housing Prices pipeline starting")

    target_column = _require_str(problem_cfg, "target_column")
    log_to_wandb = bool(run_cfg.get("log_to_wandb", False))

    # Build the full required-column list from config for training validation.
    # Order: target first, then all feature groups — mirrors schema.py.
    feature_cols = (
        features_cfg.get("log_transform_cols", [])
        + features_cfg.get("numeric_passthrough", [])
        + features_cfg.get("binary_cols", [])
        + features_cfg.get("categorical_onehot", [])
    )
    required_columns = [target_column] + feature_cols
    binary_cols = features_cfg.get("binary_cols", [])
    valid_furnishing_values = features_cfg.get("valid_furnishing_values", [])

    # 8. Initialise W&B run
    wandb_run = None
    _wandb_exit_code = 0

    if log_to_wandb:
        wandb_run = wandb.init(
            project=_wandb_get_str(cfg, "project"),
            config=cfg,
            job_type=_wandb_get_str(cfg, "job_type", "training-pipeline"),
            group=_wandb_get_str(cfg, "group", ""),
            tags=cfg.get("wandb", {}).get("tags", []),
            notes=_wandb_get_str(cfg, "notes", ""),
        )
        logger.info(
            "W&B run initialised: name=%s  project=%s",
            wandb.run.name,
            wandb.run.project,
        )
    else:
        logger.info("W&B disabled (run.log_to_wandb=false in config.yaml)")

    # 9. Pipeline steps — W&B run is always finished in finally
    try:
        # ------------------------------------------------------------------
        # 10. LOAD raw training data
        # ------------------------------------------------------------------
        fetch_if_missing = data_cfg.get("fetch_if_missing", True)
        use_dummy_on_failure = data_cfg.get("use_dummy_on_failure", True)
        kaggle_dataset = data_cfg.get("kaggle_dataset", "yasserh/housing-prices-dataset")
        kaggle_filename = data_cfg.get("kaggle_filename", "Housing.csv")
        ensure_raw_data_exists(
            raw_data_path,
            fetch_if_missing=fetch_if_missing,
            kaggle_dataset=kaggle_dataset,
            kaggle_filename=kaggle_filename,
        )
        df_raw = load_raw_data(
            raw_data_path,
            use_dummy_on_failure=use_dummy_on_failure,
            kaggle_dataset=kaggle_dataset,
            kaggle_filename=kaggle_filename,
        )
        logger.info(
            "Raw data loaded: %d rows, %d cols",
            df_raw.shape[0], df_raw.shape[1]
            )

        if wandb_run:
            wandb.log(
                {"data/raw_rows": df_raw.shape[0],
                 "data/raw_cols": df_raw.shape[1]}
                )

        # ------------------------------------------------------------------
        # 11. CLEAN
        # ------------------------------------------------------------------
        drop_missing_rows = data_cfg.get("drop_missing_rows", False)
        allow_duplicates = data_cfg.get("allow_duplicates", False)
        binary_cols = features_cfg.get("binary_cols", [])
        log_transform_cols = features_cfg.get("log_transform_cols", [])
        df_clean = clean_dataframe(
            df_raw,
            binary_cols=binary_cols,
            log_transform_cols=log_transform_cols,
            drop_missing_rows=drop_missing_rows,
            allow_duplicates=allow_duplicates,
        )

        if wandb_run:
            wandb.log(
                {"data/clean_rows": df_clean.shape[0],
                 "data/clean_cols": df_clean.shape[1]}
                )

        # ------------------------------------------------------------------
        # 12. VALIDATE (full schema including target)
        # ------------------------------------------------------------------
        validate_dataframe(
            df_clean,
            required_columns=required_columns,
            binary_cols=binary_cols,
            valid_furnishing_values=valid_furnishing_values,
        )

        # ------------------------------------------------------------------
        # 13. SAVE PROCESSED
        # ------------------------------------------------------------------
        save_csv(df_clean, processed_data_path)
        logger.info("Processed data saved → %s", processed_data_path)

        if wandb_run and _wandb_get_bool(cfg, "log_dataset"):
            dataset_artifact = wandb.Artifact(
                name="housing-dataset", type="dataset"
                )
            dataset_artifact.add_file(str(processed_data_path))
            wandb.log_artifact(dataset_artifact)
            logger.info("Dataset artifact logged to W&B")

        # ------------------------------------------------------------------
        # 14. TRAIN — K-Fold CV then final refit on all rows (Model 5)
        # n_folds=%d and fit_intercept are configured in config.yaml and
        # used by train.py internally; train_model() reads them at import time.
        # ------------------------------------------------------------------
        logger.info(
            "Training: model_type=%s  n_folds=%d  target=%s",
            training_cfg.get("regression", {})
            .get("model_type", "linear_regression"),
            split_cfg.get("n_folds", 5),
            target_column,
        )
        n_folds = split_cfg.get("n_folds", 5)
        random_state = split_cfg.get("random_state", 42)
        shuffle = split_cfg.get("shuffle", True)
        fit_intercept = training_cfg.get("regression", {}).get("fit_intercept", True)
        numeric_cols = features_cfg.get("numeric_passthrough", [])
        categorical_cols = features_cfg.get("categorical_onehot", [])
        model_pipeline, cv_results = train_model(
            df_clean, target_column,
            n_folds=n_folds, random_state=random_state, shuffle=shuffle,
            fit_intercept=fit_intercept,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            binary_cols=binary_cols,
            )

        # ------------------------------------------------------------------
        # 15. EVALUATE
        # ------------------------------------------------------------------
        n_bins_residuals = evaluation_cfg.get("n_bins_residuals", 30)
        plot_title_suffix = evaluation_cfg.get("plot_title_suffix", "K-Fold CV")
        metrics = evaluate_model(cv_results, n_folds=n_folds)
        logger.info("CV metrics: %s", metrics)

        if wandb_run:
            wandb.log({
                "metrics/val/rmse":   metrics["rmse"],
                "metrics/val/mae":    metrics["mae"],
                "metrics/val/r2":     metrics["r2"],
                "metrics/val/adj_r2": metrics["adjusted_r2"],
            })

        if evaluation_cfg.get("save_plots", False):
            save_evaluation_plots(
                cv_results,
                reports_dir=reports_dir,
                n_folds=n_folds,
                plot_title_suffix=plot_title_suffix,
                n_bins_residuals=n_bins_residuals,
            )

            if wandb_run and _wandb_get_bool(cfg, "log_plots"):
                wandb.log({
                    "plots/actual_vs_predicted": wandb.Image(
                        str(reports_dir / "actual_vs_predicted.png")
                    ),
                    "plots/residuals": wandb.Image(
                        str(reports_dir / "residuals.png")
                    ),
                })
                logger.info("Evaluation plots logged to W&B")

        # ------------------------------------------------------------------
        # 16. SAVE MODEL
        # ------------------------------------------------------------------
        save_model(model_pipeline, model_artifact_path)
        logger.info("Model saved → %s", model_artifact_path)

        if wandb_run and _wandb_get_bool(cfg, "log_model"):
            artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", "housing-model"
                )
            model_artifact = wandb.Artifact(name=artifact_name, type="model")
            model_artifact.add_file(str(model_artifact_path))
            wandb.log_artifact(model_artifact)
            logger.info(
                "Model artifact '%s' logged to W&B from %s",
                artifact_name,
                model_artifact_path,
            )

        # ------------------------------------------------------------------
        # 17. INFERENCE
        # ------------------------------------------------------------------
        df_infer_raw = load_csv(inference_data_path)
        logger.info(
            "Inference data loaded: %d rows, %d cols",
            df_infer_raw.shape[0], df_infer_raw.shape[1]
            )

        # Clean inference data — no target column present
        df_infer_clean = clean_dataframe(
            df_infer_raw,
            binary_cols=binary_cols,
            log_transform_cols=log_transform_cols,
            drop_missing_rows=drop_missing_rows,
            allow_duplicates=allow_duplicates,
        )

        # Validate feature columns only (no target in inference data)
        validate_dataframe(
            df_infer_clean,
            required_columns=feature_cols,
            binary_cols=binary_cols,
            valid_furnishing_values=valid_furnishing_values,
        )

        df_predictions = run_inference(
            pipeline=model_pipeline,
            X_infer=df_infer_clean
            )

        if run_cfg.get("save_predictions", False):
            save_csv(df_predictions, predictions_artifact_path)
            logger.info("Predictions saved → %s", predictions_artifact_path)

            if wandb_run and _wandb_get_bool(cfg, "log_predictions"):
                pred_artifact = wandb.Artifact(
                    name="housing-predictions",
                    type="predictions"
                    )
                pred_artifact.add_file(str(predictions_artifact_path))
                wandb.log_artifact(pred_artifact)
                logger.info("Predictions artifact logged to W&B")

        # ------------------------------------------------------------------
        # 18. Done
        # ------------------------------------------------------------------
        logger.info("Pipeline complete")

    except Exception:
        _wandb_exit_code = 1
        logger.exception("Pipeline failed with unhandled exception")
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish(exit_code=_wandb_exit_code)


if __name__ == "__main__":
    main()
