"""FastAPI serving layer for the Housing Price Predictor.

Loads the trained sklearn Pipeline once at startup, exposes a /predict
endpoint that accepts batch records, and streams inference logs to W&B
asynchronously.

Endpoints:
    GET  /        → API info
    GET  /health  → liveness check (503 if model not loaded)
    POST /predict → batch price prediction

Model loading is controlled by the MODEL_SOURCE env var:
    "local" → loads models/model.joblib from disk (default)
    "wandb"  → downloads the artifact aliased by WANDB_MODEL_ALIAS from
               the W&B Model Registry
"""

# Standard library
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Third-party
import pandas as pd
import wandb
import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

# Local
from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.logger import configure_logging
from src.utils import load_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HousingRecord(BaseModel):
    """Input schema for a single prediction request.

    All yes/no fields must be the lowercase strings "yes" or "no" —
    matching the raw Housing.csv format before clean_dataframe() runs.
    """

    model_config = ConfigDict(extra="forbid")  # reject any unexpected fields

    area: int                  # property area in square feet
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str              # "yes" or "no"
    guestroom: str             # "yes" or "no"
    basement: str              # "yes" or "no"
    hotwaterheating: str       # "yes" or "no"
    airconditioning: str       # "yes" or "no"
    parking: int
    prefarea: str              # "yes" or "no"
    furnishingstatus: str      # "furnished", "semi-furnished", or "unfurnished"


class PredictRequest(BaseModel):
    """Batch prediction request — one or more housing records."""

    records: list[HousingRecord]


class PredictionItem(BaseModel):
    """Single predicted price in original currency units."""

    prediction: float


class PredictResponse(BaseModel):
    """Batch prediction response — aligned 1-to-1 with the input records."""

    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Health check response body."""

    status: str
    model_version: str


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load and parse PROJECT_ROOT/config.yaml.

    Raises FileNotFoundError if the file is absent.
    Returns the parsed dict.
    """
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config.yaml must be a YAML mapping, got: {type(cfg)}")
    return cfg


# ---------------------------------------------------------------------------
# Inference log buffer
# ---------------------------------------------------------------------------

_inference_buffer: list[dict] = []
_buffer_lock = threading.Lock()


def _flush_inference_buffer_to_wandb(rows: list[dict]) -> None:
    """Log a batch of inference rows to W&B as a Table artifact.

    Designed to run inside a BackgroundTask — never raises; logs a warning
    if W&B is unavailable so the API remains unaffected.
    """
    if not rows:
        return

    try:
        table = wandb.Table(dataframe=pd.DataFrame(rows))
        wandb.log({"api/inference_log": table})
        logger.info("Flushed %d inference rows to W&B.", len(rows))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "W&B inference log flush failed (non-fatal): %s", exc
        )


# ---------------------------------------------------------------------------
# Lifespan — model loaded once, stored in app.state
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and config at startup; log shutdown when done."""

    # Load secrets from .env; do not override env vars already set by CI/shell
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    # Load config
    cfg = _load_config()

    # Set up logging as early as possible so all startup messages are captured
    logging_cfg = cfg.get("logging", {})
    log_file_str = cfg.get("paths", {}).get("log_file", "logs/api.log")
    configure_logging(
        log_level=logging_cfg.get("level", "INFO"),
        log_file=PROJECT_ROOT / log_file_str,
    )

    logger.info("API starting up — loading model...")

    # Derive feature column list from config (same order as main.py / validate)
    features_cfg = cfg.get("features", {})
    feature_cols: list[str] = (
        features_cfg.get("log_transform_cols", [])
        + features_cfg.get("numeric_passthrough", [])
        + features_cfg.get("binary_cols", [])
        + features_cfg.get("categorical_onehot", [])
    )
    app.state.feature_cols = feature_cols
    app.state.binary_cols = features_cfg.get("binary_cols", [])
    app.state.log_transform_cols = features_cfg.get("log_transform_cols", [])
    app.state.valid_furnishing_values = features_cfg.get("valid_furnishing_values", [])
    app.state.batch_size = cfg.get("wandb", {}).get("inference_buffer_size", 50)

    model_source = os.getenv("MODEL_SOURCE", "local")

    if model_source == "wandb":
        # Download the registered model artifact from W&B Model Registry
        entity       = os.getenv("WANDB_ENTITY", "")
        alias        = os.getenv("WANDB_MODEL_ALIAS", "prod")
        registry_name = cfg.get("wandb", {}).get(
            "model_registry_name", "housing-price-predictor"
        )

        logger.info(
            "MODEL_SOURCE=wandb — downloading artifact %s/%s:%s",
            entity, registry_name, alias,
        )

        api_client = wandb.Api()
        artifact = api_client.artifact(
            f"{entity}/model-registry/{registry_name}:{alias}"
        )
        artifact_dir = Path(
            artifact.download(root=str(PROJECT_ROOT / "models"))
        )

        # Locate the .joblib file inside the downloaded directory
        joblib_candidates = sorted(artifact_dir.rglob("*.joblib"))
        if not joblib_candidates:
            raise FileNotFoundError(
                f"No .joblib file found in downloaded artifact at {artifact_dir}"
            )
        model_path = joblib_candidates[0]
        app.state.model_version = alias

    else:
        # Default: load from local disk path specified in config
        model_path_str = cfg.get("paths", {}).get("model_artifact", "models/model.joblib")
        model_path = PROJECT_ROOT / model_path_str
        app.state.model_version = model_path.name
        logger.info("MODEL_SOURCE=local — loading from %s", model_path)

    app.state.model_pipeline = load_model(model_path)

    logger.info(
        "Model loaded successfully. version=%s  source=%s",
        app.state.model_version,
        model_source,
    )

    yield  # --- app is serving requests below this line ---

    logger.info("API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Housing Price Predictor",
    description="Predicts house prices from property attributes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# HTTP middleware — correlation ID + request logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def correlation_and_logging_middleware(
    request: Request, call_next: Any
) -> Response:
    """Attach X-Correlation-ID to every response and log request metadata."""
    correlation_id = str(uuid.uuid4())
    start_time = time.time()

    response = await call_next(request)

    latency_ms = (time.time() - start_time) * 1000
    model_version = getattr(request.app.state, "model_version", "unknown")

    response.headers["X-Correlation-ID"] = correlation_id

    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f "
        "correlation_id=%s model_version=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        correlation_id,
        model_version,
    )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict:
    """API info and navigation links."""
    return {
        "message": "Housing Price Predictor API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> Any:
    """Liveness check.

    Returns 503 if the model has not been loaded (e.g. startup failed).
    Returns 200 with model version if the pipeline is ready.
    """
    model_pipeline = getattr(request.app.state, "model_pipeline", None)

    if model_pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_version": "none"},
        )

    return HealthResponse(
        status="ok",
        model_version=getattr(request.app.state, "model_version", "unknown"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> PredictResponse:
    """Batch price prediction.

    Accepts one or more HousingRecord objects, runs the full
    clean → validate → infer pipeline, and returns predicted prices.
    Inference rows are buffered and flushed to W&B asynchronously.

    FastAPI automatically returns 422 for malformed/missing fields
    (enforced by Pydantic before this function is called).
    """
    model_pipeline: Any = request.app.state.model_pipeline
    feature_cols: list[str] = request.app.state.feature_cols
    binary_cols: list[str] = request.app.state.binary_cols
    log_transform_cols: list[str] = request.app.state.log_transform_cols
    valid_furnishing_values: list[str] = request.app.state.valid_furnishing_values

    # 1. Convert Pydantic records to raw DataFrame (pre-cleaning format)
    df_raw = pd.DataFrame([r.model_dump() for r in body.records])

    # 2. Clean — binary encoding + log1p(area), using the same function
    #    as the training pipeline to prevent train/serve skew
    try:
        df_clean = clean_dataframe(
            df_raw,
            allow_duplicates=True,
            binary_cols=binary_cols,
            log_transform_cols=log_transform_cols,
        )
    except ValueError as exc:
        logger.warning("Cleaning failed for request: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"detail": f"Data cleaning failed: {exc}"},
        )

    # 3. Validate feature schema — no target column in inference data
    try:
        validate_dataframe(
            df_clean,
            required_columns=feature_cols,
            binary_cols=binary_cols,
            valid_furnishing_values=valid_furnishing_values,
        )
    except ValueError as exc:
        logger.warning("Validation failed for request: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"detail": f"Validation failed: {exc}"},
        )

    # 4. Run inference — returns DataFrame with "prediction" column
    df_predictions = run_inference(pipeline=model_pipeline, X_infer=df_clean)
    predictions = df_predictions["prediction"].tolist()

    # 5. Buffer inference rows for async W&B logging
    rows_for_buffer = []
    for record, pred in zip(body.records, predictions):
        row = record.model_dump()
        row["prediction"] = pred
        rows_for_buffer.append(row)

    rows_to_flush: list[dict] = []
    with _buffer_lock:
        _inference_buffer.extend(rows_for_buffer)
        if len(_inference_buffer) >= request.app.state.batch_size:
            rows_to_flush = _inference_buffer.copy()
            _inference_buffer.clear()

    if rows_to_flush:
        background_tasks.add_task(_flush_inference_buffer_to_wandb, rows_to_flush)

    return PredictResponse(
        predictions=[PredictionItem(prediction=p) for p in predictions]
    )
