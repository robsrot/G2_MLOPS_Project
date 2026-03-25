"""Tests for the FastAPI serving layer (src/api.py).

Pre-condition: run python -m src.main at least once to produce
models/model.joblib. Without it the lifespan will raise FileNotFoundError
and all tests in this module will be collected but skipped via the fixture.

Environment flags set BEFORE the app import so the lifespan reads them:
    MODEL_SOURCE=local  → avoids any W&B artifact download
    WANDB_MODE=disabled → silences all wandb calls (init, log, etc.)
"""

# Standard library
import os

# Environment must be set before app import so lifespan picks them up
os.environ.setdefault("MODEL_SOURCE", "local")
os.environ.setdefault("WANDB_MODE", "disabled")

# Third-party
import pytest
from starlette.testclient import TestClient

# Local
from src.api import app

# ---------------------------------------------------------------------------
# Valid payload constant — field names must match HousingRecord exactly
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "records": [
        {
            "area": 7420,
            "bedrooms": 4,
            "bathrooms": 2,
            "stories": 3,
            "mainroad": "yes",
            "guestroom": "no",
            "basement": "no",
            "hotwaterheating": "no",
            "airconditioning": "yes",
            "parking": 2,
            "prefarea": "yes",
            "furnishingstatus": "furnished",
        }
    ]
}


# ---------------------------------------------------------------------------
# Session-scoped fixture — model loaded once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Start the FastAPI app (including lifespan) once for the whole session.

    If models/model.joblib does not exist the lifespan raises FileNotFoundError
    and all tests that depend on this fixture are automatically skipped.
    Run `python -m src.main` first to produce the model artifact.
    """
    try:
        with TestClient(app) as c:
            yield c
    except Exception as exc:
        pytest.skip(
            f"Could not start API — model file missing or lifespan failed: {exc}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    """GET / returns 200 and includes a 'message' key in the body."""
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_returns_200_or_503(client):
    """GET /health returns 200 (model loaded) or 503 (model absent).

    Both are valid — 503 is expected in CI environments where the model
    artifact has not been produced yet.
    """
    response = client.get("/health")

    assert response.status_code in (200, 503)

    body = response.json()
    if response.status_code == 200:
        assert "status" in body
        assert "model_version" in body
    else:
        assert body.get("status") == "unavailable"


def test_health_200_has_model_version(client):
    """When the model is loaded, model_version is a non-empty string."""
    response = client.get("/health")

    if response.status_code == 503:
        pytest.skip("Model not loaded — skipping model_version check")

    body = response.json()
    assert isinstance(body["model_version"], str)
    assert len(body["model_version"]) > 0


def test_predict_valid_payload_returns_200(client):
    """POST /predict with a valid record returns 200 with a positive prediction."""
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert "prediction" in body["predictions"][0]

    predicted_price = body["predictions"][0]["prediction"]
    assert isinstance(predicted_price, float)
    assert predicted_price > 0, (
        f"Predicted house price must be positive, got {predicted_price}"
    )


def test_predict_returns_correct_count(client):
    """Output length matches input length for a batch of 3 records."""
    single_record = VALID_PAYLOAD["records"][0]
    payload = {"records": [single_record, single_record, single_record]}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_predict_missing_field_returns_422(client):
    """A record missing the required 'area' field is rejected with 422.

    Pydantic raises RequestValidationError before the endpoint function
    runs, so no model inference is attempted.
    """
    record_without_area = {
        k: v
        for k, v in VALID_PAYLOAD["records"][0].items()
        if k != "area"
    }
    payload = {"records": [record_without_area]}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_extra_field_returns_422(client):
    """A record with an unknown extra field is rejected with 422.

    Enforced by ConfigDict(extra='forbid') on HousingRecord before the
    endpoint function runs.
    """
    record_with_extra = {**VALID_PAYLOAD["records"][0], "price": 500000}
    payload = {"records": [record_with_extra]}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_response_has_correlation_id_header(client):
    """Every /predict response carries a non-empty X-Correlation-ID header.

    HTTP headers are case-insensitive; the requests library lowercases them,
    so we check for 'x-correlation-id'.
    """
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert "x-correlation-id" in response.headers
    assert len(response.headers["x-correlation-id"]) > 0
