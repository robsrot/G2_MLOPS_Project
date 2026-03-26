# Housing Price Predictor

A production-ready MLOps service that generates instant, data-driven price estimates for residential property listings from 12 property attributes.

[![MLOps Quality Gate](https://github.com/robsrot/G2_MLOPS_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/robsrot/G2_MLOPS_Project/actions/workflows/ci.yml)
[![Live on Render](https://img.shields.io/badge/API-Live%20on%20Render-brightgreen)](https://g2-mlops-project.onrender.com/health)

---

## Business Objective

Real estate agencies price new listings inconsistently. Without a formal appraisal, agents rely on intuition — the same property receives different estimates depending on who handles it. Overpriced listings sit on the market; underpriced ones close fast but leave revenue on the table.

This service is an automated first-pass valuation tool. The moment a new listing is registered, agents input 12 attributes they already collect at intake — size, layout, amenities, location indicators — and receive a data-driven reference price before any further assessment is needed.

**Users:** Listing agents (primary), sellers, agency management, and financial institutions seeking an independent cross-check on declared property values.

**Deployment condition:** Model output is always reviewed by an agent before being communicated to a seller. It is never surfaced as a final price.

---

## Project Background

This project was built as the final assignment for the MLOps Engineering course at IE University (MsC Business Analytics and Data Science, March 2026).

The pipeline evolved in two phases:

- **Phase 1** converted a Jupyter Notebook into a modular Python project with a clean `src/` layout, deterministic preprocessing, and a reproducible training script.
- **Phase 2** added production-grade MLOps practices: Weights & Biases experiment tracking, FastAPI serving, Docker containerisation, and CI/CD via GitHub Actions with automated deployment to Render.

The dataset is the [Kaggle Housing Prices dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) — 545 observations, 13 columns — representing residential property sales in a single market.

---

## Success Metrics

**Business KPIs:**
- Pricing turnaround — estimate available at listing creation (0 wait time)
- Agent consistency — every agent starts from the same model-generated reference
- Estimation accuracy — predicted price within ±20% of actual sale price at the median

*At the median sale price of 4.34M, ±20% = ±868,000. The achieved MAE of 747,580 falls within this band, making predictions commercially useful as a first-pass anchor.*

**Technical acceptance criteria:**

| Criterion | Threshold | Result |
|---|---|---|
| R² | ≥ 0.65 | 0.653 |
| Adjusted R² | ≥ 0.64 | 0.644 |
| MAE | ≤ 868,000 (±20% of median) | 747,580 |
| RMSE | — | 1,039,102 |
| CV stability | No single fold deviates > 5% R² from mean | Confirmed across 5 folds |

---

## Architecture Overview

The system is a linear pipeline from raw data to a live REST endpoint:

```
data/raw/Housing.csv → main.py → W&B (metrics + artifacts) → models/model.joblib → api.py → Render
```

**MLOps layers:**

| Layer | Implementation |
|---|---|
| Configuration | `config.yaml` — single source of truth for all non-secret settings |
| Secrets | `.env` — loaded at runtime via `python-dotenv`, never committed |
| Logging | `src/logger.py` — dual `StreamHandler` + `FileHandler` output |
| Experiment tracking | Weights & Biases — metrics, model artifacts, inference logs |
| CI/CD | GitHub Actions — quality gate on PRs, deploy hook on Release |
| Deployment | Render — containerised FastAPI service |

**Layer details:**

- **Configuration:** All non-secret runtime settings (model hyperparameters, file paths, feature lists, thresholds) live in `config.yaml`. Hardcoding values in source files is avoided so that any change requires touching only one place and is immediately visible in version control.
- **Secrets:** The `.env` file is listed in `.gitignore` and never committed. `python-dotenv` loads it into the process environment at runtime, keeping credentials out of the codebase entirely.
- **Logging:** `src/logger.py` writes every log line to both the console (`StreamHandler`) and a rotating file (`FileHandler`). Replacing `print()` with structured log calls means severity levels, timestamps, and module names are captured in `logs/pipeline.log` for post-run diagnosis.
- **Experiment tracking:** On every training run, Weights & Biases records the CV metrics (R², MAE, RMSE), the processed dataset, the trained model artifact, and diagnostic plots. This creates a permanent, reproducible record of every experiment so results can be compared and rolled back.
- **CI/CD:** Every pull request to `main` automatically runs the full test suite and validates the Docker build. A human-triggered GitHub Release sends a deploy hook to Render, so no code reaches production without passing the quality gate first.
- **Deployment:** The FastAPI service runs inside a Docker container on Render. The container is built from the project's `Dockerfile` and started by Render when the deploy hook fires, ensuring the live environment matches the local Docker build exactly.

**Model comparison — five models were developed; Model 5 was selected:**

| Model | Approach | R² | Adj. R² | MAE | RMSE | Features Used |
|---|---|---|---|---|---|---|
| 1 — Baseline | Binary encoding + OHE + StandardScaler | 0.65 | 0.61 | 9.70e+05 | 1.32e+06 | 13 |
| 2 — Log Price | + log(price) + log(area) | 0.66 | 0.61 | 9.70e+05 | 1.31e+06 | 13 |
| 3 — Log + Lasso | + LassoCV feature selection | 0.63 | 0.60 | 1.00e+06 | 1.36e+06 | 10 |
| 4 — Log + Outlier Removal | + IQR outlier removal on training set | 0.65 | 0.61 | 9.75e+05 | 1.32e+06 | 13 |
| **5 — K-Fold CV** | **Model 2 preprocessing + 5-fold cross-validation** | **0.66** | **0.65** | **7.68e+05** | **1.05e+06** | **13** |

Model 5 was selected because K-Fold CV ensures performance is not an artefact of a single favourable train/test split — the result holds across all five data partitions.

---

## Repository Structure

```text
G2_MLOPS_Project/
├── README.md
├── config.yaml                        # All non-secret runtime settings (60+ keys)
├── environment.yml                    # Conda environment specification
├── conda-lock.yml                     # Pinned Linux-64 lockfile for reproducibility
├── Dockerfile                         # Container image for the FastAPI service
├── .dockerignore                      # Allowlist — only src/, models/, config.yaml copied
├── pytest.ini                         # Test discovery configuration
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Quality gate — runs on all PRs to main
│       └── deploy.yml                 # CD — triggers on published GitHub Release
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # Pipeline orchestrator (entry point)
│   ├── logger.py                      # Root logging configuration
│   ├── load_data.py                   # Data ingestion (local CSV or Kaggle)
│   ├── clean_data.py                  # Deterministic cleaning and encoding
│   ├── validate.py                    # Schema, dtype, and value checks
│   ├── features.py                    # Unfitted ColumnTransformer recipe
│   ├── train.py                       # 5-fold CV training + final refit
│   ├── evaluate.py                    # Metrics + diagnostic plots
│   ├── infer.py                       # Inference on new data
│   ├── api.py                         # FastAPI serving layer
│   └── utils.py                       # File I/O helpers
│
├── data/
│   ├── raw/                           # Housing.csv (not committed)
│   ├── processed/                     # clean.csv (generated)
│   └── inference/                     # housing_inference.csv
│
├── models/
│   └── model.joblib                   # Trained pipeline artifact
│
├── reports/
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── predictions.csv
│
├── notebooks/
│   └── HousingPricesPrediction.ipynb  # Exploratory analysis (read-only sandbox)
│
├── logs/
│   └── pipeline.log                   # Runtime log output (not committed)
│
└── tests/
    ├── __init__.py
    ├── mock_data/
    │   └── housing_small.csv
    ├── test_api.py
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py
```

---

## Setup — Local Development

**Prerequisites:** conda, Docker Desktop

> **⚠️ Required before running: add the dataset**
>
> This project uses the [Kaggle Housing Prices dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
> You must download it and place it at the exact path below before running anything:
>
> ```
> data/raw/Housing.csv
> ```
>
> Quick setup:
> ```bash
> mkdir -p data/raw
> mv ~/Downloads/Housing.csv data/raw/Housing.csv
> ```
>
> The file must be named exactly `Housing.csv` (capital H). The pipeline will
> fail fast with a clear error if it is missing.

### 1. Clone the repository

```bash
git clone https://github.com/robsrot/G2_MLOPS_Project.git
cd G2_MLOPS_Project
```

### 2. Create `.env` with your secrets

```bash
# .env — never commit this file
WANDB_API_KEY="paste_your_40_character_key_here"
WANDB_ENTITY="paste_your_wandb_username_or_team_here"
MODEL_SOURCE="local"
WANDB_MODEL_ALIAS="prod"
```

### 3. Install the environment from the lockfile

```bash
conda-lock install -n housing_prices_mlops conda-lock.yml
```

### 4. Activate

```bash
conda activate housing_prices_mlops
```

### 5. Add the dataset (see callout above)

If you have not already done so, save `Housing.csv` to `data/raw/Housing.csv`:

```bash
mkdir -p data/raw
mv ~/Downloads/Housing.csv data/raw/Housing.csv
```

The filename is case-sensitive. The `.gitignore` intentionally excludes this
file from version control.

### 6. Run the training pipeline

```bash
python -m src.main
```

This produces `models/model.joblib`, `data/processed/clean.csv`, `reports/actual_vs_predicted.png`, `reports/residuals.png`, and `reports/predictions.csv`. Metrics and artifacts are logged to W&B if `WANDB_API_KEY` is set and `run.log_to_wandb: true` in `config.yaml`.

---

## Running the API Locally

**Native (with hot reload):**

```bash
MODEL_SOURCE=local uvicorn src.api:app --reload
```

**Docker:**

```bash
docker build -t housing-api:latest .
docker run -p 8000:8000 --env-file .env housing-api:latest
```

**Health check:**

```bash
curl http://127.0.0.1:8000/health
```

**Interactive docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Usage

| Environment | Base URL |
|---|---|
| Local | `http://127.0.0.1:8000` |
| Live (Render) | `https://g2-mlops-project.onrender.com` |

### GET /health

```bash
curl https://g2-mlops-project.onrender.com/health
```

```json
{"status": "ok", "model_version": "model.joblib"}
```

Returns `503` with `{"status": "unavailable", "model_version": "none"}` if the model has not loaded.

### POST /predict

```bash
curl -X POST https://g2-mlops-project.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
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
      "furnishingstatus": "furnished"
    }]
  }'
```

```json
{"predictions": [{"prediction": 8477932.149443712}]}
```

The endpoint accepts batches — include multiple objects in `records` to get multiple predictions in a single call. Extra fields return `422`. Missing fields return `422`.

---

## W&B Experiment Tracking

Project: [https://wandb.ai/charliesuarez10-ie/housing-price-mlops](https://wandb.ai/charliesuarez10-ie/housing-price-mlops)

Each training run logs:

- **Data:** raw row count, clean row count
- **Metrics:** CV RMSE, MAE, R², Adjusted R² (mean over 5 folds)
- **Artifacts:** processed dataset, trained model, predictions CSV
- **Plots:** actual vs. predicted, residuals panel
- **API inference logs:** buffered in `src/api.py` (batch size 50) and flushed as a W&B Table

To disable W&B (e.g. for local dev without credentials), set `run.log_to_wandb: false` in `config.yaml` or `WANDB_MODE=disabled` in the environment.

---

## CI/CD

### ci.yml — Quality Gate

Triggers on every pull request and push to `main`. Steps:

1. Checkout repository
2. Setup Miniconda
3. Install exact environment from `conda-lock.yml`
4. `pytest -q` — full test suite
5. `docker build` — validates the container builds without errors

W&B is fully disabled in CI (`WANDB_MODE=disabled`, `MODEL_SOURCE=local`). No secrets are required.

[View runs →](https://github.com/robsrot/G2_MLOPS_Project/actions)

### deploy.yml — Continuous Deployment

Triggers only when a human explicitly publishes a GitHub Release from `main`. Sends a deploy hook to Render, which pulls the latest image and restarts the service. The `RENDER_DEPLOY_HOOK_URL` secret is set in GitHub repository settings — it is never embedded in code.

---

## Testing

The test suite uses `pytest` and covers all `src/` modules. Tests are located in `tests/` with one file per source module.

**Run all tests:**

```bash
python -m pytest -v
```

**Run with coverage:**

```bash
python -m pytest --cov=src
```

**Current coverage:** 87% overall. Five modules are at 100% coverage: `clean_data`, `features`, `logger`, `utils`, and `__init__`.

**API tests:** `tests/test_api.py` is skipped automatically in local runs when no model artifact is present. In CI, `MODEL_SOURCE=local` and `WANDB_MODE=disabled` are set so tests never require external credentials. API tests that require a running server are skipped — all other tests pass.

---

## Model Card

| Field | Detail |
|---|---|
| **Model type** | Linear Regression — scikit-learn `Pipeline` with `ColumnTransformer` |
| **Training data** | Kaggle Housing Prices dataset — 545 observations × 13 columns, no missing values |
| **Target** | `price` — house sale price (range: 1.75M–13.3M, median: 4.34M) |
| **Features** | 12 inputs: `area` (log1p + StandardScaler), `bedrooms`, `bathrooms`, `stories`, `parking` (StandardScaler); `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` (binary 0/1); `furnishingstatus` (one-hot, drop-first) |
| **Preprocessing** | Binary encoding of yes/no columns → log1p on `area` → StandardScaler on numeric features → OneHotEncoder on `furnishingstatus`. All transforms fit on training data only (no leakage). |
| **Evaluation** | 5-fold cross-validation: R² 0.653, Adj R² 0.644, MAE 747,580, RMSE 1,039,102 |
| **Intended use** | First-pass automated valuation for residential listing agents. Always reviewed by a human before communicating to sellers. |
| **Limitations** | Trained on 545 rows from a single market. Performance may degrade on properties outside the training distribution (high-value outliers above ~10M, non-residential properties, geographies not represented in the data). Not suitable for financial instruments or legal valuations. |

---

## Changelog

### [1.0.1] — 2026-03-26

#### Fixed

- CI pipeline environment name corrected from `mlops` to `housing_prices_mlops`
- `PYTHONPATH` added to CI pytest step to resolve `ModuleNotFoundError` in GitHub Actions
- `conda-lock.yml` regenerated to include all Phase 2 dependencies
- Render URL updated to <https://g2-mlops-project.onrender.com>
- Inference CSV whitelisted in `.gitignore` for pipeline reproducibility

---

### [1.0.0] — 2026-03-22

#### Added

- Full MLOps upgrade from Phase 1 notebook-converted pipeline
- `config.yaml` expanded to 60+ keys across 10 sections
- `src/logger.py` with dual `StreamHandler` + `FileHandler` output
- W&B experiment tracking in `main.py` (metrics, artifacts, plots)
- `src/api.py` — FastAPI serving layer with `/health` and `/predict` endpoints
- Pydantic strict input contract (`extra="forbid"`)
- HTTP middleware with correlation IDs and latency logging
- Async W&B inference log buffer (batch size 50)
- `Dockerfile` + `.dockerignore` (allowlist strategy)
- `conda-lock.yml` for reproducible Linux-64 environment
- `.github/workflows/ci.yml` — quality gate on all PRs
- `.github/workflows/deploy.yml` — CD triggered by GitHub Release
- Render deployment at <https://g2-mlops-project.onrender.com>
- `tests/test_api.py` — 8 API tests including 422 contract enforcement
- `pytest.ini`, `src/__init__.py`, `tests/__init__.py`
- `data/inference/housing_inference.csv` committed for pipeline reproducibility

#### Removed

- `src/schema.py` removed — all constants moved to `config.yaml` for single source of truth

---

## Authors

**Group 2:** Tom Biefel, Kishan Dhulashia, Álvaro Perez La Rosa, Robyn Rothlin, Carlos Suarez Álvarez, Natalia Urrea

**Course:** MLOps — IE University MsC Business Analytics & Data Science, March 2026
