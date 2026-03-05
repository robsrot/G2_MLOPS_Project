# Housing Prices Prediction

**Author:** Group 2: Tom Biefel, Kishan Dhulashia, Álvaro Perez La Rosa, Robyn Rothlin, Carlos Suarez Álvarez, and Natalia Urrea 

**Course:** MLOps: Master in Business Analytics and Data Science

**Status:** Session 3 - Modularization & Production Readiness

---

## 1. Business Objective
Real estate agencies price new listings inconsistently. Without a formal appraisal agents rely on intuition. The same property gets different estimates depending on who handles it. Overpriced listings sit; underpriced ones close fast but leave revenue on the table.

* **The Goal:** 
  > *An automated first-pass valuation tool that generates an instant price estimate the moment a new listing is registered. Agents input 12 attributes they already collect at intake (size, layout, amenities, location indicators) and receive a data-driven reference price before any further assessment is needed.*

* **The User:** 
  > *Listing agents at a residential real estate agency. Every agent, regardless of experience, starts from the same model-generated estimate, enforcing pricing consistency across the agency.*

  **Secondary User:**
  > Sellers: Transparent, attribute-based explanation of their estimated price
  
  > Agency management: Auditable pricing decisions across all agents
  
  > Financial institutions: Independent cross-check on declared property values for collateral assessment


---

## 2. Success Metrics
*How do we know if the project is successful?*

* **Business KPI:**
  > **Pricing turnaround**: Estimate available at listing creation - 0 wait time

  > **Agent pricing consistency**: All agents use the same model output as their starting point

  > **Estimation accuracy**: Predicted price within ±20% of actual sale price at the median

  > **Coverage**: Valid estimate for 100% of listings with all 12 attributes complete

    *The ±20% tolerance is grounded in the price range (1.75M–13.3M). At the median of 4.34M, ±20% = ±868,000. The achieved MAE of ~768,000 falls within this band, making predictions commercially useful as a first-pass anchor.*

* **Technical Acceptance Criteria:**

  | Criterion | Threshold | Result
  | --- | --- | --- |
  | R² | ≥ 0.65 | 0.653 |
  | Adjusted R² | ≥ 0.64 | 0.644 |
  |MAE | ≤ 868,000 (≤ 20% of median)| 747,580 |
  |RMSE | — | 1,039,102 |
  |CV stability | No single fold deviates > 5% R² from the mean | Confirmed across 5 folds |
  |Feature validity | All 12 features contribute meaningfully| Confirmed — no zero-weight features |


* **Deployment Condition:**
  > *The model output is always reviewed by an agent before being communicated to a seller. It is never surfaced as a final price.*

---

## 3. The Data
Source: Kaggle - yasserh/housing-prices-dataset
File: Housing.csv - 545 observations × 13 columns, no missing values
Target: price - house sale price, range 1.75M–13.3M, median 4.34M
PII/ Sensitive Information: None
* **Source:** Kaggle - yasserh/housing-prices-dataset.
* **Target Variable:** price - house sale price, range 1.75M–13.3M, median 4.34M
* **Sensitive Info:** None


| Feature      | Type          |Description        |
|--------------------|---------------------|----------------------------------------------------------|
| price              | Numeric (Target)    | Sale price                                               |
| area               | Numeric             | Plot area (sq ft), log-transformed due to right skew    |
| bedrooms           | Numeric             | Number of bedrooms                                       |
| bathrooms          | Numeric             | Number of bathrooms                                      |
| stories            | Numeric             | Number of floors                                         |
| parking            | Numeric             | Number of parking spots                                  |
| mainroad           | Binary (Yes/No)     | Property faces a main road                               |
| guestroom          | Binary (Yes/No)     | Property includes a guest room                           |
| basement           | Binary (Yes/No)     | Property includes a basement                             |
| hotwaterheating    | Binary (Yes/No)     | Property has hot water heating                           |
| airconditioning    | Binary (Yes/No)     | Property has air conditioning                            |
| prefarea           | Binary (Yes/No)     | Property is located in a preferred area                  |
| furnishingstatus   | Categorical         | Furnished / Semi-furnished / Unfurnished

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
housing_prices/
├── README.md                          # Project definition and guide
├── environment.yml                    # Reproducible Conda environment
├── config.yaml  
│
├── notebooks/
│   └── HousingPricesPrediction.ipynb  # Exploratory analysis (sandbox, read-only)
│
├── src/                               # Production code
│   ├── __init__.py
│   ├── main.py                        # Pipeline orchestrator (entry point)
│   ├── load_data.py                   # Data ingestion (local CSV or Kaggle)
│   ├── clean_data.py                  # Data cleaning and deterministic transforms
│   ├── schema.py                      # Schema definitions and column contracts
│   ├── validate.py                    # Schema, dtype, and value checks
│   ├── features.py                    # Model-side preprocessing (scaling + encoding)
│   ├── train.py                       # K-Fold CV training + artifact saving
│   ├── evaluate.py                    # Metrics + diagnostic plots
│   ├── infer.py                       # Inference on new data
│   └── utils.py                       # Utility functions (save_csv, save_model)
│
├── data/
│   ├── raw/                           # Source data (Housing.csv)
│   ├── processed/                     # Cleaned training input (clean.csv)
│   └── inference/                     # Prediction outputs (predictions_<timestamp>.csv)
│
├── models/
│   └── model_<timestamp>.joblib
│
├── reports/                           # Generated outputs
│   ├── actual_vs_predicted.png
│   └── residuals.png
│
└── tests/
    ├── __init__.py
    ├── test_clean_data.py
    ├── test_evaluate.py
    ├── test_features.py
    ├── test_infer.py
    ├── test_load_data.py
    ├── test_main.py
    ├── test_schema.py
    ├── test_train.py
    ├── test_utils.py
    └── test_validate.py

```

## 5. Execution Model

Five models were developed and compared. Model 5 (K-Fold CV) was selected as the production model.


| Model                     | Approach                                      | R²     | Adj. R² |
|----------------------------|----------------------------------------------|--------|---------|
| 1 — Baseline               | Binary encoding + OHE + StandardScaler      | ~0.63  | ~0.62   |
| 2 — Log Price              | + log(price) + log(area)                    | ~0.65  | ~0.64   |
| 3 — Log + Lasso            | + LassoCV feature selection                 | ~0.64  | ~0.63   |
| 4 — Log + Outlier Removal  | + IQR outlier removal on training set       | ~0.65  | ~0.64   |
| 5 — K-Fold CV              | Model 2 preprocessing + 5-fold cross-val    | 0.653  | 0.644   |

> Model 5 was selected because K-Fold CV ensures performance is not an artifact of a single favorable train/test split - the result holds across all 5 data partitions.

**Data Split Strategy**
- **Inference holdout:** 50 rows held out purely for final predictions (smoke test)
- **Training set:** All remaining data (~495 rows), evaluated using 5-fold cross-validation
- **Rationale:** K-fold CV eliminates the need for a separate test set by providing robust performance estimates across all data partitions. This maximizes training data while maintaining rigorous evaluation.

**Preprocessing Pipeline (applied per fold)**
  1. Binary encoding: yes → 1, no → 0
  2. One-hot encoding of furnishingstatus (drop first)
  3. Log1p on area (right-skewed)
  4. Log1p on price (right-skewed target)
  5. StandardScaler on numeric features (fit on train fold only - no leakage)

## 6. Setup

  ### 1. Clone
    git clone https://github.com/robsrot/housing_prices.git
    cd housing_prices

  ### 2. Create environment
    conda env create -f environment.yml
    conda activate housing_prices_mlops

  ### 3. Add dataset (or let the pipeline download from Kaggle on first run)
    Place `Housing.csv` in `data/raw/Housing.csv`


## 7. Running the Pipeline

 Full pipeline: load → validate → train → evaluate → save
  
    python -m src.main

Output after a full run:
> data/processed/clean.csv
models/model_<timestamp>.joblib
reports/actual_vs_predicted.png
reports/residuals.png
data/inference/predictions_<timestamp>.csv

## 7.1 Module Contracts (Quick Reference)

- `src/load_data.py`: Loads only from disk/Kaggle helper and fails fast for invalid paths and malformed files.
- `src/clean_data.py`: Applies deterministic cleaning and encoding, then returns a clean DataFrame.
- `src/schema.py`: Defines required columns and data contracts for validation.
- `src/validate.py`: Enforces strict schema contract (required columns present, no unexpected columns, no invalid values).
- `src/features.py`: Returns an unfitted preprocessing blueprint (ColumnTransformer).
- `src/train.py`: Trains with 5-fold CV and returns a fitted pipeline plus CV metrics payload.
- `src/evaluate.py`: Validates CV payload contracts before metric reporting and plot generation.
- `src/infer.py`: Requires a callable `predict` pipeline and non-empty DataFrame input.
- `src/utils.py`: Utility functions for saving CSV files and joblib models.


## 8. Coding Standards
- PEP 8 - enforced via flake8
- Type hints on all functions
- Docstrings on all modules and functions
- Parameters and paths are centralized in src/main.py
- No silent failures - key modules use fail-fast exceptions with explicit messages
- No data leakage - scalers/encoders fit on train only

  > flake8 src/ tests/        # Lint check

  > black src/ tests/         # Auto-format

  > python -m pytest -q       # Run tests

  > coverage run -m pytest -q
  > coverage report -m        # Coverage report

## Risks & Limitations 

- Small dataset (545 rows): K-Fold CV reduces variance; retrain as more data accumulates
- Single market / geography:
Do not generalize without retraining on local data
- No condition or location features:
Residual 34% variance unexplained; model complements, not replaces, agent judgment
- High-value outliers:
Log transformation applied; predictions less reliable above ~10M
- Data drift over time:
Retrain periodically; monitor MAE against actual sale prices in production

## Roadmap

  | Session        | Milestone                                                                 |
  |---------------|---------------------------------------------------------------------------|
  | 8 | Group Work 1 submission — business case + production-ready code |
  | 9–11 | Hydra config management, MLflow experiment tracking, W&B integration |
  | 12–14 | CI/CD via GitHub Actions; model serving via FastAPI |


## References 

- [Kaggle — Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)