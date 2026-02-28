# Housing Prices Prediction

**Author:** Group 2: Tom Biefel, Kishan Dhulashia, Álvaro Perez La Rosa, Robyn Rothlin, Carlos Suarez Álvarez, and Natalia Urrea 
**Course:** MLOps: Master in Business Analytics and Data Sciense
**Status:** Session 3 - Modularization & Production Readiness

---

## 1. Business Objective
Real estate agencies price new listings inconsistently. Without a formal appraisal—which takes days and costs money—agents rely on intuition. The same property gets different estimates depending on who handles it. Overpriced listings sit; underpriced ones close fast but leave revenue on the table.

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
  | R² | ≥ 0.65 | 0.661 |
  | Adjusted R² | ≥ 0.64 | 0.653 |
  |MAE | ≤ 868,000 (≤ 20% of median)| ~768,000 |
  |CV stability | No single fold deviates > 5% R² from the mean | Confirmed across 5 folds |
  |Feature validity | All 12 features contribute meaningfully| Confirmed — no zero-weight features |


* **Deployment Codition:**
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
  > *⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.*


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
├── configs/config.yaml                # All pipeline parameters
├── .env.example                       # Secrets template
├── .gitignore
│
├── notebooks/
│   └── HousingPricesPrediction.ipynb  # Exploratory analysis (sandbox, read-only)
│
├── src/                               # Production code
│   ├── __init__.py
│   ├── main.py                        # Pipeline orchestrator (entry point)
│   ├── load_data.py                   # Data ingestion (local CSV or Kaggle)
│   ├── validate.py                    # Schema, dtype, and value checks
│   ├── preprocess.py                  # Encoding, log transforms, scaling
│   ├── train.py                       # K-Fold CV training + artifact saving
│   ├── evaluate.py                    # Metrics + diagnostic plots
│   └── infer.py                       # Inference on new data
│
├── data/                              # IGNORED by Git
│   ├── raw/                           # Source data (Housing.csv)
│   └── processed/                     # Train/test splits
│
├── models/                            # IGNORED by Git
│   ├── model_5_kfold.pkl
│   ├── scaler.pkl
│   └── ohe_encoder.pkl
│
├── reports/                           # Generated outputs
│   ├── pipeline.log
│   ├── model_5_kfold_metrics.txt
│   ├── model_5_kfold_actual_vs_predicted.png
│   └── model_5_kfold_residuals.png
│
└── tests/
    ├── conftest.py
    ├── test_load_data.py
    ├── test_validate.py
    ├── test_preprocess.py
    ├── test_train.py
    └── test_evaluate.py

```

## 5. Execution Model

Five models were developed and compared. Model 5 (K-Fold CV) was selected as the production model.


| Model                     | Approach                                      | R²     | Adj. R² |
|----------------------------|----------------------------------------------|--------|---------|
| 1 — Baseline               | Binary encoding + OHE + StandardScaler      | ~0.63  | ~0.62   |
| 2 — Log Price              | + log(price) + log(area)                    | ~0.65  | ~0.64   |
| 3 — Log + Lasso            | + LassoCV feature selection                 | ~0.64  | ~0.63   |
| 4 — Log + Outlier Removal  | + IQR outlier removal on training set       | ~0.65  | ~0.64   |
| 5 — K-Fold CV              | Model 2 preprocessing + 5-fold cross-val    | 0.661  | 0.653   |

> Model 5 was selected because K-Fold CV ensures performance is not an artifact of a single favorable train/test split - the result holds across all 5 data partitions.

**Preprocessing Pipeline (applied per fold)**
  1. Binary encoding: yes → 1, no → 0
  2. One-hot encoding of furnishingstatus (drop first)
  3. Log1p on area (right-skewed)
  4. Log1p on price (right-skewed target)
  5. StandardScaler on numeric features (fit on train fold only - no leakage)

## 6. Setup

  ### 1. Clone
    git clone https://github.com/<YOUR_GITHUB_USER>/housing_prices.git
    cd housing_prices

  ### 2. Create environment
    conda env create -f environment.yml
    conda activate housing_prices_mlops

  ### 3. Add dataset (or let the pipeline download from Kaggle on first run)
    mkdir -p data/raw
    cp /path/to/Housing.csv data/raw/Housing.csv


## 7. Running the Pipeline

 Full pipeline: load → validate → train → evaluate → save
  
    python -m src.main

Data stage only (load + validate)

    python -m src.main --pipeline data

Custom config

    python -m src.main --config configs/config.yaml

Output after a full run:
> models/model_5_kfold.pkl\
models/scaler.pkl
models/ohe_encoder.pkl
reports/model_5_kfold_metrics.txt
reports/model_5_kfold_actual_vs_predicted.png
reports/model_5_kfold_residuals.png
reports/pipeline.log


## 8. Coding Standards
- PEP 8 - enforced via flake8
- Type hints on all functions
- Docstrings on all modules and functions
- No hardcoded values - all params from config.yaml
- No silent failures - all errors caught, logged, re-raised
- No data leakage - scalers/encoders fit on train only

  > flake8 src/ tests/   # Lint check

  > black src/ tests/    # Auto-format

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
  | 15 | Final exam  |


## References 

- [Kaggle — Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

- [2026 IE MLOps Course Reference Project](https://github.com/2026-IE-MLOps-Course/mlops_project)

