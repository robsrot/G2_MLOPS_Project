# [Project Name: Housing Prices]

**Author:** Robyn
**Course:** MLOps: Master in Business Analytics and Data Sciense
**Status:** Session 1 (Initialization)

---

## 1. Business Objective
*Replace this section with your project definition.*

* **The Goal:** What business value does this model create?
  > *Example: Reduce food waste by 10% by predicting daily bakery demand.*

* **The User:** Who consumes the output and how?
  > *Example: Store managers receive a weekly PDF report on Monday mornings.*

---

## 2. Success Metrics
*How do we know if the project is successful?*

* **Business KPI (The "Why"):**
  > *Example: Reduce unsold inventory costs by $5,000/month.*

* **Technical Metric (The "How"):**
  > *Example: Model MAPE (Mean Absolute Percentage Error) < 15% on the test set.*

* **Acceptance Criteria:**
  > *Example: The model must outperform the current "moving average" baseline.*

---

## 3. The Data

* **Source:** (e.g., Company Database, Kaggle CSV, API).
* **Target Variable:** What specifically are you predicting/ classifying?
* **Sensitive Info:** Are there emails, credit cards, or any PII (Personally Identifiable Information)?
  > *⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.*

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python src/main.py`



