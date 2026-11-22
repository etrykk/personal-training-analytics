# V2 — Phase 4E: External Gym Churn Research

## Purpose

This is a **research-only**, cross-sectional classifier demonstration using a **separately sourced** gym membership dataset. It does not train a business-specific model, modify the Streamlit dashboard, or enable churn probabilities for clients.

The Phase 4B–4D membership-cancellation classifier specifically targets cancellation in the **30 days after a historical snapshot**; it remains blocked for the existing business dataset because there are too few independent cancellation events. This Phase 4E external experiment has a **different, cross-sectional target** (`Churn`) and must never be presented as proof of 30-day forecasting performance.

## Dataset

Original source: https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn

Download the original CSV yourself and place it at:

`data/external/gym_churn_us.csv`

The files in this package **do not include** the external CSV. Its provenance, licensing, and temporal validity must be independently reviewed before public redistribution or operational reuse. The dataset is often described as a dataset for a *fictional gym*; do not assume the records are real gym customers.

Required CSV fields:

| Field | Use |
|---|---|
| `Churn` | Cross-sectional binary label (0/1) |
| `Near_Location` | Exploratory feature |
| `Partner` | Exploratory feature |
| `Promo_friends` | Exploratory feature |
| `Phone` | Exploratory feature |
| `Contract_period` | Exploratory feature |
| `Group_visits` | Exploratory feature |

`Age`, `Gender`, `Lifetime`, `Month_to_end_contract`, and retrospective class-frequency columns are **not** used. Some of these may be post-outcome features or raise fairness questions. The six permitted fields also lack verified timestamps: their inclusion is **not** proof that they were observed before churn.

## Implementation

- `src/ml/external_churn_experiment.py` has a fixed, documented feature allowlist.
- It validates the target and feature types before training.
- If available event counts or holdout counts are insufficient, it returns a blocked decision, without model metrics.
- It uses a reproducible **stratified random holdout** (75% train / 25% test) because verified event dates are absent. This is *not* chronological validation.
- Fitting and imputing/scaling occur only on the training split via scikit-learn pipelines.
- It compares Logistic Regression with a majority-class baseline, reports accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, average precision, confusion-matrix counts, and 95% Wilson intervals for recall/precision.
- It does **not** save fitted models, row-level predictions, or client identifiers.
- The decision always has `business_predictions_enabled = False` and explicitly notes the limits of external dataset results.

## Run in Windows Anaconda Prompt

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_external_churn_experiment.py -v
python -m pytest tests/ -v
python -m src.ml.external_churn_experiment
```

The last command creates:

- `outputs/predictive/external_experiment/external_churn_checks.csv`
- `outputs/predictive/external_experiment/external_churn_metrics.csv`
- `outputs/predictive/external_experiment/external_churn_decision.csv`

If the source file is missing, these reports will show `external_dataset_missing` and contain **no model metrics**. That is expected; do not fabricate them.

## Interpretation and next requirements

A high external holdout score is not equivalent to good business-specific churn prediction. The external dataset lacks a verified prediction date, a defined 30-day horizon, and confirmed pre-outcome measurements. An independent temporal validation, sufficient cancellation events per business, probability calibration, and out-of-sample performance monitoring would be required before displaying customer-level prediction probabilities in the product.

External records must **never be merged with** business-specific client data for per-business model training without revisiting the user's isolation choice. For the current project, they remain a separate research artifact.

## GitHub note

Do not upload `data/external/gym_churn_us.csv`, `outputs/`, real client data, or any hidden credentials. Git ignores CSVs in `data/external/` through the provided nested `.gitignore`, but **manual GitHub uploads bypass the Git ignore mechanism**—check the selected files yourself.
