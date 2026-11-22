# V2 Phase 4B — Historical Churn Labels and Features

## Purpose

This phase prepares **experimental historical observations** for later evaluation. It does not train a model, create churn probabilities, or change any existing analytics dashboard or V1 files.

**Target:** A recorded **membership cancellation** within the 30 days following a historical prediction date. Membership expiration, package nonrenewal, and a lack of session attendance are not cancellation events.

## Methods

- Construct calendar-month-start snapshots for memberships active at that snapshot.
- Include a snapshot only if its complete 30-day future observation window falls on or before the selected `--as-of` date.
- Label `1` only if the actual cancellation date falls **after** the snapshot and **on or before** the horizon end.
- Label `0` only when the membership is known to have continued without cancellation throughout that period.
- Censor (leave the label missing) when a membership expires during the horizon, event dates are missing, or the data is ambiguous.
- Build session features strictly using sessions **before** the snapshot date. Sessions on that date and later are excluded.
- Do not feed current `status`, `cancellation_date`, `cancellation_reason`, or `end_date` into the model as predictors.
- Current plan, fee and trainer assignments are also omitted because this dataset does not contain their historical changes.

The standardized feature list is `MODEL_FEATURE_COLUMNS` in `src/ml/historical_churn.py`.

## How to install and test

Place `historical_churn.py` in `src/ml/`, `test_historical_churn.py` in `tests/`, and this document in `docs/`.

Open **Anaconda Prompt** and run:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_historical_churn.py -v
python -m pytest tests/ -v
python -m src.ml.historical_churn --as-of 2026-11-20
```

Reports are written to `outputs/predictive/historical/`:

- `historical_churn_snapshots.csv` — membership-level historical examples and features; contains client and membership IDs.
- `historical_churn_summary.csv` — aggregate counts, including number of positive, negative, and censored snapshots.

## Interpret carefully

- A repeated snapshot from the same membership is **not an independent churn event**. Use `positive_unique_memberships`, not positive snapshot rows, to judge the number of distinct observed positive outcomes.
- Historical observations of the same person can be correlated. Do not randomly split snapshot rows into training and testing groups. Use a time-aware evaluation design with precautions for overlapping clients and prediction horizons.
- The sample data supplied with V1 is **synthetic**, so it cannot prove real-world model accuracy.
- The baseline and any machine learning model must be evaluated on unseen, representative data. In particular, a classifier that predicts no churn for everyone can show high accuracy but have 0% recall.
- The output sets `training_approved=False` intentionally: producing historical labels is *not* sufficient approval to train or deploy a business-specific model.
- With real customer data, treat the snapshots report as sensitive operational data. Do not upload it to a public GitHub repository.
