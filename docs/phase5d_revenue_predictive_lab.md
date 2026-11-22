# V2 Phase 5D — Historical Revenue Forecasting in Predictive Lab

## Purpose
Display the Phase 5C revenue forecasting **retrospective backtest** alongside the Phase 4F gym churn study. The business intelligence pages and V1 CSV import are unchanged. Individualized churn predictions and live revenue forecasting remain disabled.

## Files

- `app/dashboard_v2.py` — replaces the current V2 dashboard; keeps all existing navigation, CSV upload, trainer name editor, and churn research.
- `src/ml/revenue_lab_reports.py` — validates five Phase 5C report CSV files and produces **historical aggregate data only**.
- `tests/test_revenue_lab_reports.py` — self-contained tests with synthetic historical fixtures.

## Required inputs (generated locally)

The reader looks for:

```
outputs/predictive/revenue_evaluation/
├── forecast_backtest_decision.csv
├── forecast_backtest_checks.csv
├── forecast_validation_scores.csv
├── forecast_holdout_scores.csv
└── forecast_backtest_predictions.csv
```

Run `python -m src.ml.revenue_forecast_evaluation` from your project root if these files do not exist. The UI shows an explanation if files are missing or inconsistent. No model fitting occurs in Streamlit.

## What the page shows

- Historical development-validation winner (chosen *before* reserved holdout evaluation)
- Its historical development MAE, reserved holdout MAE and RMSE
- Aggregate per-model error table for all or a chosen 1-, 2-, or 3-month horizon
- Plot of six **historical** one-month-ahead predictions vs recorded net cash on the reserved May–October 2026 holdout
- Research-only caveats about small sample, overlapping horizons, and uncertain future accuracy

The readout is independent of `Sample Data`/`Imported Data` selectors. It represents the fixed sample-data research exercise, not imported businesses. Avoid presenting it as a model deployed to an imported business.

## Assessment from actual Phase 5C reports

- All five Phase 5C checks passed.
- Selected on development: `last_month` (development MAE $691.39).
- Selected model on holdout: MAE $2,201.67; RMSE $2,744.47.
- Lowest *observed* holdout error: six-month linear trend, MAE $2,024.47, RMSE $2,504.56. This result does not justify changing model selection after test evaluation.
- Live forecasting remains disabled.

## Install and run — Anaconda Prompt

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_revenue_lab_reports.py -v
python -m pytest tests/ -v
python -m src.ml.revenue_forecast_evaluation
python -m streamlit run app/dashboard_v2.py
```

**Verification:** 355 existing + new tests passed in the verification copy; one existing pandas date-parsing warning. Live UI visual layout remains for user browser review.

## Safety and GitHub

Do not upload real clients, sessions, or payments to the public repository. Do not put individual-client predictions in this research dashboard. Store data/report exports according to privacy and licensing constraints.
