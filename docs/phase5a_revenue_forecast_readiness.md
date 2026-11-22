# V2 Phase 5A — Revenue Forecasting Data Readiness

## Goal

Determine whether the processed business payments provide enough **complete calendar months of completed net cash collections** to justify experimenting with monthly forecasts. This is a *readiness assessment*, not a forecast, and has no effect on churn prediction or the main dashboard.

## Definitions and safeguards

- Reuses `monthly_cash_performance()` from `src/analytics/financial_membership_insights.py` to prevent revenue-accounting discrepancies.
- Target: **completed payments minus completed refunds by transaction month**. This is net cash collected, not profit, invoices, or accrued revenue.
- Excludes pending and failed attempts from collected cash.
- Excludes the reporting month when it is incomplete. A potentially truncated first observed month is also excluded conservatively.
- Preserves zero-cash months, but flags them for review; absence of recorded payments does not prove complete historical coverage.
- Blocks backtesting when the source appears stale instead of generating artificial zero-revenue months after the final transaction.
- Defaults to requiring **18 months for model development plus 6 months reserved for out-of-time evaluation**. These are initial engineering readiness rules, **not a guarantee of forecast quality**.
- Negative net-cash months can be real when refunds exceed completed payments; they are flagged for review but not silently clipped.
- Never trains a model, never produces future projections, and never enables live forecasting during Phase 5A.

## Run

From Anaconda Prompt:

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_revenue_forecast_readiness.py -v
python -m pytest tests/ -v
python -m src.ml.revenue_forecast_readiness --as-of 2026-11-20
```

Reports are written to `outputs/predictive/revenue_readiness/`:

1. `forecast_history.csv` — historical monthly cash, partial-month flags, eligibility and exclusion reasons
2. `forecast_readiness_checks.csv` — data sufficiency and coverage diagnostics
3. `forecast_readiness_decision.csv` — whether *experimental backtesting* can start (never a deployment decision)

## Next phase

Phase 5B will compare straightforward monthly forecasting baselines. Phase 5C will use strictly forward-in-time rolling-origin validation with MAE/RMSE and include uncertainty/error intervals. No model will be shown as reliable based solely on Phase 5A readiness.
