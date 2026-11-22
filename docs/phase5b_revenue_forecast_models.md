# V2 Phase 5B — Experimental Revenue Forecast Models

## Purpose

Create simple, reproducible forecasts of **monthly completed net cash collected** (successful payments minus completed refunds) using eligible *complete calendar months* from Phase 5A. This stage **does not** validate accuracy and must **not** enable live forecasts in Predictive Lab.

## Historical split (November 20, 2026 snapshot)

- **27 development months:** February 2024–April 2026
- **6 months held out:** May 2026–October 2026
- **May–July 2026 candidate forecasts:** examples generated as if the model were operating at the April 2026 cutoff. These are retrospective historical candidates, **not** November 2026 forecasts and **not** advice for a current business.
- January 2024 is excluded for uncertain first-month coverage; November 2026 is an incomplete month and excluded.

## Three candidate methods

| Name in output | Calculation | Rationale |
| --- | --- | --- |
| `last_month` | Repeat the final eligible training month's cash total | Transparent baseline |
| `trailing_3_month_mean` | Repeat the mean of the most recent three training months | Smooths short-term variation |
| `linear_trend_last_6_months` | Fit a straight line to six training months and extrapolate up to three months ahead | Tests whether a local trend helps |

Forecasts are **not** clipped to zero: negative net cash can occur when refunds exceed payments. A straight-line trend can extrapolate poorly and is not necessarily better than a simple baseline.

## Guardrails

1. Read Phase 5A's `forecast_history.csv` and `forecast_readiness_decision.csv`; don't rebuild financial definitions.
2. Reject missing, duplicate, or out-of-order months; non-finite cash; invalid eligibility flags; insufficient or nonconsecutive eligible history; and mismatched readiness metadata.
3. Require at least 18 development months and reserve the final six eligible months.
4. **Never use held-out net cash totals** for development-stage predictions, feature selection, or fitting.
5. Report all candidates as experimental and keep `approved_for_live_forecasting=False`.
6. Use a consistent train cutoff and identical holdout horizons for each model. No model has been evaluated in this phase.

## Installation

Extract the ZIP into the project root, preserving the folder structure:

- `src/ml/revenue_forecast_models.py` (new)
- `tests/test_revenue_forecast_models.py` (new)
- `docs/phase5b_revenue_forecast_models.md` (this file)

## Commands (Anaconda Prompt / Windows Command Prompt)

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_revenue_forecast_models.py -v
python -m pytest tests/ -v
python -m src.ml.revenue_forecast_models
```

The last command requires the Phase 5A files in `outputs/predictive/revenue_readiness/`. If they're missing, rerun:

```cmd
python -m src.ml.revenue_forecast_readiness --as-of 2026-11-20
```

## Generated reports

Saved under `outputs/predictive/revenue_models/`:

- `forecast_model_candidates.csv` — nine 1–3 month forecast candidates (3 models × 3 horizons)
- `forecast_model_checks.csv` — split and input validations
- `forecast_model_decision.csv` — cutoff, holdout metadata, and non-deployment status

## Next: Phase 5C

Develop **rolling-origin backtesting**, compare 1–3 month-ahead errors using MAE/RMSE, and report uncertainty where data support it. Use development history for selection; preserve the six-month out-of-time block for a final comparison, recognizing that six months is still a small sample. Never interpret a good backtest as proof of future commercial performance.
