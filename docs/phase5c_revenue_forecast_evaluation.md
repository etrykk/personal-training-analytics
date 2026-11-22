# V2 Phase 5C — Experimental Revenue Forecast Evaluation

## Purpose

Compare three simple candidate models for **monthly completed net cash collected** (completed payments minus completed refunds) while preserving the distinction between model selection and a reserved historical test period. Forecast errors are reported in dollars. The project dataset is sample/example business data.

## Inputs (from existing phases)

- `outputs/predictive/revenue_readiness/forecast_history.csv`
- `outputs/predictive/revenue_readiness/forecast_readiness_decision.csv`
- `outputs/predictive/revenue_models/forecast_model_candidates.csv`
- `outputs/predictive/revenue_models/forecast_model_decision.csv`

The program validates dates, eligibility, historical split, and Phase 5B model outputs. It refuses mismatched or tampered candidates rather than silently recomputing unrelated results.

## Historical setup

- **Complete eligible months:** February 2024–October 2026 (33 months).
- **Development:** February 2024–April 2026 (27 months).
- **Internal model selection:** the final six development months; validation forecasts use preceding development months only.
- **Reserved holdout:** May–October 2026 (six months).
- **Horizons:** one, two, and three months ahead.

### Selection and evaluation

1. Simulate rolling forecast origins for each target month in the development-validation window. Compute MAE/RMSE for each of the three models.
2. Select the model with the lowest **development-validation MAE**, with RMSE as tie-breaker. **Do not select based on held-out outcomes.**
3. Backtest **all three models** on May–October 2026. For every holdout forecast, only months before that forecast's target are used. Earlier *observed* holdout months may inform *later* forecasts in this walk-forward simulation.
4. Present holdout results and the selected model without reselecting anything based on those results.

The reserved holdout contains six distinct target months. The 1–3-month forecasts for the same targets overlap: 18 forecast errors per model **are not 18 independent monthly observations**. This short period is not enough to certify performance or determine stable prediction intervals.

## Models

| Model identifier | Meaning |
| --- | --- |
| `last_month` | Repeat the last observed completed month's cash |
| `trailing_3_month_mean` | Mean of the last three observed months |
| `linear_trend_last_6_months` | Straight-line extrapolation from the previous six months |

## Reports

Saved under `outputs/predictive/revenue_evaluation/`:

- `forecast_backtest_predictions.csv` — 108 historical one-, two-, and three-month forecasts, separated into development validation and held-out evaluation; includes target dates, values, errors, and model names.
- `forecast_validation_scores.csv` — selection-only scores for the three candidates.
- `forecast_holdout_scores.csv` — out-of-time MAE, RMSE, and mean error by model and horizon.
- `forecast_backtest_checks.csv` — consistency diagnostics.
- `forecast_backtest_decision.csv` — development-selected model and research-only status. `approved_for_live_forecasting` remains `False`.

**Do not publish data from real businesses or identifiable payment records.** Aggregated reports using synthetic sample data may be used in the portfolio with a suitable disclosure.

## Local commands (Windows Anaconda Prompt)

```cmd
cd /d "C:\Users\Ethan\Downloads\personal-training-analytics"
python -m pytest tests/test_revenue_forecast_evaluation.py -v
python -m pytest tests/ -v
python -m src.ml.revenue_forecast_evaluation
```

If input files are missing, run Phases 5A and 5B first:

```cmd
python -m src.ml.revenue_forecast_readiness --as-of 2026-11-20
python -m src.ml.revenue_forecast_models
```

## Observations from the sample dataset (2026-11-20 reporting date)

- Development selects the last-month baseline (development-validation MAE **$691.39**).
- On the reserved holdout, the development-selected baseline has MAE **$2,201.67** and RMSE **$2,744.47**.
- The linear trend has lower holdout MAE (**$2,024.47**) but is **not retroactively chosen**. This discrepancy cautions against trusting results from a small backtest.
- These numbers are **not a guarantee of future revenue forecasts**, and this phase never enables live prediction.

## Next step

Phase 5D can display only labeled research/backtest scores in Predictive Lab until forecasting quality is validated with substantially more independent history. Predicted net cash should not be mixed with collected cash or described as reliable operational guidance.
