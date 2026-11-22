#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5C - Experimental Revenue Forecast Backtesting

"""Evaluate historical monthly cash forecasts without selecting on the test set.

Selection uses rolling-origin, 1-3-month evaluations restricted to the
*development* history. Once the winning model is selected, all three methods
are evaluated on the six reserved months using genuine walk-forward origins.
Previously observed test months may inform later forecasts, never earlier ones.
The resulting reports are historical research, not live business forecasts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.revenue_forecast_models import (
    MODELS, _candidate_values, _month_periods, _true_false,
    _validate_and_split,
)

PREDICTION_COLUMNS = [
    'split', 'model', 'origin_month', 'target_month', 'horizon_months',
    'actual_net_cash', 'predicted_net_cash', 'error', 'absolute_error',
    'selected_on_development', 'experimental_only',
]
SCORE_COLUMNS = [
    'model', 'horizon_months', 'target_months', 'predictions',
    'mae', 'rmse', 'mean_error', 'selected_on_development',
]
CHECK_COLUMNS = ['check', 'observed', 'requirement', 'passed', 'detail']
DECISION_COLUMNS = [
    'status', 'selected_model', 'selection_method', 'training_end_month',
    'holdout_start_month', 'holdout_end_month', 'holdout_months',
    'forecast_horizons', 'development_validation_mae', 'holdout_mae',
    'holdout_rmse', 'holdout_last_month_mae', 'beats_last_month_on_holdout',
    'research_results_available', 'approved_for_live_forecasting', 'note',
]


def _validate_candidate_reports(candidates, candidate_decision, training,
                                holdout_periods, max_horizon):
    """Ensure Phase 5B forecasts are complete, untouched, and aligned."""
    if not isinstance(candidates, pd.DataFrame) or not isinstance(candidate_decision, pd.DataFrame):
        raise TypeError('Phase 5B candidates and decision must be DataFrames')
    if len(candidate_decision) != 1:
        raise ValueError('Phase 5B decision must contain one row')
    required_decision = {
        'status', 'training_start_month', 'training_end_month', 'training_months',
        'holdout_start_month', 'holdout_end_month', 'reserved_holdout_months',
        'forecast_horizons', 'models_compared', 'approved_for_live_forecasting',
    }
    if not required_decision.issubset(candidate_decision):
        raise ValueError('Phase 5B decision is missing required fields')
    row = candidate_decision.iloc[0]
    expected_start = str(training['month_period'].iloc[0])
    expected_end = str(training['month_period'].iloc[-1])
    expected_hold_start, expected_hold_end = str(holdout_periods[0]), str(holdout_periods[-1])
    for key, expected in (
        ('training_start_month', expected_start),
        ('training_end_month', expected_end),
        ('holdout_start_month', expected_hold_start),
        ('holdout_end_month', expected_hold_end),
    ):
        if str(row[key]) != expected:
            raise ValueError(f'Phase 5B {key} disagrees with Phase 5A history')
    for key, expected in (
        ('training_months', len(training)),
        ('reserved_holdout_months', len(holdout_periods)),
        ('forecast_horizons', max_horizon),
    ):
        value = pd.to_numeric(row[key], errors='coerce')
        if pd.isna(value) or float(value) != expected:
            raise ValueError(f'Phase 5B {key} disagrees with configured split')
    if row['status'] != 'candidates_ready_for_out_of_time_evaluation':
        raise ValueError('Phase 5B has not approved historical candidate evaluation')
    if _true_false(candidate_decision['approved_for_live_forecasting'],
                   'approved_for_live_forecasting').iloc[0]:
        raise ValueError('Live forecasting must remain disabled in Phase 5B')
    described = [x.strip() for x in str(row['models_compared']).split(',')]
    if set(described) != set(MODELS) or len(described) != len(MODELS):
        raise ValueError('Phase 5B models do not match the available implementations')

    needed = {'model', 'training_end_month', 'forecast_month',
              'horizon_months', 'forecast_net_cash', 'experimental_only'}
    if not needed.issubset(candidates):
        raise ValueError('Phase 5B candidates are missing required fields')
    if len(candidates) != len(MODELS) * max_horizon:
        raise ValueError('Phase 5B candidates have incorrect row count')
    if not _true_false(candidates['experimental_only'], 'experimental_only').all():
        raise ValueError('All candidate forecasts must remain experimental')
    amounts = pd.to_numeric(candidates['forecast_net_cash'], errors='coerce')
    horizons = pd.to_numeric(candidates['horizon_months'], errors='coerce')
    if not np.isfinite(amounts.to_numpy(float)).all() or horizons.isna().any():
        raise ValueError('Invalid candidate forecasts or horizons')

    expected = _candidate_values(training['net_cash_collected'].to_numpy(float), max_horizon)
    observed_keys = set()
    for idx, candidate in candidates.iterrows():
        model = candidate['model']
        h = horizons.loc[idx]
        if model not in MODELS or h != int(h) or not 1 <= int(h) <= max_horizon:
            raise ValueError('Unexpected candidate model or horizon')
        h = int(h)
        key = (model, h)
        if key in observed_keys:
            raise ValueError('Duplicate candidate model-horizon pair')
        observed_keys.add(key)
        if (str(candidate['training_end_month']) != expected_end or
                str(candidate['forecast_month']) != str(training['month_period'].iloc[-1] + h)):
            raise ValueError('Candidate forecast dates are inconsistent')
        if abs(float(amounts.loc[idx]) - round(expected[model][h - 1], 2)) > .011:
            raise ValueError('Phase 5B candidate predictions differ from training-only forecasts')
    return True


def _backtest_records(months, cash, target_indices, max_horizon, split_name):
    """Use only observations available before each forecast target."""
    result = []
    for target in target_indices:
        for horizon in range(1, max_horizon + 1):
            origin = target - horizon
            history_through_origin = cash[:origin + 1]
            candidates = _candidate_values(history_through_origin, max_horizon)
            for model in MODELS:
                prediction = float(candidates[model][horizon - 1])
                actual = float(cash[target])
                result.append({
                    'split': split_name,
                    'model': model,
                    'origin_month': str(months[origin]),
                    'target_month': str(months[target]),
                    'horizon_months': horizon,
                    'actual_net_cash': actual,
                    'predicted_net_cash': prediction,
                    'error': prediction - actual,
                    'absolute_error': abs(prediction - actual),
                    'experimental_only': True,
                })
    return result


def _score(df, model, horizon=None, selected=False):
    subset = df.loc[df['model'].eq(model)]
    if horizon is not None:
        subset = subset.loc[subset['horizon_months'].eq(horizon)]
    if subset.empty:
        raise ValueError('Cannot score an empty historical prediction set')
    errors = subset['error'].to_numpy(float)
    return {
        'model': model,
        'horizon_months': 'all' if horizon is None else horizon,
        'target_months': int(subset['target_month'].nunique()),
        'predictions': len(subset),
        'mae': round(float(np.abs(errors).mean()), 2),
        'rmse': round(float(np.sqrt(np.mean(errors ** 2))), 2),
        'mean_error': round(float(errors.mean()), 2),
        'selected_on_development': bool(selected),
    }


def evaluate_revenue_forecasts(history, readiness_decision,
                               forecast_candidates, model_decision,
                               holdout_months=6, min_train_months=18,
                               validation_months=6, max_horizon=3):
    """Return five aggregate reports for retrospective experimental backtests.

    The winner is selected solely on expanding-window development backtests.
    Every held-out target is forecast using only months preceding that target.
    Holdout metrics are descriptive and are NOT used to tune or reselect a model.
    """
    for name, value, minimum in (
        ('validation_months', validation_months, 2),
        ('max_horizon', max_horizon, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if max_horizon > holdout_months:
        raise ValueError('Forecast horizon cannot exceed holdout months')
    training, holdout_periods = _validate_and_split(
        history, readiness_decision, holdout_months, min_train_months)
    if len(training) - validation_months < min_train_months:
        raise ValueError('Development history is too short for internal validation')
    if len(training) - validation_months < max_horizon:
        raise ValueError('Insufficient pre-validation history for selected horizons')
    _validate_candidate_reports(forecast_candidates, model_decision,
                                training, holdout_periods, max_horizon)

    #_validate_and_split already checked unique, ordered, contiguous eligibility.
    #Only use the complete eligible series (never the incomplete current month).
    eligible_flag = _true_false(history['eligible_for_backtest'], 'eligible_for_backtest')
    eligible = history.loc[eligible_flag].copy().reset_index(drop=True)
    months = _month_periods(eligible['payment_month']).tolist()
    cash = pd.to_numeric(eligible['net_cash_collected'], errors='raise').to_numpy(float)
    dev_length = len(training)
    if len(eligible) != dev_length + holdout_months:
        raise ValueError('Eligible history does not match the expected development/test split')
    if months[dev_length:] != holdout_periods:
        raise ValueError('Holdout month ordering differs from Phase 5B')

    #Reserve the last validation_months WITHIN development to select a model.
    #No May-October 2026 cash values influence this selection.
    validation_targets = range(dev_length - validation_months, dev_length)
    validation_records = _backtest_records(
        months, cash, validation_targets, max_horizon, 'development_validation')
    validation = pd.DataFrame(validation_records)

    errors = {}
    for model in MODELS:
        e = validation.loc[validation['model'].eq(model), 'error'].to_numpy(float)
        errors[model] = (float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))))
    selected = min(MODELS, key=lambda m: (errors[m][0], errors[m][1], MODELS.index(m)))

    #This is a walk-forward evaluation: earlier observed holdout months may be
    #known at LATER origins. No forecast uses its own target or future months.
    holdout_targets = range(dev_length, len(eligible))
    holdout_records = _backtest_records(
        months, cash, holdout_targets, max_horizon, 'reserved_holdout')
    predictions = pd.DataFrame(validation_records + holdout_records)
    predictions['selected_on_development'] = predictions['model'].eq(selected)
    for col in ('predicted_net_cash', 'error', 'absolute_error'):
        predictions[col] = predictions[col].round(2)
    predictions = predictions[PREDICTION_COLUMNS]

    validation_scores = pd.DataFrame([
        _score(validation, m, selected=(m == selected)) for m in MODELS
    ], columns=SCORE_COLUMNS)
    validation_scores['selection_rank'] = [
        sorted(MODELS, key=lambda m: (errors[m][0], errors[m][1], MODELS.index(m))).index(m) + 1
        for m in MODELS
    ]
    holdout = predictions.loc[predictions['split'].eq('reserved_holdout')]
    holdout_scores = pd.DataFrame([
        _score(holdout, m, horizon=h, selected=(m == selected))
        for m in MODELS for h in [None, *range(1, max_horizon + 1)]
    ], columns=SCORE_COLUMNS)
    selected_summary = holdout_scores.loc[
        holdout_scores['model'].eq(selected) & holdout_scores['horizon_months'].eq('all')
    ].iloc[0]
    baseline_summary = holdout_scores.loc[
        holdout_scores['model'].eq('last_month') & holdout_scores['horizon_months'].eq('all')
    ].iloc[0]

    checks = pd.DataFrame([
        {'check': 'Phase 5A readiness', 'observed': True,
         'requirement': 'Historical evaluation approved', 'passed': True,
         'detail': 'Development/holdout split passed Phase 5A validation'},
        {'check': 'Phase 5B candidates', 'observed': len(forecast_candidates),
         'requirement': len(MODELS) * max_horizon, 'passed': True,
         'detail': 'Candidate values recomputed solely from development months'},
        {'check': 'Development validation', 'observed': validation_months,
         'requirement': f'{validation_months} months inside development', 'passed': True,
         'detail': 'Only this segment chooses the winner'},
        {'check': 'Reserved holdout', 'observed': holdout_months,
         'requirement': f'{holdout_months} complete months', 'passed': True,
         'detail': 'Holdout outcomes used only after selection; never future-target data'},
        {'check': 'Forecast horizons', 'observed': max_horizon,
         'requirement': f'1 to {holdout_months} months ahead', 'passed': True,
         'detail': 'Overlapping horizon errors are correlated, not independent samples'},
    ], columns=CHECK_COLUMNS)

    decision = pd.DataFrame([{
        'status': 'retrospective_backtest_complete',
        'selected_model': selected,
        'selection_method': 'Lowest development-validation MAE; RMSE tie-break',
        'training_end_month': str(months[dev_length - 1]),
        'holdout_start_month': str(holdout_periods[0]),
        'holdout_end_month': str(holdout_periods[-1]),
        'holdout_months': holdout_months,
        'forecast_horizons': max_horizon,
        'development_validation_mae': validation_scores.loc[
            validation_scores['model'].eq(selected), 'mae'].iloc[0],
        'holdout_mae': selected_summary['mae'],
        'holdout_rmse': selected_summary['rmse'],
        'holdout_last_month_mae': baseline_summary['mae'],
        'beats_last_month_on_holdout': bool(selected_summary['mae'] < baseline_summary['mae']),
        'research_results_available': True,
        'approved_for_live_forecasting': False,
        'note': ('This is a retrospective rolling-origin exercise on example data. '
                 'Six held-out months give limited evidence; 1-3-month-ahead errors '
                 'for the same targets overlap. Past observed holdout months are '
                 'allowed at later origins. No live forecasts or credible prediction '
                 'intervals are authorized.'),
    }], columns=DECISION_COLUMNS)

    return {
        'forecast_backtest_predictions': predictions,
        'forecast_validation_scores': validation_scores,
        'forecast_holdout_scores': holdout_scores,
        'forecast_backtest_checks': checks,
        'forecast_backtest_decision': decision,
    }


def run_forecast_evaluation(history, readiness_decision,
                            forecast_candidates, model_decision, output_dir=None,
                            **kwargs):
    reports = evaluate_revenue_forecasts(
        history, readiness_decision, forecast_candidates, model_decision, **kwargs)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in reports.items():
            table.to_csv(destination / f'{name}.csv', index=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Phase 5C: historical revenue forecast backtesting')
    parser.add_argument('--history', default='outputs/predictive/revenue_readiness/forecast_history.csv')
    parser.add_argument('--readiness', default='outputs/predictive/revenue_readiness/forecast_readiness_decision.csv')
    parser.add_argument('--candidates', default='outputs/predictive/revenue_models/forecast_model_candidates.csv')
    parser.add_argument('--models-decision', default='outputs/predictive/revenue_models/forecast_model_decision.csv')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = []
    for name in ('history', 'readiness', 'candidates', 'models_decision'):
        p = Path(getattr(args, name))
        paths.append(p if p.is_absolute() else root / p)
    if not all(p.exists() for p in paths):
        parser.error('Required Phase 5A/5B reports are missing. Run Phases 5A and 5B first.')
    reports = run_forecast_evaluation(
        *(pd.read_csv(p) for p in paths),
        output_dir=root / 'outputs' / 'predictive' / 'revenue_evaluation',
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    decision = reports['forecast_backtest_decision'].iloc[0]
    print('Development-selected method:', decision['selected_model'])
    print('Held-out MAE:', f"${decision['holdout_mae']:,.2f}")
    print('Live forecasting remains DISABLED. These are experimental historical results.')


if __name__ == '__main__':
    main()
