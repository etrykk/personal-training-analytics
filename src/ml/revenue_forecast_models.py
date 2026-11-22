#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5B - Experimental Monthly Cash Forecast Candidates

"""Create simple forecast candidates using TRAINING months only.

This is NOT a live prediction service, and it does not evaluate test errors.
Phase 5A's complete-month history is split chronologically; the last six
eligible months are reserved for Phase 5C backtesting and are NEVER used for
model selection, fitting, or candidate forecasts in this module.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = [
    'model', 'training_end_month', 'forecast_month', 'horizon_months',
    'forecast_net_cash', 'experimental_only',
]
CHECK_COLUMNS = ['check', 'observed', 'requirement', 'passed', 'detail']
DECISION_COLUMNS = [
    'status', 'training_start_month', 'training_end_month', 'training_months',
    'holdout_start_month', 'holdout_end_month', 'reserved_holdout_months',
    'forecast_horizons', 'models_compared', 'approved_for_live_forecasting', 'note',
]
MODELS = ('last_month', 'trailing_3_month_mean', 'linear_trend_last_6_months')


def _true_false(series, column):
    """Read boolean flags reliably from DataFrames or CSV strings."""
    normalized = series.astype('string').str.strip().str.lower()
    invalid = ~normalized.isin(['true', 'false'])
    if invalid.any():
        raise ValueError(f'{column} must contain only True or False')
    return normalized.eq('true')


def _month_periods(values):
    strings = values.astype('string').str.strip()
    if strings.isna().any() or not strings.str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('payment_month must have YYYY-MM format')
    try:
        result = pd.PeriodIndex(strings.tolist(), freq='M')
    except ValueError as exc:
        raise ValueError('Invalid calendar month in payment_month') from exc
    return result


def _validate_and_split(history, readiness_decision, holdout_months, min_train_months):
    """Validate dates and readiness before using history for development."""
    if not isinstance(history, pd.DataFrame) or not isinstance(readiness_decision, pd.DataFrame):
        raise TypeError('history and readiness_decision must be DataFrames')
    if isinstance(holdout_months, bool) or not isinstance(holdout_months, int) or holdout_months < 1:
        raise ValueError('holdout_months must be a positive integer')
    if isinstance(min_train_months, bool) or not isinstance(min_train_months, int) or min_train_months < 6:
        raise ValueError('min_train_months must be an integer of at least 6')
    if len(readiness_decision) != 1:
        raise ValueError('readiness_decision must contain exactly one row')

    needed = {'payment_month', 'net_cash_collected', 'eligible_for_backtest'}
    missing = needed - set(history)
    if missing:
        raise ValueError(f'Missing history fields: {sorted(missing)}')
    decision_needed = {'ready_for_experimental_backtesting',
                       'reserved_backtest_months', 'training_months_available'}
    missing = decision_needed - set(readiness_decision)
    if missing:
        raise ValueError(f'Missing decision fields: {sorted(missing)}')

    ready_flag = _true_false(readiness_decision['ready_for_experimental_backtesting'],
                             'ready_for_experimental_backtesting').iloc[0]
    if not bool(ready_flag):
        raise ValueError('Phase 5A did not approve experimental backtesting')

    decision = readiness_decision.iloc[0]
    reserved = pd.to_numeric(decision['reserved_backtest_months'], errors='coerce')
    expected_train = pd.to_numeric(decision['training_months_available'], errors='coerce')
    if pd.isna(reserved) or int(reserved) != reserved or int(reserved) != holdout_months:
        raise ValueError('Holdout size must match the Phase 5A decision')
    if pd.isna(expected_train) or int(expected_train) != expected_train:
        raise ValueError('Phase 5A training-month count is invalid')

    data = history.copy(deep=True)
    periods = _month_periods(data['payment_month'])
    if periods.has_duplicates or not periods.is_monotonic_increasing:
        raise ValueError('History months must be strictly increasing and unique')
    data['month_period'] = periods
    data['eligible'] = _true_false(data['eligible_for_backtest'], 'eligible_for_backtest')
    data['net_cash_collected'] = pd.to_numeric(data['net_cash_collected'], errors='coerce')
    if not np.isfinite(data['net_cash_collected'].to_numpy(dtype=float)).all():
        raise ValueError('Net cash values must be finite numbers')

    #Only the eligible complete months form the forecasting series.
    eligible = data.loc[data['eligible']].copy().reset_index(drop=True)
    if len(eligible) < min_train_months + holdout_months:
        raise ValueError('Not enough complete months for training and holdout')
    consecutive = eligible['month_period'].tolist()
    if any(consecutive[i] != consecutive[i-1] + 1 for i in range(1, len(consecutive))):
        raise ValueError('Eligible months must be consecutive without gaps')
    if (len(eligible) - holdout_months) != int(expected_train):
        raise ValueError('The Phase 5A training-month count does not match history')

    #Crucial: NEVER use the reserved holdout cash values in Phase 5B.
    training = eligible.iloc[:-holdout_months].copy()
    holdout_periods = eligible['month_period'].iloc[-holdout_months:].tolist()
    return training, holdout_periods


def _candidate_values(training_cash, max_horizon):
    """Produce one- to three-month-ahead baselines and a short linear trend."""
    values = np.asarray(training_cash, dtype=float)
    if len(values) < 6 or not np.isfinite(values).all():
        raise ValueError('At least six finite training months are required')

    last_cash = float(values[-1])
    moving_cash = float(values[-3:].mean())
    #Six-month ordinary least squares with months indexed 0,1,2,3,4,5.
    #This fitted trend uses NO month from the reserved holdout.
    recent = values[-6:]
    slope, intercept = np.polyfit(np.arange(6, dtype=float), recent, 1)
    return {
        'last_month': [last_cash] * max_horizon,
        'trailing_3_month_mean': [moving_cash] * max_horizon,
        'linear_trend_last_6_months': [float(slope * (5 + h) + intercept)
                                      for h in range(1, max_horizon + 1)],
    }


def create_revenue_forecast_candidates(history, readiness_decision,
                                       holdout_months=6, min_train_months=18,
                                       max_horizon=3):
    """Build comparison candidates, diagnostics and a non-deployment decision.

    Forecast dates begin immediately AFTER the training period; therefore the
    displayed candidates fall within the *reserved* evaluation interval. This
    phase doesn't assess whether these predictions are good: Phase 5C does.
    """
    if isinstance(max_horizon, bool) or not isinstance(max_horizon, int) or max_horizon < 1:
        raise ValueError('max_horizon must be a positive integer')
    if max_horizon > holdout_months:
        raise ValueError('max_horizon cannot exceed the reserved holdout size')

    training, holdout_periods = _validate_and_split(
        history, readiness_decision, holdout_months, min_train_months)
    train_periods = training['month_period'].tolist()
    end = train_periods[-1]
    values = _candidate_values(training['net_cash_collected'].to_numpy(), max_horizon)

    rows = []
    for model in MODELS:
        for h, cash in enumerate(values[model], start=1):
            rows.append({
                'model': model,
                'training_end_month': str(end),
                'forecast_month': str(end + h),
                'horizon_months': h,
                'forecast_net_cash': round(float(cash), 2),
                'experimental_only': True,
            })
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    checks = pd.DataFrame([
        {'check': 'Phase 5A readiness', 'observed': True,
         'requirement': 'Approved for experimental comparison', 'passed': True,
         'detail': 'Readiness does not authorize live forecasts'},
        {'check': 'Training months', 'observed': len(training),
         'requirement': f'At least {min_train_months}', 'passed': True,
         'detail': 'Only completed historical months before reserved holdout'},
        {'check': 'Reserved holdout', 'observed': len(holdout_periods),
         'requirement': f'Exactly {holdout_months} months', 'passed': True,
         'detail': 'Holdout cash values never used by candidate models'},
        {'check': 'Contiguous months', 'observed': True,
         'requirement': 'No gaps in eligible history', 'passed': True,
         'detail': 'Calendar-month ordering verified'},
        {'check': 'Candidate horizons', 'observed': max_horizon,
         'requirement': f'Between 1 and {holdout_months} months', 'passed': True,
         'detail': 'Models extrapolate from the same training cutoff'},
    ], columns=CHECK_COLUMNS)

    decision = pd.DataFrame([{
        'status': 'candidates_ready_for_out_of_time_evaluation',
        'training_start_month': str(train_periods[0]),
        'training_end_month': str(end),
        'training_months': len(training),
        'holdout_start_month': str(holdout_periods[0]),
        'holdout_end_month': str(holdout_periods[-1]),
        'reserved_holdout_months': len(holdout_periods),
        'forecast_horizons': max_horizon,
        'models_compared': ', '.join(MODELS),
        'approved_for_live_forecasting': False,
        'note': ('Model candidates only. No out-of-time forecast errors evaluated; '
                 'Phase 5C must compare MAE/RMSE before any display. '
                 'This dataset is example business data.'),
    }], columns=DECISION_COLUMNS)
    return {'forecast_model_candidates': candidates,
            'forecast_model_checks': checks,
            'forecast_model_decision': decision}


def run_forecast_candidates(history, readiness_decision, output_dir=None, **kwargs):
    """Run candidate forecasting and optionally export three CSV reports."""
    result = create_revenue_forecast_candidates(history, readiness_decision, **kwargs)
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_csv(folder / f'{name}.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description='Phase 5B: historical cash forecast candidates')
    parser.add_argument('--history', default='outputs/predictive/revenue_readiness/forecast_history.csv')
    parser.add_argument('--decision', default='outputs/predictive/revenue_readiness/forecast_readiness_decision.csv')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    history_file = Path(args.history)
    decision_file = Path(args.decision)
    if not history_file.is_absolute():
        history_file = root / history_file
    if not decision_file.is_absolute():
        decision_file = root / decision_file
    if not history_file.exists() or not decision_file.exists():
        parser.error('Missing Phase 5A reports. Run python -m src.ml.revenue_forecast_readiness --as-of 2026-11-20 first.')
    outputs = run_forecast_candidates(
        pd.read_csv(history_file), pd.read_csv(decision_file),
        output_dir=root / 'outputs/predictive/revenue_models')
    for name, frame in outputs.items():
        print(f'{name}: {len(frame)} row(s)')
    decision = outputs['forecast_model_decision'].iloc[0]
    print(f"Development: {decision['training_start_month']} to {decision['training_end_month']}")
    print(f"Reserved holdout: {decision['holdout_start_month']} to {decision['holdout_end_month']}")
    print('Live forecasts: DISABLED. Candidates are for Phase 5C backtesting only.')


if __name__ == '__main__':
    main()
