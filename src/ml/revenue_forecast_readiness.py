#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5A - Revenue Forecasting Data Readiness

"""Audit historical *net cash collected* before attempting any forecasts.

This is a readiness check, NOT a forecasting model.  It intentionally reuses
Phase 2C's payment accounting to avoid disagreements over refunds/failed charges.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.analytics.financial_membership_insights import monthly_cash_performance


HISTORY_COLUMNS = [
    'payment_month', 'gross_cash_collected', 'refunded_cash', 'net_cash_collected',
    'completed_charge_count', 'refund_count', 'partial_month',
    'first_month_coverage_uncertain', 'eligible_for_backtest', 'exclusion_reason',
]
CHECK_COLUMNS = ['check', 'observed', 'requirement', 'passed', 'detail']
DECISION_COLUMNS = [
    'as_of', 'forecast_target', 'first_payment_date', 'last_payment_date',
    'calendar_months', 'eligible_complete_months', 'training_months_available',
    'reserved_backtest_months', 'earliest_eligible_month', 'latest_eligible_month',
    'source_freshness_days', 'ready_for_experimental_backtesting',
    'ready_for_live_forecasting', 'status', 'note',
]


def _as_of(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError('as_of must be a valid date')
    return result.normalize()


def assess_revenue_forecast_readiness(payments, as_of, min_train_months=18,
                                      backtest_months=6, max_source_lag_days=7):
    """Return history, checks and decision DataFrames; never train a model.

    Each month is calendar-aligned, and cash = completed payments - completed
    refunds, using the same definitions as Phase 2C. Failed/pending payment
    attempts are not cash. Months with zero transactions are marked, not treated
    as proof of missing data. The first observed month is conservatively excluded
    if its observation coverage is ambiguous. The report month is excluded if
    partial. An old data export cannot be silently extended with fake zero months.
    """
    if not isinstance(payments, pd.DataFrame):
        raise TypeError('payments must be a pandas DataFrame')
    for name, value, minimum in (
        ('min_train_months', min_train_months, 1),
        ('backtest_months', backtest_months, 1),
        ('max_source_lag_days', max_source_lag_days, 0),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')

    date = _as_of(as_of)
    required = {'payment_date', 'amount', 'payment_status', 'transaction_type'}
    missing = required - set(payments.columns)
    if missing:
        raise ValueError(f'Missing payment columns: {sorted(missing)}')

    data = payments.copy()
    data['payment_date'] = pd.to_datetime(data['payment_date'], errors='raise').dt.normalize()
    if data['payment_date'].isna().any():
        raise ValueError('Payment dates cannot be missing')
    if 'payment_id' in data.columns:
        ids = data['payment_id'].astype('string').str.strip()
        if ids.isna().any() or ids.eq('').any() or ids.duplicated().any():
            raise ValueError('Payment IDs must be present and unique')

    #Only transactions known by the reporting date can contribute to history.
    observed = data.loc[data['payment_date'] <= date].copy()
    if not observed.empty:
        #Validate amounts and statuses through the existing financial module.
        monthly = monthly_cash_performance(observed, date)
        first_date = observed['payment_date'].min()
        last_date = observed['payment_date'].max()
        #Prevent unobserved months after the last transaction from becoming
        #artificially recorded $0 months in a stale dataset.
        monthly = monthly.loc[
            monthly['payment_month'] <= last_date.to_period('M')
        ].copy()
    else:
        first_date = pd.NaT
        last_date = pd.NaT
        monthly = pd.DataFrame(columns=HISTORY_COLUMNS)

    if not monthly.empty:
        monthly['first_month_coverage_uncertain'] = False
        first_month = first_date.to_period('M')
        uncertain_first_month = first_date.day > 1
        if uncertain_first_month:
            monthly.loc[monthly['payment_month'].eq(first_month),
                        'first_month_coverage_uncertain'] = True
        #The most recent month is also uncertain if the report refers to a
        #later date but the latest available transaction is not recent enough.
        monthly['eligible_for_backtest'] = (
            ~monthly['partial_month'].astype(bool) &
            ~monthly['first_month_coverage_uncertain'].astype(bool)
        )
        monthly['exclusion_reason'] = ''
        monthly.loc[monthly['first_month_coverage_uncertain'], 'exclusion_reason'] = \
            'First observed month: coverage may be incomplete'
        monthly.loc[monthly['partial_month'], 'exclusion_reason'] = \
            'Reporting month is incomplete'
        monthly['payment_month'] = monthly['payment_month'].astype(str)
        monthly = monthly[HISTORY_COLUMNS].reset_index(drop=True)

    eligible = monthly.loc[monthly['eligible_for_backtest']] if len(monthly) else monthly
    eligible_count = len(eligible)
    train_count = max(eligible_count - backtest_months, 0)
    freshness = int((date - last_date).days) if pd.notna(last_date) else None
    fresh = freshness is not None and freshness <= max_source_lag_days
    months_ok = train_count >= min_train_months
    enough_for_backtest = eligible_count >= (min_train_months + backtest_months)
    no_negative_month = bool((eligible['net_cash_collected'] >= 0).all()) if eligible_count else True
    #Negative NET months can occur legitimately due to refunds; retain them,
    #but require review before we choose a non-negative forecasting model.
    zero_cash_count = int(eligible['net_cash_collected'].eq(0).sum()) if eligible_count else 0

    checks = pd.DataFrame([
        {'check': 'Observed transactions', 'observed': len(observed),
         'requirement': 'At least one transaction', 'passed': not observed.empty,
         'detail': 'Records dated after as_of are excluded'},
        {'check': 'Complete usable months', 'observed': eligible_count,
         'requirement': f'At least {min_train_months + backtest_months}',
         'passed': enough_for_backtest,
         'detail': 'Excludes incomplete reporting months and uncertain first month'},
        {'check': 'Training history after holdout', 'observed': train_count,
         'requirement': f'At least {min_train_months}', 'passed': months_ok,
         'detail': f'{backtest_months} months reserved for future time-ordered evaluation'},
        {'check': 'Source freshness', 'observed': freshness,
         'requirement': f'Last observed transaction within {max_source_lag_days} days',
         'passed': fresh,
         'detail': 'A stale export must not be interpreted as months of zero revenue'},
        {'check': 'Negative net cash months', 'observed': int(not no_negative_month),
         'requirement': 'Advisory: review any negative net months',
         'passed': no_negative_month,
         'detail': 'Completed refunds can exceed collections; not automatically invalid'},
        {'check': 'Zero-cash months', 'observed': zero_cash_count,
         'requirement': 'Advisory: verify zero-cash periods with the business',
         'passed': True,
         'detail': 'Zero cash is not necessarily missing data'},
    ], columns=CHECK_COLUMNS)

    approval = bool(not observed.empty and enough_for_backtest and fresh)
    decision = pd.DataFrame([{
        'as_of': date.date().isoformat(),
        'forecast_target': 'Monthly completed net cash collected (payments minus refunds)',
        'first_payment_date': first_date.date().isoformat() if pd.notna(first_date) else '',
        'last_payment_date': last_date.date().isoformat() if pd.notna(last_date) else '',
        'calendar_months': len(monthly),
        'eligible_complete_months': eligible_count,
        'training_months_available': train_count,
        'reserved_backtest_months': backtest_months,
        'earliest_eligible_month': eligible['payment_month'].iloc[0] if eligible_count else '',
        'latest_eligible_month': eligible['payment_month'].iloc[-1] if eligible_count else '',
        'source_freshness_days': freshness,
        'ready_for_experimental_backtesting': approval,
        'ready_for_live_forecasting': False,
        'status': 'ready_for_model_comparison' if approval else 'insufficient_or_unverified_history',
        'note': ('Readiness only. No forecast trained or validated. Source is example data; '
                 'a separate out-of-time backtest and error comparison are required.'),
    }], columns=DECISION_COLUMNS)
    return {'forecast_history': monthly, 'forecast_readiness_checks': checks,
            'forecast_readiness_decision': decision}


def run_revenue_readiness(payments, as_of, output_dir=None, **kwargs):
    reports = assess_revenue_forecast_readiness(payments, as_of, **kwargs)
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(folder / f'{name}.csv', index=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Audit monthly net cash forecasting readiness')
    parser.add_argument('--as-of', required=True, help='YYYY-MM-DD reporting date')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    payments_path = root / 'data' / 'processed' / 'payments.csv'
    if not payments_path.exists():
        parser.error(f'Processed payment data not found: {payments_path}')
    payments = pd.read_csv(payments_path)
    reports = run_revenue_readiness(
        payments, args.as_of,
        output_dir=root / 'outputs' / 'predictive' / 'revenue_readiness',
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    row = reports['forecast_readiness_decision'].iloc[0]
    print(f"Experimental backtest readiness: {row['ready_for_experimental_backtesting']}")
    print('Reports saved to outputs/predictive/revenue_readiness')


if __name__ == '__main__':
    main()
