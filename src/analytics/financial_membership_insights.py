#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Advanced Revenue & Membership Insights

from pathlib import Path
import argparse

import pandas as pd


MONTHLY_COLUMNS = [
    'payment_month', 'gross_cash_collected', 'refunded_cash',
    'net_cash_collected', 'completed_charge_count', 'refund_count',
    'rolling_3m_net_cash', 'growth_pct', 'partial_month'
]
PAYMENT_COLUMNS = ['payment_status', 'transaction_count', 'attempted_amount',
                   'share_of_attempts_pct']
WATCHLIST_COLUMNS = ['membership_id', 'client_id', 'plan_name', 'end_date',
                     'days_until_expiry', 'monthly_fee']
SNAPSHOT_COLUMNS = ['as_of', 'active_clients', 'active_membership_records',
                    'expiring_clients_30d']
RETENTION_COLUMNS = ['period_start', 'period_end', 'starting_clients',
                     'retained_clients', 'no_longer_active',
                     'retention_pct', 'churn_pct']


def _as_of_date(value):
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError('Reporting date must be valid')
    return date.normalize()


def _payments_as_of(payments, as_of):
    #Use transaction amounts/statuses, not an assumed revenue or profit field.
    required = {'payment_date', 'amount', 'payment_status', 'transaction_type'}
    if not required.issubset(payments.columns):
        raise ValueError(f'Missing payment columns: {sorted(required - set(payments.columns))}')
    data = payments.copy()
    data['payment_date'] = pd.to_datetime(data['payment_date'], errors='raise').dt.normalize()
    data['amount'] = pd.to_numeric(data['amount'], errors='raise')
    if data['payment_date'].isna().any() or data['amount'].isna().any() or (data['amount'] <= 0).any():
        raise ValueError('Payment dates must be valid and amounts positive')
    if not data['payment_status'].isin({'completed', 'pending', 'failed'}).all():
        raise ValueError('Unknown payment status')
    if not data['transaction_type'].isin({'payment', 'refund'}).all():
        raise ValueError('Unknown transaction type')
    if (data['transaction_type'].eq('refund') & ~data['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions')
    return data.loc[data['payment_date'] <= _as_of_date(as_of)].copy()


def monthly_cash_performance(payments, as_of):
    '''
    Report actual completed cash transactions by calendar month, with refunds
    recorded in the month processed. Pending/failed charges are excluded.

    3-month rolling averages require 3 observed calendar months. The final
    month may be incomplete; it is explicitly marked rather than annualized.
    '''
    as_of = _as_of_date(as_of)
    data = _payments_as_of(payments, as_of)
    if data.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    completed = data.loc[data['payment_status'].eq('completed')].copy()
    completed['payment_month'] = completed['payment_date'].dt.to_period('M')
    charges = completed['transaction_type'].eq('payment')
    refunds = completed['transaction_type'].eq('refund')
    completed['gross_cash_collected'] = completed['amount'].where(charges, 0.0)
    completed['refunded_cash'] = completed['amount'].where(refunds, 0.0)
    completed['completed_charge_count'] = charges.astype(int)
    completed['refund_count'] = refunds.astype(int)

    #A month with only pending/failed transactions still appears with zero cash.
    month_range = pd.period_range(data['payment_date'].min().to_period('M'),
                                  as_of.to_period('M'), freq='M')
    if completed.empty:
        totals = pd.DataFrame(index=month_range)
    else:
        totals = completed.groupby('payment_month')[[
            'gross_cash_collected', 'refunded_cash',
            'completed_charge_count', 'refund_count'
        ]].sum().reindex(month_range, fill_value=0)

    for column in ['gross_cash_collected', 'refunded_cash',
                   'completed_charge_count', 'refund_count']:
        if column not in totals:
            totals[column] = 0
    totals = totals.fillna(0)
    totals['net_cash_collected'] = (totals['gross_cash_collected'] - totals['refunded_cash'])
    totals['rolling_3m_net_cash'] = totals['net_cash_collected'].rolling(3, min_periods=3).mean()
    previous = totals['net_cash_collected'].shift(1)
    totals['growth_pct'] = (100 * (totals['net_cash_collected'] - previous) / previous).where(previous > 0)
    totals['partial_month'] = totals.index == as_of.to_period('M')
    #If the report is on the month's last day it is not a partial month.
    if as_of == as_of.to_period('M').end_time.normalize():
        totals.loc[as_of.to_period('M'), 'partial_month'] = False
    totals.index.name = 'payment_month'
    result = totals.reset_index()
    return result[MONTHLY_COLUMNS]


def payment_status_summary(payments, as_of, days=90):
    '''
    Count charge attempts by status over the last `days` calendar dates.
    Excludes refunds, which are not additional payment attempts.
    Pending and failed amounts are NOT necessarily collectible balances.
    '''
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise ValueError('days must be a positive integer')
    as_of = _as_of_date(as_of)
    data = _payments_as_of(payments, as_of)
    first_date = as_of - pd.Timedelta(days=days - 1)
    attempts = data.loc[data['transaction_type'].eq('payment') &
                        data['payment_date'].between(first_date, as_of)]
    statuses = ['completed', 'pending', 'failed']
    grouped = attempts.groupby('payment_status')['amount'].agg(['size', 'sum'])
    grouped = grouped.reindex(statuses, fill_value=0)
    total = int(grouped['size'].sum())
    grouped['share_of_attempts_pct'] = (
        100 * grouped['size'] / total if total else pd.NA
    )
    return grouped.rename(columns={'size': 'transaction_count', 'sum': 'attempted_amount'}).rename_axis(
        'payment_status').reset_index()[PAYMENT_COLUMNS]


def _memberships_with_dates(memberships):
    required = {'membership_id', 'client_id', 'start_date', 'end_date'}
    if not required.issubset(memberships.columns):
        raise ValueError(f'Missing membership columns: {sorted(required - set(memberships.columns))}')
    data = memberships.copy()
    for field in ['start_date', 'end_date']:
        data[field] = pd.to_datetime(data[field], errors='raise').dt.normalize()
    if data[['membership_id', 'client_id', 'start_date']].isna().any().any():
        raise ValueError('Membership identifiers and start dates are required')
    if data['membership_id'].duplicated().any():
        raise ValueError('Membership identifiers must be unique')
    if (data['end_date'].notna() & (data['end_date'] < data['start_date'])).any():
        raise ValueError('Membership end_date cannot precede start_date')
    if 'plan_name' not in data:
        data['plan_name'] = pd.NA
    if 'monthly_fee' not in data:
        data['monthly_fee'] = pd.NA
    return data


def _active_memberships(data, date):
    return data.loc[(data['start_date'] <= date) &
                    (data['end_date'].isna() | (data['end_date'] >= date))]


def membership_expiry_watchlist(memberships, as_of, days_ahead=30):
    '''
    Current coverage records ending within days_ahead days, inclusive.
    Ignore a record if its client already has another membership covering
    the next calendar day (continuous coverage/plan change). Expiration is
    a renewal opportunity, NOT a proven cancellation or predicted churn.
    '''
    if not isinstance(days_ahead, int) or isinstance(days_ahead, bool) or days_ahead < 0:
        raise ValueError('days_ahead must be a nonnegative integer')
    as_of = _as_of_date(as_of)
    data = _memberships_with_dates(memberships)
    active = _active_memberships(data, as_of)
    cutoff = as_of + pd.Timedelta(days=days_ahead)
    watch = active.loc[active['end_date'].notna() & active['end_date'].between(as_of, cutoff)].copy()
    if watch.empty:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    keep = []
    for _, row in watch.iterrows():
        next_day = row['end_date'] + pd.Timedelta(days=1)
        same_client = data.loc[(data['client_id'] == row['client_id']) &
                               (data['membership_id'] != row['membership_id'])]
        continuation = not _active_memberships(same_client, next_day).empty
        keep.append(not continuation)
    watch = watch.loc[keep].copy()
    watch['days_until_expiry'] = (watch['end_date'] - as_of).dt.days
    return watch[WATCHLIST_COLUMNS].sort_values(['end_date', 'client_id']).reset_index(drop=True)


def membership_coverage_summary(memberships, as_of):
    '''Point-in-time membership coverage based on dates, not current statuses.'''
    as_of = _as_of_date(as_of)
    data = _memberships_with_dates(memberships)
    active = _active_memberships(data, as_of)
    watch = membership_expiry_watchlist(data, as_of, days_ahead=30)
    return pd.DataFrame([{
        'as_of': as_of.date().isoformat(),
        'active_clients': int(active['client_id'].nunique()),
        'active_membership_records': len(active),
        'expiring_clients_30d': int(watch['client_id'].nunique()),
    }], columns=SNAPSHOT_COLUMNS)


def membership_period_retention(memberships, period_start, period_end):
    '''
    Historical client-level subscription retention between two dates.
    Clients joining during the window are not included in the starting
    denominator. A lapse followed by reactivation by period end is retained.
    If no clients were active at period_start, percentages are unavailable.
    '''
    start, end = _as_of_date(period_start), _as_of_date(period_end)
    if end < start:
        raise ValueError('period_end must be on or after period_start')
    data = _memberships_with_dates(memberships)
    starting = set(_active_memberships(data, start)['client_id'])
    ending = set(_active_memberships(data, end)['client_id'])
    retained = len(starting & ending)
    count = len(starting)
    return pd.DataFrame([{
        'period_start': start.date().isoformat(),
        'period_end': end.date().isoformat(),
        'starting_clients': count,
        'retained_clients': retained,
        'no_longer_active': count - retained,
        'retention_pct': 100 * retained / count if count else pd.NA,
        'churn_pct': 100 * (count - retained) / count if count else pd.NA,
    }], columns=RETENTION_COLUMNS)


def run_financial_membership_insights(payments, memberships, as_of, output_dir=None):
    '''Generate Phase 2C reports; optionally write reports to CSV.'''
    date = _as_of_date(as_of)
    reports = {
        'monthly_cash_performance': monthly_cash_performance(payments, date),
        'payment_status_90d': payment_status_summary(payments, date, days=90),
        'membership_coverage_snapshot': membership_coverage_summary(memberships, date),
        'membership_expiry_watchlist_30d': membership_expiry_watchlist(memberships, date),
        'membership_retention_30d': membership_period_retention(
            memberships, date - pd.Timedelta(days=30), date),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(output_dir / f'{name}.csv', index=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Generate V2 revenue and membership insights')
    parser.add_argument('--as-of', required=True, help='Report date (YYYY-MM-DD)')
    args = parser.parse_args()

    from src.revenue_analysis import load_payments

    project_root = Path(__file__).resolve().parents[2]
    source = project_root / 'data' / 'processed'
    payments = load_payments(source / 'payments.csv')
    memberships = pd.read_csv(source / 'memberships.csv')
    reports = run_financial_membership_insights(
        payments, memberships, as_of=args.as_of,
        output_dir=project_root / 'outputs' / 'financial' / 'v2',
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    print('Reports saved to outputs/financial/v2')


if __name__ == '__main__':
    main()
