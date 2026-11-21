#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Business Health Overview and Rule-Based Insights

from pathlib import Path
import argparse

import pandas as pd

from .client_insights import run_client_insights
from .financial_membership_insights import run_financial_membership_insights
from .session_trainer_insights import run_session_trainer_insights


OVERVIEW_COLUMNS = [
    'as_of', 'total_clients', 'clients_attending_30d',
    'clients_needing_followup', 'clients_without_session_history',
    'activity_continuation_pct', 'active_membership_clients',
    'membership_retention_30d_pct', 'memberships_expiring_30d',
    'scheduled_sessions_30d', 'completed_sessions_30d',
    'session_completion_rate_30d_pct', 'no_show_rate_30d_pct',
    'completed_training_hours_30d', 'trainers_with_completed_sessions_30d',
    'current_month_net_cash', 'current_month_partial',
    'last_completed_month', 'last_completed_month_net_cash',
    'last_completed_month_growth_pct', 'failed_payment_attempts_90d',
    'pending_payment_attempts_90d',
]
INSIGHT_COLUMNS = ['priority', 'category', 'headline', 'detail', 'observed_value']
TREND_COLUMNS = [
    'month', 'eligible_clients', 'clients_with_completed_sessions',
    'completed_sessions', 'monthly_activity_rate_pct',
    'gross_cash_collected', 'refunded_cash', 'net_cash_collected',
    'partial_month',
]


def _reporting_date(value):
    #Use the same reporting date across all analytics modules.
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError('as_of must be a valid reporting date')
    return date.normalize()


def _numeric_or_nan(value):
    #Keep unavailable percentages as missing, rather than inventing 0%.
    return float(value) if pd.notna(value) else float('nan')


def dashboard_monthly_trends(monthly_activity, monthly_cash):
    '''Align existing monthly activity and cash series for dashboard charts.

    Missing financial observations stay missing, not zero. The current
    calendar month can be incomplete and is flagged by partial_month.
    '''
    activity_columns = {'month', 'eligible_clients', 'clients_with_completed_sessions',
                        'completed_sessions', 'monthly_activity_rate_pct'}
    cash_columns = {'payment_month', 'gross_cash_collected', 'refunded_cash',
                    'net_cash_collected', 'partial_month'}
    if not activity_columns.issubset(monthly_activity.columns):
        raise ValueError('monthly_activity is missing required columns')
    if not cash_columns.issubset(monthly_cash.columns):
        raise ValueError('monthly_cash is missing required columns')

    #Select an explicit column order for predictable CSV and chart inputs.
    trends = monthly_activity[[
        'month', 'eligible_clients', 'clients_with_completed_sessions',
        'completed_sessions', 'monthly_activity_rate_pct'
    ]].copy()
    cash = monthly_cash[[
        'payment_month', 'gross_cash_collected', 'refunded_cash',
        'net_cash_collected', 'partial_month'
    ]].copy()
    trends['month'] = trends['month'].astype(str)
    cash['month'] = cash['payment_month'].astype(str)
    result = trends.merge(cash.drop(columns=['payment_month']), on='month', how='left',
                          validate='one_to_one').sort_values('month')
    return result[TREND_COLUMNS].reset_index(drop=True)


def business_health_insights(overview, no_show_threshold_pct=10.0):
    '''Create deterministic dashboard alerts, not churn predictions.

    no_show_threshold_pct is a configurable product rule, not an industry
    benchmark. No-show indicators require at least 10 scheduled sessions.
    '''
    if not isinstance(overview, pd.DataFrame) or len(overview) != 1:
        raise ValueError('overview must contain exactly one summary row')
    if not set(OVERVIEW_COLUMNS).issubset(overview.columns):
        raise ValueError('overview is missing required columns')
    if not 0 <= no_show_threshold_pct <= 100:
        raise ValueError('no_show_threshold_pct must be between 0 and 100')

    row = overview.iloc[0]
    results = []

    def add(priority, category, headline, detail, value):
        results.append({'priority': priority, 'category': category,
                        'headline': headline, 'detail': detail,
                        'observed_value': value})

    count = int(row['clients_needing_followup'])
    if count > 0:
        add('review', 'engagement', 'Review changes in client engagement',
            f'{count} clients have inactivity, recent attendance gaps, or declining session activity. '
            'These are activity indicators, not predicted cancellations.', count)

    expiring = int(row['memberships_expiring_30d'])
    if expiring > 0:
        add('review', 'membership', 'Membership expirations coming up',
            f'{expiring} clients have membership coverage scheduled to end within 30 days '
            'without a known continuous renewal.', expiring)

    no_shows = row['no_show_rate_30d_pct']
    if (int(row['scheduled_sessions_30d']) >= 10 and pd.notna(no_shows)
            and no_shows >= no_show_threshold_pct):
        add('monitor', 'sessions', 'Review session no-shows',
            f'The last 30-day no-show rate is {no_shows:.1f}%, meeting the configured '
            f'{no_show_threshold_pct:.1f}% indicator threshold.', round(float(no_shows), 2))

    failures = int(row['failed_payment_attempts_90d'])
    if failures > 0:
        add('review', 'payments', 'Review failed payment attempts',
            f'{failures} payment attempts failed during the last 90 days. '
            'This is an attempt count, not an outstanding balance.', failures)

    growth = row['last_completed_month_growth_pct']
    if pd.notna(growth) and growth < 0:
        add('monitor', 'revenue', 'Review last completed month cash change',
            f'Net cash collected changed {growth:.1f}% versus the preceding calendar month. '
            'Incomplete months are excluded from this comparison.', round(float(growth), 2))

    if not results:
        add('info', 'no_flags', 'No configured indicators triggered',
            'These rules found no flagged conditions; this does not establish business health '
            'or guarantee retention.', 0)
    return pd.DataFrame(results, columns=INSIGHT_COLUMNS)


def run_business_health(clients, memberships, sessions, payments, as_of,
                        output_dir=None, no_show_threshold_pct=10.0):
    '''Combine existing V2 analytics into a dashboard-ready business report.

    The module does not recalculate or invent an overall business score.
    Percentages keep their specific denominators. Cash is completed
    payments minus completed refunds, not profit or accrued revenue.
    '''
    date = _reporting_date(as_of)
    if not {'client_id', 'join_date'}.issubset(clients.columns):
        raise ValueError('clients must contain client_id and join_date')
    if clients['client_id'].isna().any() or clients['client_id'].duplicated().any():
        raise ValueError('clients must contain unique nonmissing client IDs')

    roster = clients.copy()
    roster['join_date'] = pd.to_datetime(roster['join_date'], errors='raise').dt.normalize()
    if roster['join_date'].isna().any():
        raise ValueError('join_date must contain valid dates')
    roster = roster.loc[roster['join_date'] <= date].copy()

    client = run_client_insights(roster, sessions, date)
    financial = run_financial_membership_insights(payments, memberships, date)
    session = run_session_trainer_insights(sessions, date)

    engagement = client['client_engagement_status']
    continuation = client['activity_continuation'].iloc[0]
    coverage = financial['membership_coverage_snapshot'].iloc[0]
    retention = financial['membership_retention_30d'].iloc[0]
    period = session['session_period_comparison'].set_index('period').loc['recent']
    trainers = session['trainer_period_comparison']
    pay_attempts = financial['payment_status_90d'].set_index('payment_status')
    cash = financial['monthly_cash_performance']

    current_month = str(date.to_period('M'))
    current_cash = cash.loc[cash['payment_month'].astype(str).eq(current_month)]
    if current_cash.empty:
        month_cash = 0.0
    else:
        month_cash = float(current_cash.iloc[0]['net_cash_collected'])

    completed_cash = cash.loc[~cash['partial_month'].astype(bool)]
    if completed_cash.empty:
        latest_month, latest_cash, latest_growth = pd.NA, float('nan'), float('nan')
    else:
        latest = completed_cash.iloc[-1]
        latest_month = str(latest['payment_month'])
        latest_cash = float(latest['net_cash_collected'])
        latest_growth = _numeric_or_nan(latest['growth_pct'])

    followup = int(engagement['engagement_status'].isin([
        'inactive', 'no_recent_sessions', 'declining_activity'
    ]).sum())
    never_attended = int(engagement['engagement_status'].eq('no_session_history').sum())

    result = pd.DataFrame([{
        'as_of': date.date().isoformat(),
        'total_clients': len(roster),
        'clients_attending_30d': int(continuation['recently_active_clients']),
        'clients_needing_followup': followup,
        'clients_without_session_history': never_attended,
        'activity_continuation_pct': _numeric_or_nan(continuation['activity_continuation_pct']),
        'active_membership_clients': int(coverage['active_clients']),
        'membership_retention_30d_pct': _numeric_or_nan(retention['retention_pct']),
        'memberships_expiring_30d': int(coverage['expiring_clients_30d']),
        'scheduled_sessions_30d': int(period['scheduled_sessions']),
        'completed_sessions_30d': int(period['completed_sessions']),
        'session_completion_rate_30d_pct': _numeric_or_nan(period['completion_rate_pct']),
        'no_show_rate_30d_pct': _numeric_or_nan(period['no_show_rate_pct']),
        'completed_training_hours_30d': _numeric_or_nan(period['completed_training_hours']),
        'trainers_with_completed_sessions_30d': int((trainers['recent_completed'] > 0).sum()) if not trainers.empty else 0,
        'current_month_net_cash': month_cash,
        'current_month_partial': bool(date != date.to_period('M').end_time.normalize()),
        'last_completed_month': latest_month,
        'last_completed_month_net_cash': latest_cash,
        'last_completed_month_growth_pct': latest_growth,
        'failed_payment_attempts_90d': int(pay_attempts.loc['failed', 'transaction_count']),
        'pending_payment_attempts_90d': int(pay_attempts.loc['pending', 'transaction_count']),
    }], columns=OVERVIEW_COLUMNS)

    reports = {
        'business_overview': result,
        'business_health_insights': business_health_insights(result, no_show_threshold_pct),
        'dashboard_monthly_trends': dashboard_monthly_trends(
            client['monthly_client_activity'], cash),
    }
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(folder / f'{name}.csv', index=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Generate V2 business health reports')
    parser.add_argument('--as-of', required=True, help='Report date (YYYY-MM-DD)')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = root / 'data' / 'processed'
    clients = pd.read_csv(source / 'clients.csv')
    memberships = pd.read_csv(source / 'memberships.csv')
    sessions = pd.read_csv(source / 'sessions.csv')
    payments = pd.read_csv(source / 'payments.csv')
    reports = run_business_health(
        clients, memberships, sessions, payments, args.as_of,
        output_dir=root / 'outputs' / 'business_health' / 'v2',
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    print('Reports saved to outputs/business_health/v2')


if __name__ == '__main__':
    main()
