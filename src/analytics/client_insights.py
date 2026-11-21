#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Client Activity Continuation and Business Insights

from pathlib import Path
import argparse

import pandas as pd

from .client_engagement import (
    client_activity_summary,
    attendance_consistency,
    client_engagement_status,
)


MONTHLY_COLUMNS = [
    'month', 'eligible_clients', 'clients_with_completed_sessions',
    'completed_sessions', 'monthly_activity_rate_pct'
]
CONTINUATION_COLUMNS = [
    'total_clients', 'previously_active_clients', 'recently_active_clients',
    'active_in_both_periods', 'stopped_attending', 'newly_attending',
    'inactive_in_both_periods', 'activity_continuation_pct'
]
INSIGHT_COLUMNS = ['priority', 'category', 'headline', 'description', 'client_count']


def monthly_client_activity(clients, sessions, as_of, months=6):
    '''
    Summarize recorded client training activity by calendar month.

    The denominator includes every client in the supplied roster whose
    join_date is on or before that month's observed end date. This is NOT
    active membership retention or historical contract status.

    The last month is partial if as_of is before its calendar-month end.
    A percentage is unavailable (NA) when no clients are eligible.
    '''
    if not isinstance(months, int) or isinstance(months, bool) or months < 1:
        raise ValueError('months must be a positive integer')
    if not {'client_id', 'join_date'}.issubset(clients.columns):
        raise ValueError('clients must contain client_id and join_date')
    if not {'client_id', 'session_date', 'session_status'}.issubset(sessions.columns):
        raise ValueError('sessions must contain client_id, session_date, and session_status')
    if clients['client_id'].isna().any() or clients['client_id'].duplicated().any():
        raise ValueError('client_id must be nonmissing and unique')

    as_of = pd.Timestamp(as_of)
    if pd.isna(as_of):
        raise ValueError('as_of must be a valid reporting date')
    as_of = as_of.normalize()

    roster = clients[['client_id', 'join_date']].copy()
    roster['join_date'] = pd.to_datetime(roster['join_date'], errors='coerce').dt.normalize()
    if roster['join_date'].isna().any():
        raise ValueError('join_date must contain valid dates')

    activity = sessions[['client_id', 'session_date', 'session_status']].copy()
    activity['session_date'] = pd.to_datetime(activity['session_date'], errors='coerce').dt.normalize()
    activity = activity.loc[
        activity['session_status'].astype('string').str.strip().str.lower().eq('completed')
        & activity['session_date'].notna()
        & (activity['session_date'] <= as_of)
    ]

    first_month = as_of.to_period('M') - (months - 1)
    rows = []
    for period in pd.period_range(first_month, as_of.to_period('M'), freq='M'):
        month_start = period.to_timestamp()
        observed_end = min(period.end_time.normalize(), as_of)
        eligible_ids = set(roster.loc[roster['join_date'] <= observed_end, 'client_id'])
        observed = activity.loc[
            activity['session_date'].between(month_start, observed_end)
            & activity['client_id'].isin(eligible_ids)
        ]
        count_eligible = len(eligible_ids)
        count_active = observed['client_id'].nunique()
        rows.append({
            'month': str(period),
            'eligible_clients': count_eligible,
            'clients_with_completed_sessions': int(count_active),
            'completed_sessions': len(observed),
            'monthly_activity_rate_pct': (
                100.0 * count_active / count_eligible if count_eligible else pd.NA
            ),
        })

    return pd.DataFrame(rows, columns=MONTHLY_COLUMNS)


def activity_continuation_summary(attendance_summary):
    '''
    Compare the prior and recent 30-day activity periods from V2.

    Continuation means a client with >=1 completed session in the previous
    30-day window also has >=1 completed session in the recent window.
    It does NOT mean subscription retention or predicted non-churn.
    '''
    required = {'client_id', 'previous_30d', 'recent_30d'}
    if not required.issubset(attendance_summary.columns):
        raise ValueError('attendance_summary is missing required columns')
    if (attendance_summary['client_id'].isna().any()
            or attendance_summary['client_id'].duplicated().any()):
        raise ValueError('client_id must be nonmissing and unique')

    previous = pd.to_numeric(attendance_summary['previous_30d'], errors='coerce')
    recent = pd.to_numeric(attendance_summary['recent_30d'], errors='coerce')
    for counts in (previous, recent):
        if counts.isna().any() or ((counts < 0) | (counts % 1 != 0)).any():
            raise ValueError('session counts must be nonnegative whole numbers')

    previously_active = previous > 0
    recently_active = recent > 0
    both = int((previously_active & recently_active).sum())
    previous_total = int(previously_active.sum())
    return pd.DataFrame([{
        'total_clients': len(attendance_summary),
        'previously_active_clients': previous_total,
        'recently_active_clients': int(recently_active.sum()),
        'active_in_both_periods': both,
        'stopped_attending': int((previously_active & ~recently_active).sum()),
        'newly_attending': int((~previously_active & recently_active).sum()),
        'inactive_in_both_periods': int((~previously_active & ~recently_active).sum()),
        'activity_continuation_pct': (
            100.0 * both / previous_total if previous_total else pd.NA
        ),
    }], columns=CONTINUATION_COLUMNS)


def generate_engagement_insights(engagement_report, continuation_summary):
    '''
    Create transparent, rule-based business insights without ML predictions.
    Each engagement category is mutually exclusive, and an indicator is
    not proof that a client has canceled or will cancel.
    '''
    if not {'client_id', 'engagement_status'}.issubset(engagement_report.columns):
        raise ValueError('engagement_report is missing required columns')
    if engagement_report['client_id'].isna().any() or engagement_report['client_id'].duplicated().any():
        raise ValueError('client_id must be nonmissing and unique')
    if len(continuation_summary) != 1 or not {
        'previously_active_clients', 'active_in_both_periods', 'activity_continuation_pct'
    }.issubset(continuation_summary.columns):
        raise ValueError('continuation_summary must contain one summary row')

    category_specs = [
        ('inactive', 'review', 'Review clients with prolonged inactivity',
         'These clients have completed sessions in the past but have exceeded the inactivity threshold.'),
        ('declining_activity', 'review', 'Review declining training frequency',
         'These clients completed fewer sessions in the recent 30-day period than in the previous period.'),
        ('no_recent_sessions', 'monitor', 'Follow up on recent attendance gaps',
         'These clients have completed training before but no sessions in the recent 30-day period.'),
        ('no_session_history', 'info', 'Check clients without recorded session history',
         'These clients have no completed sessions in the supplied history as of the reporting date.'),
    ]
    rows = []
    counts = engagement_report['engagement_status'].value_counts()
    for category, priority, headline, description in category_specs:
        n = int(counts.get(category, 0))
        if n:
            rows.append({
                'priority': priority, 'category': category,
                'headline': headline, 'description': description,
                'client_count': n,
            })

    previous_total = int(continuation_summary.iloc[0]['previously_active_clients'])
    if previous_total:
        n = int(continuation_summary.iloc[0]['active_in_both_periods'])
        pct = 100.0 * n / previous_total
        rows.append({
            'priority': 'info', 'category': 'activity_continuation',
            'headline': 'Completed-session activity continuation',
            'description': (
                f'{n} of {previous_total} previously attending clients also completed a session '
                f'in the recent 30-day period ({pct:.1f}%). This is not membership retention.'
            ),
            'client_count': n,
        })
    if not rows:
        rows.append({
            'priority': 'info', 'category': 'no_signals',
            'headline': 'No follow-up indicators identified',
            'description': ('No defined engagement category triggered a review indicator. '
                            'This does not guarantee continued client retention.'),
            'client_count': 0,
        })
    return pd.DataFrame(rows, columns=INSIGHT_COLUMNS)


def run_client_insights(clients, sessions, as_of, months=6, output_dir=None):
    '''Build all Phase 2B reports; optionally save each report as a CSV file.'''
    activity = client_activity_summary(clients, sessions, as_of)
    attendance = attendance_consistency(clients, sessions, as_of)
    engagement = client_engagement_status(activity, attendance)
    continuation = activity_continuation_summary(attendance)

    reports = {
        'monthly_client_activity': monthly_client_activity(clients, sessions, as_of, months),
        'activity_continuation': continuation,
        'client_engagement_status': engagement,
        'engagement_insights': generate_engagement_insights(engagement, continuation),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(output_dir / f'{name}.csv', index=False)
    return reports


def main():
    #Optional command-line report generation using the existing processed CSVs
    parser = argparse.ArgumentParser(description='Generate V2 client activity insights')
    parser.add_argument('--as-of', required=True, help='Report date (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=6, help='Months of activity history')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source = project_root / 'data' / 'processed'
    clients = pd.read_csv(source / 'clients.csv')
    sessions = pd.read_csv(source / 'sessions.csv')
    reports = run_client_insights(
        clients, sessions, as_of=args.as_of, months=args.months,
        output_dir=project_root / 'outputs' / 'clients' / 'v2',
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    print('Reports saved to outputs/clients/v2')


if __name__ == '__main__':
    main()
