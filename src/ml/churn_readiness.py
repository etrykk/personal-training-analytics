#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4A: Experimental ML Data Readiness

"""Audit whether historical membership data can support churn-label development.

This module DOES NOT train a model, calculate churn probabilities, or label
inactive clients as churned. A membership cancellation is a different event
from membership expiration, package nonrenewal, or lack of attendance.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'readiness'

REQUIRED_FIELDS = {
    'clients': {'client_id', 'join_date'},
    'memberships': {'membership_id', 'client_id', 'start_date', 'status',
                    'cancellation_date'},
    'sessions': {'client_id', 'session_date', 'session_status'},
}


def _clean_ids(values):
    #Normalize identifiers for unique counts and relationship checks
    return values.astype('string').str.strip().replace('', pd.NA)


def _dates(values):
    #Invalid dates become NaT and are reported instead of silently accepted
    return pd.to_datetime(values, errors='coerce', format='mixed')


def _check(name, passed, observed, requirement, details):
    return {
        'check': name,
        'passed': bool(passed),
        'observed': observed,
        'requirement': requirement,
        'details': details,
    }


def assess_churn_readiness(clients, memberships, sessions, as_of,
                           horizon_days=30, min_events=30,
                           min_history_days=180):
    """Return (summary, checks, metrics) DataFrames for membership churn readiness.

    All figures are *descriptive*. Even a passed screen does not establish
    historical negative labels, justify training, or validate a 30-day model.
    Historical point-in-time snapshots must be constructed in a later phase.

    The min_events/min_history_days parameters are configurable screening
    thresholds, not statistical guarantees of model performance.
    """
    datasets = {'clients': clients, 'memberships': memberships, 'sessions': sessions}
    for name, frame in datasets.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f'{name} must be a pandas DataFrame')
        absent = sorted(REQUIRED_FIELDS[name].difference(frame.columns))
        if absent:
            raise ValueError(f'{name} is missing required column(s): {absent}')

    report_date = pd.Timestamp(as_of).normalize()
    if pd.isna(report_date):
        raise ValueError('as_of must be a valid reporting date')
    if not isinstance(horizon_days, int) or horizon_days < 1:
        raise ValueError('horizon_days must be a positive integer')
    if not isinstance(min_events, int) or min_events < 1:
        raise ValueError('min_events must be a positive integer')
    if not isinstance(min_history_days, int) or min_history_days < 1:
        raise ValueError('min_history_days must be a positive integer')

    #Work on copies: never change the V1/V2 analytics inputs
    c = clients.copy()
    m = memberships.copy()
    s = sessions.copy()
    c['client_id'] = _clean_ids(c['client_id'])
    m['client_id'] = _clean_ids(m['client_id'])
    m['membership_id'] = _clean_ids(m['membership_id'])
    s['client_id'] = _clean_ids(s['client_id'])

    c['join_date'] = _dates(c['join_date'])
    m['start_date'] = _dates(m['start_date'])
    m['cancellation_date'] = _dates(m['cancellation_date'])
    s['session_date'] = _dates(s['session_date'])
    m['status'] = m['status'].astype('string').str.strip().str.lower()
    s['session_status'] = s['session_status'].astype('string').str.strip().str.lower()

    #Only historical memberships can be used in the audit as of the report date.
    observed_memberships = m.loc[m['start_date'].notna() &
                                 (m['start_date'] <= report_date)]
    canceled = observed_memberships.loc[observed_memberships['status'].eq('canceled')]
    valid_cancellations = canceled.loc[
        canceled['cancellation_date'].notna() &
        (canceled['cancellation_date'] <= report_date) &
        (canceled['cancellation_date'] >= canceled['start_date'])
    ]
    full_horizon_history = (
        (valid_cancellations['cancellation_date'] - valid_cancellations['start_date'])
        .dt.days >= horizon_days
    )

    #This count is NOT a labeled negative class. A non-canceled status at the
    #reporting date cannot establish a historical 30-day non-churn outcome.
    noncanceled_memberships = int(observed_memberships['status'].ne('canceled').fillna(False).sum())
    expired = int(observed_memberships['status'].eq('expired').sum())
    active = int(observed_memberships['status'].eq('active').sum())
    missing_cancellation_dates = int(canceled['cancellation_date'].isna().sum())
    invalid_cancellation_order = int((canceled['cancellation_date'].notna() &
                                     (canceled['cancellation_date'] < canceled['start_date'])).sum())
    future_cancellations = int((canceled['cancellation_date'] > report_date).sum())
    duplicate_membership_ids = int((m['membership_id'].notna() &
                                    m['membership_id'].duplicated(keep=False)).sum())
    valid_clients = c.loc[c['join_date'].notna() & (c['join_date'] <= report_date)]
    client_ids = set(valid_clients['client_id'].dropna())
    unknown_membership_clients = int((observed_memberships['client_id'].notna() &
                                      ~observed_memberships['client_id'].isin(client_ids)).sum())
    usable_completed = s.loc[s['session_date'].notna() &
                             (s['session_date'] <= report_date) &
                             s['session_status'].eq('completed')]
    dated_sessions = s.loc[s['session_date'].notna() & (s['session_date'] <= report_date)]
    if dated_sessions.empty:
        history_days = 0
    else:
        history_days = int((dated_sessions['session_date'].max() -
                            dated_sessions['session_date'].min()).days)

    metrics = {
        'total_client_records': len(c),
        'clients_joined_by_as_of': int(valid_clients['client_id'].nunique()),
        'total_membership_records': len(m),
        'membership_records_started_by_as_of': len(observed_memberships),
        'membership_clients_by_as_of': int(observed_memberships['client_id'].nunique()),
        'recorded_cancellations_by_as_of': len(valid_cancellations),
        'canceled_clients_by_as_of': int(valid_cancellations['client_id'].nunique()),
        'cancellations_with_full_horizon_membership_history': int(full_horizon_history.sum()),
        'expired_memberships_by_as_of_not_churn_labels': expired,
        'active_memberships_by_as_of': active,
        'noncanceled_membership_records_not_negative_labels': noncanceled_memberships,
        'canceled_memberships_missing_date': missing_cancellation_dates,
        'cancellations_before_membership_start': invalid_cancellation_order,
        'cancellations_after_as_of': future_cancellations,
        'duplicate_membership_id_rows': duplicate_membership_ids,
        'unknown_membership_client_rows': unknown_membership_clients,
        'invalid_client_join_date_rows': int(c['join_date'].isna().sum()),
        'invalid_membership_start_date_rows': int(m['start_date'].isna().sum()),
        'invalid_session_date_rows': int(s['session_date'].isna().sum()),
        'completed_sessions_through_as_of': len(usable_completed),
        'clients_with_completed_sessions_through_as_of': int(usable_completed['client_id'].nunique()),
        'observed_session_history_days': history_days,
    }

    checks = [
        _check('historical_cancellation_events',
               metrics['recorded_cancellations_by_as_of'] >= min_events,
               metrics['recorded_cancellations_by_as_of'], f'>= {min_events}',
               'Cancellation events are membership cancellations only, not expirations or inactivity.'),
        _check('full_horizon_cancellation_examples',
               metrics['cancellations_with_full_horizon_membership_history'] >= min_events,
               metrics['cancellations_with_full_horizon_membership_history'], f'>= {min_events}',
               f'Cancellations with at least {horizon_days} days of membership tenure beforehand; '
               'shorter tenures may be useful under a different snapshot design.'),
        _check('session_history_span', history_days >= min_history_days,
               history_days, f'>= {min_history_days} days',
               'A long session history does not guarantee enough labeled cancellation events.'),
        _check('cancellation_date_completeness', missing_cancellation_dates == 0,
               missing_cancellation_dates, '0 missing',
               'Canceled memberships require an actual event date.'),
        _check('cancellation_date_order', invalid_cancellation_order == 0,
               invalid_cancellation_order, '0 out-of-order',
               'Cancellation must not precede membership start.'),
        _check('unique_membership_ids', duplicate_membership_ids == 0,
               duplicate_membership_ids, '0 duplicate rows',
               'Avoid double counting outcomes.'),
        _check('member_client_relationships', unknown_membership_clients == 0,
               unknown_membership_clients, '0 unknown clients',
               'Every membership should reference a client who joined by the reporting date.'),
        _check('required_event_dates_parse',
               metrics['invalid_client_join_date_rows'] == 0 and
               metrics['invalid_membership_start_date_rows'] == 0 and
               metrics['invalid_session_date_rows'] == 0,
               (metrics['invalid_client_join_date_rows'] +
                metrics['invalid_membership_start_date_rows'] +
                metrics['invalid_session_date_rows']),
               '0 invalid dates', 'Invalid or blank required dates reduce the usable history.'),
    ]
    checks_df = pd.DataFrame(checks, columns=['check', 'passed', 'observed', 'requirement', 'details'])
    passed_basic_screen = bool(checks_df['passed'].all())

    #Passing a basic screen is NOT authorization to train or deploy. Historical
    #point-in-time labeled examples and holdout evaluation are still required.
    summary = pd.DataFrame([{
        'as_of': report_date.date().isoformat(),
        'prediction_horizon_days': horizon_days,
        'target_event': 'membership cancellation within a future prediction window',
        'basic_readiness_screen': 'passed' if passed_basic_screen else 'insufficient_data_or_quality',
        'training_approved': False,
        'next_step': ('Design historical time-aware feature/label snapshots before training'
                      if passed_basic_screen else
                      'Resolve failed checks; do not train a business-specific churn model'),
    }])
    metrics_df = pd.DataFrame([
        {'metric': name, 'value': value} for name, value in metrics.items()
    ])
    return summary, checks_df, metrics_df


def run_readiness_report(as_of='2026-11-20', horizon_days=30,
                         processed_dir=PROCESSED_DIR, output_dir=OUTPUT_DIR,
                         min_events=30, min_history_days=180):
    """Load V1 processed data and export three CSV audit reports."""
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    tables = {name: pd.read_csv(processed_dir / f'{name}.csv')
              for name in ('clients', 'memberships', 'sessions')}
    summary, checks, metrics = assess_churn_readiness(
        tables['clients'], tables['memberships'], tables['sessions'], as_of,
        horizon_days=horizon_days, min_events=min_events,
        min_history_days=min_history_days)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / 'churn_readiness_decision.csv', index=False)
    checks.to_csv(output_dir / 'churn_readiness_checks.csv', index=False)
    metrics.to_csv(output_dir / 'churn_readiness_metrics.csv', index=False)
    print('Churn readiness reports saved to:', output_dir)
    print(summary.to_string(index=False))
    return {'summary': summary, 'checks': checks, 'metrics': metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit historical membership churn data readiness')
    parser.add_argument('--as-of', default='2026-11-20', help='Reporting date, YYYY-MM-DD')
    parser.add_argument('--horizon-days', type=int, default=30)
    parser.add_argument('--min-events', type=int, default=30)
    args = parser.parse_args()
    run_readiness_report(as_of=args.as_of, horizon_days=args.horizon_days,
                         min_events=args.min_events)
