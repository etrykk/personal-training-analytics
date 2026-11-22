#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4B: Historical Membership Churn Labels & Features

"""Experimental preparation of point-in-time membership cancellation examples.

This module does NOT train ML models or report churn probabilities.
Outcome: cancellation of a specific membership within (snapshot, snapshot + horizon].
Membership expiry, package nonrenewal and inactivity are NOT cancellations.

Only snapshots with a fully observed future horizon are created. An expiry
inside that horizon is censored rather than turned into a negative label.
Features use only sessions dated BEFORE the snapshot; no current client
status, cancellation reason/date, or final membership status is a feature.

Important limitations:
- Monthly snapshots from the same member are correlated. Do not randomly
  split snapshot rows for evaluation; split by time and guard client overlap.
- The V1 sample data is synthetic: results are software demonstrations, not
  evidence of real-world predictive accuracy.
- Current tables do not capture historical changes to plans, prices, or
  trainer assignments; they are intentionally omitted as model features.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'historical'

LABEL_COLUMNS = [
    'membership_id', 'client_id', 'snapshot_date', 'horizon_end',
    'churn_within_horizon', 'label_status', 'label_reason',
]

#Only these feature columns should be used when training a future model.
MODEL_FEATURE_COLUMNS = [
    'membership_tenure_days', 'client_tenure_days',
    'completed_sessions_30d', 'completed_sessions_90d',
    'scheduled_sessions_30d', 'no_shows_30d', 'canceled_sessions_30d',
    'days_since_last_completed_session',
]


def _as_date(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f'{name} must be a valid date')
    return result.normalize()


def _parse_dates(values):
    return pd.to_datetime(values, errors='coerce', format='mixed').dt.normalize()


def _validate_frame(frame, name, columns):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{name} must be a pandas DataFrame')
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} missing columns: {missing}')


def _clean_id(series):
    return series.astype('string').str.strip().replace('', pd.NA)


def build_historical_labels(memberships, as_of, horizon_days=30):
    """Generate monthly at-risk membership snapshots with observed future labels.

    A positive label requires a recorded cancellation date strictly after the
    snapshot and on/before its future horizon end. A negative label requires
    evidence of the membership remaining uncanceled throughout the horizon.
    Expiry on/before horizon end and ambiguous historical outcomes are censored.

    Snapshot dates are calendar-month starts; snapshots without a full future
    observation horizon are excluded. Only memberships that had begun and not
    ended before the snapshot are considered.
    """
    _validate_frame(memberships, 'memberships', [
        'membership_id', 'client_id', 'start_date', 'end_date',
        'status', 'cancellation_date',
    ])
    if not isinstance(horizon_days, int) or isinstance(horizon_days, bool) or horizon_days <= 0:
        raise ValueError('horizon_days must be a positive integer')
    as_of = _as_date(as_of, 'as_of')

    m = memberships.copy()
    for field in ('start_date', 'end_date', 'cancellation_date'):
        m[field] = _parse_dates(m[field])
    for field in ('membership_id', 'client_id'):
        m[field] = _clean_id(m[field])
    m['status'] = m['status'].astype('string').str.strip().str.lower().fillna('unknown')

    #Unknown dates or membership identities cannot create valid observations.
    m = m.dropna(subset=['membership_id', 'client_id', 'start_date'])
    m = m.loc[m['start_date'] <= as_of].copy()
    if m['membership_id'].duplicated().any():
        raise ValueError('membership_id must be unique')
    if m.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS).astype({'churn_within_horizon': 'Int64'})

    final_snapshot = as_of - pd.Timedelta(days=horizon_days)
    first_month = m['start_date'].min().to_period('M').to_timestamp()
    snapshots = pd.date_range(first_month, final_snapshot, freq='MS')
    result = []

    for snap in snapshots:
        horizon_end = snap + pd.Timedelta(days=horizon_days)
        for member in m.itertuples(index=False):
            membership_id = member.membership_id
            client_id = member.client_id
            started = member.start_date
            ended = member.end_date
            canceled = member.cancellation_date
            status = member.status

            #Only memberships known to be in force at snapshot date.
            if started > snap or (pd.notna(ended) and ended <= snap):
                continue
            if pd.notna(canceled) and canceled <= snap:
                continue

            label = pd.NA
            label_status = 'censored'
            reason = 'unverified_outcome'

            #Conflicting status/dates must not manufacture historical outcomes.
            if status == 'canceled' and pd.isna(canceled):
                reason = 'cancellation_date_missing'
            elif status != 'canceled' and pd.notna(canceled):
                reason = 'conflicting_cancellation_status'
            elif pd.notna(canceled) and canceled < started:
                reason = 'cancellation_before_start'
            elif pd.notna(ended) and ended < started:
                reason = 'end_before_start'
            elif pd.notna(canceled) and pd.notna(ended) and ended < canceled:
                reason = 'membership_end_before_cancellation'
            elif pd.notna(canceled) and snap < canceled <= horizon_end:
                label = 1
                label_status = 'positive'
                reason = 'canceled_within_horizon'
            elif pd.notna(ended) and ended <= horizon_end:
                #Expiration within the horizon is a distinct event, not a 0.
                reason = 'membership_ended_before_horizon'
            elif status == 'expired' and pd.isna(ended):
                reason = 'expiration_date_missing'
            elif status in ('active', 'expired', 'canceled'):
                #An observed active status or a later end/cancellation date
                #establishes an uncanceled full horizon by the extraction date.
                label = 0
                label_status = 'negative'
                reason = 'no_cancellation_observed_in_horizon'
            else:
                reason = 'unknown_membership_status'

            result.append({
                'membership_id': membership_id,
                'client_id': client_id,
                'snapshot_date': snap,
                'horizon_end': horizon_end,
                'churn_within_horizon': label,
                'label_status': label_status,
                'label_reason': reason,
            })

    labels = pd.DataFrame(result, columns=LABEL_COLUMNS)
    labels['churn_within_horizon'] = labels['churn_within_horizon'].astype('Int64')
    return labels.sort_values(['snapshot_date', 'membership_id']).reset_index(drop=True)


def build_historical_features(clients, memberships, sessions, labeled_snapshots):
    """Add leakage-safe historical engagement features to the label table.

    Windows are half-open [snapshot - N days, snapshot). A session on the
    snapshot date itself is excluded because its final status might not yet
    be known at prediction time.
    """
    _validate_frame(clients, 'clients', ['client_id', 'join_date'])
    _validate_frame(memberships, 'memberships', ['membership_id', 'start_date'])
    _validate_frame(sessions, 'sessions', ['client_id', 'session_date', 'session_status'])
    _validate_frame(labeled_snapshots, 'labeled_snapshots', LABEL_COLUMNS)

    c = clients[['client_id', 'join_date']].copy()
    m = memberships[['membership_id', 'start_date']].copy()
    s = sessions[['client_id', 'session_date', 'session_status']].copy()
    for table, column in ((c, 'client_id'), (m, 'membership_id'), (s, 'client_id')):
        table[column] = _clean_id(table[column])
    c['join_date'] = _parse_dates(c['join_date'])
    m['start_date'] = _parse_dates(m['start_date'])
    s['session_date'] = _parse_dates(s['session_date'])
    s['session_status'] = s['session_status'].astype('string').str.strip().str.lower()
    if c['client_id'].duplicated().any() or m['membership_id'].duplicated().any():
        raise ValueError('client_id and membership_id must be unique in their source datasets')

    result = labeled_snapshots.copy()
    result['snapshot_date'] = _parse_dates(result['snapshot_date'])
    result = result.merge(c, on='client_id', how='left', validate='many_to_one')
    result = result.merge(m, on='membership_id', how='left', validate='many_to_one')

    rows = []
    for member in result.itertuples(index=False):
        snap = member.snapshot_date
        related = s.loc[(s['client_id'] == member.client_id) &
                        (s['session_date'].notna()) & (s['session_date'] < snap)]
        recent30 = related.loc[related['session_date'] >= snap - pd.Timedelta(days=30)]
        recent90 = related.loc[related['session_date'] >= snap - pd.Timedelta(days=90)]
        completed = related.loc[related['session_status'].eq('completed')]
        last_session = completed['session_date'].max() if not completed.empty else pd.NaT
        rows.append({
            'membership_tenure_days': int((snap - member.start_date).days)
                if pd.notna(member.start_date) else pd.NA,
            'client_tenure_days': int((snap - member.join_date).days)
                if pd.notna(member.join_date) and member.join_date <= snap else pd.NA,
            'completed_sessions_30d': int(recent30['session_status'].eq('completed').sum()),
            'completed_sessions_90d': int(recent90['session_status'].eq('completed').sum()),
            'scheduled_sessions_30d': int(len(recent30)),
            'no_shows_30d': int(recent30['session_status'].eq('no_show').sum()),
            'canceled_sessions_30d': int(recent30['session_status'].eq('canceled').sum()),
            'days_since_last_completed_session': int((snap - last_session).days)
                if pd.notna(last_session) else pd.NA,
        })
    for col in MODEL_FEATURE_COLUMNS:
        result[col] = pd.Series([row[col] for row in rows], index=result.index, dtype='Int64')
    return result.drop(columns=['join_date', 'start_date']).loc[
        :, LABEL_COLUMNS + MODEL_FEATURE_COLUMNS]


def build_historical_churn_dataset(clients, memberships, sessions, as_of,
                                   horizon_days=30):
    """Convenience wrapper returning labeled historical features."""
    labels = build_historical_labels(memberships, as_of, horizon_days=horizon_days)
    return build_historical_features(clients, memberships, sessions, labels)


def summarize_historical_labels(snapshots):
    """Summarize candidate labels; snapshot counts are NOT independent events."""
    _validate_frame(snapshots, 'snapshots', LABEL_COLUMNS)
    total = len(snapshots)
    result = {
        'total_snapshots': total,
        'labeled_snapshots': int(snapshots['churn_within_horizon'].notna().sum()),
        'positive_snapshots': int(snapshots['churn_within_horizon'].eq(1).fillna(False).sum()),
        'negative_snapshots': int(snapshots['churn_within_horizon'].eq(0).fillna(False).sum()),
        'censored_snapshots': int(snapshots['churn_within_horizon'].isna().sum()),
        'unique_memberships': int(snapshots['membership_id'].nunique()),
        'positive_unique_memberships': int(snapshots.loc[
            snapshots['churn_within_horizon'].eq(1).fillna(False), 'membership_id'].nunique()),
        'unique_clients': int(snapshots['client_id'].nunique()),
        'training_approved': False,
    }
    return pd.DataFrame([result])


def run_analysis(as_of='2026-11-20', horizon_days=30, data_dir=PROCESSED_DIR,
                 output_dir=OUTPUT_DIR):
    """Write historical examples and a summary for further review."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    clients = pd.read_csv(data_dir / 'clients.csv')
    memberships = pd.read_csv(data_dir / 'memberships.csv')
    sessions = pd.read_csv(data_dir / 'sessions.csv')
    snapshots = build_historical_churn_dataset(
        clients, memberships, sessions, as_of, horizon_days=horizon_days)
    summary = summarize_historical_labels(snapshots)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(output_dir / 'historical_churn_snapshots.csv', index=False)
    summary.to_csv(output_dir / 'historical_churn_summary.csv', index=False)
    print('Experimental historical churn reports saved to:', output_dir)
    print(summary.to_string(index=False))
    return {'snapshots': snapshots, 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build historical membership cancellation snapshots')
    parser.add_argument('--as-of', default='2026-11-20', help='Fully observed dataset cutoff date')
    parser.add_argument('--horizon-days', type=int, default=30)
    args = parser.parse_args()
    run_analysis(as_of=args.as_of, horizon_days=args.horizon_days)
