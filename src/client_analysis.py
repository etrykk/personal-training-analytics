#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Client and Membership Analysis

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'clients'
REFERENCE_DATE = pd.Timestamp('2026-11-20')


def load_data():
    clients = pd.read_csv(PROCESSED_DIR / 'clients.csv', parse_dates=['join_date'])
    memberships = pd.read_csv(PROCESSED_DIR / 'memberships.csv', parse_dates=['start_date', 'end_date', 'cancellation_date'])
    sessions = pd.read_csv(PROCESSED_DIR / 'sessions.csv', parse_dates=['session_date'])
    return clients, memberships, sessions


def client_acquisition(clients):
    data = clients.copy()
    data['join_month'] = data['join_date'].dt.to_period('M')
    return (data.groupby('join_month').size().rename('new_clients')
            .reset_index().sort_values('join_month'))


def client_status_summary(clients):
    #Snapshot only: status is not a historical churn measure
    return clients.groupby('status', dropna=False).size().reset_index(name='clients')


def membership_cancellations(memberships):
    canceled = memberships.loc[memberships['status'].eq('canceled')].copy()
    canceled['cancellation_month'] = canceled['cancellation_date'].dt.to_period('M')
    return (canceled.groupby('cancellation_month').size()
            .reset_index(name='cancellations').sort_values('cancellation_month'))


def membership_cohort_retention(memberships, as_of=REFERENCE_DATE):
    #A membership is retained at month N if still in force at its N-month anniversary.
    #Only mature cohorts are included in each month's denominator.
    #This is contract retention, not overall client retention.
    as_of = pd.Timestamp(as_of)
    records = memberships.dropna(subset=['membership_id', 'start_date']).copy()
    records = records.loc[records['start_date'] <= as_of]
    if records.empty:
        return pd.DataFrame(columns=['cohort_month', 'months_since_start', 'eligible_memberships', 'retained_memberships', 'retention_pct'])
    records['cohort_month'] = records['start_date'].dt.to_period('M')
    records['effective_end'] = records['end_date'].fillna(as_of)
    if 'cancellation_date' in records:
        records['effective_end'] = records[['effective_end', 'cancellation_date']].min(axis=1, skipna=True)
    rows = []
    for cohort, group in records.groupby('cohort_month'):
        for age in range(0, 13):
            anniversaries = group['start_date'] + pd.DateOffset(months=age)
            eligible = anniversaries <= as_of
            count = int(eligible.sum())
            if not count:
                continue
            retained = int((eligible & (group['effective_end'] >= anniversaries)).sum())
            rows.append({'cohort_month': cohort, 'months_since_start': age,
                         'eligible_memberships': count, 'retained_memberships': retained,
                         'retention_pct': 100 * retained / count})
    return pd.DataFrame(rows)


def pay_per_session_inactivity(sessions, as_of=REFERENCE_DATE, inactivity_days=60):
    #An inactivity proxy, NOT proven churn: clients may train elsewhere or irregularly
    as_of = pd.Timestamp(as_of)
    subset = sessions.loc[(sessions['pricing_model'] == 'per_session') &
                          (sessions['session_status'] == 'completed') &
                          (sessions['session_date'] <= as_of)]
    last = subset.groupby('client_id')['session_date'].max().reset_index(name='last_completed_session')
    last['days_since_last_session'] = (as_of - last['last_completed_session']).dt.days
    last['inactive_60d_proxy'] = last['days_since_last_session'] > inactivity_days
    return last


def run_analysis(output_dir=OUTPUT_DIR):
    clients, memberships, sessions = load_data()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'client_acquisition': client_acquisition(clients),
               'client_status_summary': client_status_summary(clients),
               'membership_cancellations': membership_cancellations(memberships),
               'membership_cohort_retention': membership_cohort_retention(memberships),
               'pay_per_session_inactivity': pay_per_session_inactivity(sessions)}
    for name, frame in results.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    print(f'Client reports saved to {output_dir}')
    return results


if __name__ == '__main__':
    run_analysis()
