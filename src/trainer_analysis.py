#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Trainer Activity Analysis

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'trainers'


def load_data():
    sessions = pd.read_csv(PROCESSED_DIR / 'sessions.csv', parse_dates=['session_date'])
    payments = pd.read_csv(PROCESSED_DIR / 'payments.csv', parse_dates=['payment_date'])
    return sessions, payments


def trainer_activity(sessions):
    if sessions.empty:
        return pd.DataFrame(columns=['trainer_id', 'scheduled_sessions', 'completed_sessions',
                                     'canceled_sessions', 'no_shows', 'completion_rate_pct',
                                     'completed_training_hours', 'unique_clients'])
    data = sessions.copy()
    data['completed'] = data['session_status'].eq('completed').astype(int)
    data['canceled'] = data['session_status'].eq('canceled').astype(int)
    data['no_show'] = data['session_status'].eq('no_show').astype(int)
    data['completed_minutes'] = data['duration_minutes'].where(data['completed'].eq(1), 0)
    report = data.groupby('trainer_id', dropna=False).agg(
        scheduled_sessions=('session_id', 'count'),
        completed_sessions=('completed', 'sum'),
        canceled_sessions=('canceled', 'sum'),
        no_shows=('no_show', 'sum'),
        completed_minutes=('completed_minutes', 'sum'),
        unique_clients=('client_id', 'nunique')).reset_index()
    report['completion_rate_pct'] = 100 * report['completed_sessions'] / report['scheduled_sessions']
    report['completed_training_hours'] = report.pop('completed_minutes') / 60
    return report.sort_values('completed_training_hours', ascending=False)


def trainer_monthly_activity(sessions):
    data = sessions.loc[sessions['session_status'].eq('completed')].copy()
    data['session_month'] = data['session_date'].dt.to_period('M')
    return (data.groupby(['trainer_id', 'session_month']).agg(
        completed_sessions=('session_id', 'count'),
        completed_minutes=('duration_minutes', 'sum')).reset_index()
        .assign(completed_training_hours=lambda x: x['completed_minutes'] / 60)
        .drop(columns='completed_minutes'))


def trainer_direct_session_cash(sessions, payments):
    #Only attribute pay-per-session cash to the session trainer.
    #Membership and package receipts cannot be reliably attributed without an allocation rule.
    completed_charges = payments.loc[(payments['payment_status'] == 'completed') &
                                     (payments['payment_type'] == 'session')].copy()
    completed_charges['signed_amount'] = completed_charges['amount'].where(
        completed_charges['transaction_type'].eq('payment'), -completed_charges['amount'])
    charges = completed_charges.groupby('session_id')['signed_amount'].sum().reset_index()
    matched = charges.merge(sessions[['session_id', 'trainer_id', 'pricing_model']],
                            on='session_id', how='left', validate='one_to_one')
    if matched['trainer_id'].isna().any() or not matched['pricing_model'].eq('per_session').all():
        raise ValueError('Direct session charges cannot be matched to a per-session trainer')
    return (matched.groupby('trainer_id')['signed_amount'].sum()
            .reset_index(name='attributable_net_session_cash'))


def run_analysis(output_dir=OUTPUT_DIR):
    sessions, payments = load_data()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'trainer_activity': trainer_activity(sessions),
               'trainer_monthly_activity': trainer_monthly_activity(sessions),
               'trainer_direct_session_cash': trainer_direct_session_cash(sessions, payments)}
    for name, frame in results.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    print(f'Trainer reports saved to {output_dir}')
    return results


if __name__ == '__main__':
    run_analysis()
