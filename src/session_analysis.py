#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Session and Package Analysis

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'sessions'


def load_data():
    sessions = pd.read_csv(PROCESSED_DIR / 'sessions.csv', parse_dates=['session_date'])
    packages = pd.read_csv(PROCESSED_DIR / 'packages.csv', parse_dates=['purchase_date', 'expiration_date'])
    return sessions, packages


def attendance_summary(sessions):
    counts = sessions.groupby('session_status').size().reset_index(name='sessions')
    total = len(sessions)
    counts['pct_of_scheduled'] = 100 * counts['sessions'] / total if total else 0.0
    return counts


def monthly_attendance(sessions):
    data = sessions.copy()
    data['session_month'] = data['session_date'].dt.to_period('M')
    result = (data.groupby(['session_month', 'session_status']).size()
              .unstack(fill_value=0).reindex(columns=['completed', 'canceled', 'no_show'], fill_value=0))
    result['scheduled'] = result.sum(axis=1)
    result['completion_rate_pct'] = (100 * result['completed'] / result['scheduled']).where(result['scheduled'] > 0)
    return result.reset_index()


def activity_by_pricing_model(sessions):
    return (sessions.groupby(['pricing_model', 'session_status']).size()
            .unstack(fill_value=0).reset_index())


def package_redemption(packages):
    result = packages[['package_id', 'client_id', 'package_sessions',
                       'sessions_redeemed', 'sessions_remaining', 'status']].copy()
    result['redemption_rate_pct'] = (100 * result['sessions_redeemed'] / result['package_sessions']).where(result['package_sessions'] > 0)
    return result


def run_analysis(output_dir=OUTPUT_DIR):
    sessions, packages = load_data()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'attendance_summary': attendance_summary(sessions),
               'monthly_attendance': monthly_attendance(sessions),
               'activity_by_pricing_model': activity_by_pricing_model(sessions),
               'package_redemption': package_redemption(packages)}
    for name, frame in results.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    print(f'Session reports saved to {output_dir}')
    return results


if __name__ == '__main__':
    run_analysis()
