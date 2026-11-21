#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Advanced Session and Trainer Insights

from pathlib import Path
import argparse

import pandas as pd


SESSION_STATUSES = ('completed', 'canceled', 'no_show')
WEEKLY_COLUMNS = [
    'week_start', 'scheduled_sessions', 'completed_sessions',
    'canceled_sessions', 'no_shows', 'completion_rate_pct',
    'no_show_rate_pct', 'completed_training_hours', 'partial_week'
]
WEEKDAY_COLUMNS = [
    'weekday', 'scheduled_sessions', 'completed_sessions',
    'canceled_sessions', 'no_shows', 'completion_rate_pct'
]
PERIOD_COLUMNS = [
    'period', 'period_start', 'period_end', 'scheduled_sessions',
    'completed_sessions', 'canceled_sessions', 'no_shows',
    'completion_rate_pct', 'no_show_rate_pct', 'completed_training_hours'
]
TRAINER_COLUMNS = [
    'trainer_id', 'previous_scheduled', 'recent_scheduled',
    'previous_completed', 'recent_completed', 'completed_change',
    'previous_no_shows', 'recent_no_shows', 'previous_unique_clients',
    'recent_unique_clients', 'previous_training_hours',
    'recent_training_hours', 'recent_completion_rate_pct'
]


def _as_of_date(value):
    #Normalize reporting dates for consistent calendar-day comparisons.
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError('Reporting date must be valid')
    return date.normalize()


def _validate_days(days):
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError('days must be a positive integer')


def _prepare_sessions(sessions, as_of):
    #Use historical session outcomes only; never treat future sessions as completed.
    if not isinstance(sessions, pd.DataFrame):
        raise TypeError('sessions must be a pandas DataFrame')
    required = {'session_date', 'session_status', 'client_id'}
    missing = required - set(sessions.columns)
    if missing:
        raise ValueError(f'Missing session columns: {sorted(missing)}')

    data = sessions.copy()
    data['session_date'] = pd.to_datetime(data['session_date'], errors='raise').dt.normalize()
    if data['session_date'].isna().any():
        raise ValueError('Session dates cannot be missing')
    data = data.loc[data['session_date'] <= _as_of_date(as_of)].copy()
    data['session_status'] = data['session_status'].astype('string').str.strip().str.lower()
    if not data['session_status'].isin(SESSION_STATUSES).all():
        raise ValueError('Historical sessions must have a recognized outcome')
    if data['client_id'].isna().any():
        raise ValueError('Historical sessions must have client IDs')

    #Some imported systems lack duration or trainer information.
    #Keep those sessions in counts without inventing training hours or trainer IDs.
    if 'trainer_id' not in data.columns:
        data['trainer_id'] = 'Unassigned'
    else:
        data['trainer_id'] = data['trainer_id'].astype('string').str.strip().replace('', pd.NA).fillna('Unassigned')

    if 'duration_minutes' not in data.columns:
        data['duration_minutes'] = float('nan')
    else:
        data['duration_minutes'] = pd.to_numeric(data['duration_minutes'], errors='coerce')
        invalid = data['session_status'].eq('completed') & (
            data['duration_minutes'].notna() & (data['duration_minutes'] <= 0)
        )
        if invalid.any():
            raise ValueError('Completed session durations must be positive')
    return data


def _in_window(data, start, end):
    return data.loc[data['session_date'].between(start, end)].copy()


def _completed_hours(data):
    #Return unavailable when recorded completed sessions lack duration values.
    completed = data.loc[data['session_status'].eq('completed'), 'duration_minutes']
    if completed.empty:
        return 0.0
    if completed.isna().any():
        return float('nan')
    return float(completed.sum() / 60)


def _counts(data):
    counts = data['session_status'].value_counts()
    return {key: int(counts.get(key, 0)) for key in SESSION_STATUSES}


def _rate(numerator, denominator):
    return 100 * numerator / denominator if denominator else float('nan')


def weekly_session_trends(sessions, as_of, days=90):
    '''Weekly attendance outcomes for the most recent `days` calendar dates.'''
    _validate_days(days)
    end = _as_of_date(as_of)
    start = end - pd.Timedelta(days=days - 1)
    data = _in_window(_prepare_sessions(sessions, end), start, end)
    first_monday = start - pd.Timedelta(days=start.weekday())
    mondays = pd.date_range(first_monday, end, freq='W-MON')
    rows = []
    for monday in mondays:
        sunday = monday + pd.Timedelta(days=6)
        group = _in_window(data, monday, sunday)
        counts = _counts(group)
        scheduled = len(group)
        rows.append({
            'week_start': monday,
            'scheduled_sessions': scheduled,
            'completed_sessions': counts['completed'],
            'canceled_sessions': counts['canceled'],
            'no_shows': counts['no_show'],
            'completion_rate_pct': _rate(counts['completed'], scheduled),
            'no_show_rate_pct': _rate(counts['no_show'], scheduled),
            'completed_training_hours': _completed_hours(group),
            'partial_week': bool(monday < start or sunday > end),
        })
    return pd.DataFrame(rows, columns=WEEKLY_COLUMNS)


def weekday_session_patterns(sessions, as_of, days=90):
    '''Summarize outcomes by weekday; includes days with zero sessions.'''
    _validate_days(days)
    end = _as_of_date(as_of)
    start = end - pd.Timedelta(days=days - 1)
    data = _in_window(_prepare_sessions(sessions, end), start, end)
    data['weekday_number'] = data['session_date'].dt.weekday
    names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rows = []
    for number, name in enumerate(names):
        group = data.loc[data['weekday_number'].eq(number)]
        counts = _counts(group)
        scheduled = len(group)
        rows.append({
            'weekday': name,
            'scheduled_sessions': scheduled,
            'completed_sessions': counts['completed'],
            'canceled_sessions': counts['canceled'],
            'no_shows': counts['no_show'],
            'completion_rate_pct': _rate(counts['completed'], scheduled),
        })
    return pd.DataFrame(rows, columns=WEEKDAY_COLUMNS)


def session_period_comparison(sessions, as_of, days=30):
    '''Compare the two consecutive, equally sized historical calendar windows.'''
    _validate_days(days)
    end = _as_of_date(as_of)
    first = end - pd.Timedelta(days=days - 1)
    previous_end = first - pd.Timedelta(days=1)
    previous_first = previous_end - pd.Timedelta(days=days - 1)
    data = _prepare_sessions(sessions, end)
    rows = []
    for period, start, finish in [
        ('previous', previous_first, previous_end),
        ('recent', first, end),
    ]:
        group = _in_window(data, start, finish)
        counts = _counts(group)
        scheduled = len(group)
        rows.append({
            'period': period,
            'period_start': start.date().isoformat(),
            'period_end': finish.date().isoformat(),
            'scheduled_sessions': scheduled,
            'completed_sessions': counts['completed'],
            'canceled_sessions': counts['canceled'],
            'no_shows': counts['no_show'],
            'completion_rate_pct': _rate(counts['completed'], scheduled),
            'no_show_rate_pct': _rate(counts['no_show'], scheduled),
            'completed_training_hours': _completed_hours(group),
        })
    return pd.DataFrame(rows, columns=PERIOD_COLUMNS)


def trainer_period_comparison(sessions, as_of, days=30):
    '''Compare observed workload per trainer across the two windows.

    The report shows session activity, NOT capacity utilization or trainer quality.
    Trainers with no sessions in the recent window remain visible.
    '''
    _validate_days(days)
    end = _as_of_date(as_of)
    current_start = end - pd.Timedelta(days=days - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=days - 1)
    data = _in_window(_prepare_sessions(sessions, end), previous_start, end)
    if data.empty:
        return pd.DataFrame(columns=TRAINER_COLUMNS)

    previous = _in_window(data, previous_start, previous_end)
    recent = _in_window(data, current_start, end)
    trainer_ids = sorted(set(data['trainer_id'].astype(str)))
    rows = []
    for trainer_id in trainer_ids:
        old = previous.loc[previous['trainer_id'].eq(trainer_id)]
        new = recent.loc[recent['trainer_id'].eq(trainer_id)]
        old_counts = _counts(old)
        new_counts = _counts(new)
        rows.append({
            'trainer_id': trainer_id,
            'previous_scheduled': len(old),
            'recent_scheduled': len(new),
            'previous_completed': old_counts['completed'],
            'recent_completed': new_counts['completed'],
            'completed_change': new_counts['completed'] - old_counts['completed'],
            'previous_no_shows': old_counts['no_show'],
            'recent_no_shows': new_counts['no_show'],
            'previous_unique_clients': old.loc[old['session_status'].eq('completed'), 'client_id'].nunique(),
            'recent_unique_clients': new.loc[new['session_status'].eq('completed'), 'client_id'].nunique(),
            'previous_training_hours': _completed_hours(old),
            'recent_training_hours': _completed_hours(new),
            'recent_completion_rate_pct': _rate(new_counts['completed'], len(new)),
        })
    return pd.DataFrame(rows, columns=TRAINER_COLUMNS)


def run_session_trainer_insights(sessions, as_of, output_dir=None):
    '''Generate Phase 2D reports; optionally export their CSV files.'''
    reports = {
        'weekly_session_trends': weekly_session_trends(sessions, as_of, days=90),
        'weekday_session_patterns': weekday_session_patterns(sessions, as_of, days=90),
        'session_period_comparison': session_period_comparison(sessions, as_of, days=30),
        'trainer_period_comparison': trainer_period_comparison(sessions, as_of, days=30),
    }
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(folder / f'{name}.csv', index=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Generate V2 session and trainer insights')
    parser.add_argument('--as-of', required=True, help='Report date (YYYY-MM-DD)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sessions = pd.read_csv(root / 'data' / 'processed' / 'sessions.csv')
    reports = run_session_trainer_insights(
        sessions, args.as_of, output_dir=root / 'outputs' / 'sessions' / 'v2'
    )
    for name, frame in reports.items():
        print(f'{name}: {len(frame)} row(s)')
    print('Reports saved to outputs/sessions/v2')


if __name__ == '__main__':
    main()
