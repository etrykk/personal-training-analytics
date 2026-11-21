#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Dashboard data presentation helpers (no Streamlit dependencies)

import pandas as pd


def filter_monthly_trends(trends, months=6, complete_only=False):
    '''Return chronologically ordered trends without modifying the source frame.

    Missing values stay missing. Incomplete months can be excluded from
    completed-month comparisons, but remain visible in regular trend charts.
    '''
    required = {'month', 'partial_month', 'net_cash_collected',
                'clients_with_completed_sessions', 'completed_sessions'}
    if not isinstance(trends, pd.DataFrame):
        raise TypeError('trends must be a pandas DataFrame')
    if required - set(trends.columns):
        raise ValueError(f'Missing trend columns: {sorted(required - set(trends.columns))}')
    if not isinstance(months, int) or isinstance(months, bool) or months < 1:
        raise ValueError('months must be a positive integer')
    view = trends.copy()
    view['month'] = pd.to_datetime(view['month'].astype(str), format='%Y-%m', errors='raise')
    # Both in-memory booleans and read_csv strings are handled consistently.
    raw_partial = view['partial_month'].astype('string').str.lower()
    if not raw_partial.isin(['true', 'false']).all():
        raise ValueError('partial_month must contain True or False')
    view['partial_month'] = raw_partial.eq('true').astype(bool)
    view = view.sort_values('month', kind='stable')
    if complete_only:
        view = view.loc[~view['partial_month']]
    return view.tail(months).reset_index(drop=True)


def overview_kpis(overview):
    '''Select labeled business KPIs using exactly the established V2 definitions.'''
    required = {
        'total_clients', 'clients_attending_30d', 'completed_sessions_30d',
        'session_completion_rate_30d_pct', 'active_membership_clients',
        'membership_retention_30d_pct', 'last_completed_month',
        'last_completed_month_net_cash', 'last_completed_month_growth_pct',
        'current_month_net_cash', 'current_month_partial',
        'clients_needing_followup',
    }
    if not isinstance(overview, pd.DataFrame) or len(overview) != 1:
        raise ValueError('overview must be a one-row pandas DataFrame')
    if required - set(overview.columns):
        raise ValueError(f'Missing overview columns: {sorted(required - set(overview.columns))}')
    row = overview.iloc[0]
    return {key: row[key] for key in sorted(required)}


def sorted_business_insights(insights):
    '''Keep actionable insights first and preserve original content.'''
    required = {'priority', 'category', 'headline', 'detail', 'observed_value'}
    if not isinstance(insights, pd.DataFrame):
        raise TypeError('insights must be a pandas DataFrame')
    if required - set(insights.columns):
        raise ValueError('insights missing required columns')
    view = insights.copy()
    priority_order = {'review': 0, 'monitor': 1, 'info': 2}
    view['_order'] = view['priority'].map(priority_order).fillna(3)
    return (view.sort_values('_order', kind='stable').drop(columns='_order')
            .reset_index(drop=True))


def search_workspace(reports, query, limit=20, trainer_labels=None):
    """Search existing dashboard tables without changing the source reports.

    The search matches literal text (not regular expressions) in client IDs,
    trainer IDs, and insight text. Empty queries return empty results.
    """
    if not isinstance(query, str):
        raise TypeError('query must be a string')
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('limit must be a positive integer')
    sources = {
        'clients': (reports['clients']['client_engagement_status'], ['client_id']),
        'trainers': (reports['sessions']['trainer_period_comparison'], ['trainer_id']),
        'insights': (reports['health']['business_health_insights'], ['headline', 'detail']),
    }
    result = {}
    term = query.strip()
    for name, (frame, columns) in sources.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f'{name} must be a DataFrame')
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'{name} is missing search columns: {sorted(missing)}')
        if not term:
            result[name] = frame.iloc[0:0].copy()
            continue
        matched = pd.Series(False, index=frame.index)
        for column in columns:
            matched = matched | frame[column].astype('string').str.contains(
                term, case=False, regex=False, na=False)
        if name == 'trainers' and trainer_labels:
            display = frame['trainer_id'].astype('string').map(
                lambda value: trainer_labels.get(str(value), str(value)))
            matched = matched | display.astype('string').str.contains(
                term, case=False, regex=False, na=False)
        result[name] = frame.loc[matched].head(limit).copy()
    return result


def readable_chart_label(value):
    """Return a readable label without changing chart source data or identifiers.

    Intended for axis categories, trace names and other display labels.
    Example: ``no_recent_sessions`` -> ``No recent sessions``.
    """
    if not isinstance(value, str):
        raise TypeError('chart label must be a string')
    label = ' '.join(value.replace('_', ' ').split())
    #Only capitalize generated field names; preserve user-entered trainer display names.
    return label[0].upper() + label[1:] if '_' in value and label else label
