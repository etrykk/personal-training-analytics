#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Client Retention and Engagement Analytics

import pandas as pd


def client_activity_summary(clients, sessions, as_of):
    '''
    Calculate completed training sessions over the last 30, 60, and 90 days
    for every client, including clients without any completed sessions.

    Each window includes the reporting date (as_of). For example, the
    30-day window includes as_of and the 29 preceding calendar dates.

    A missing days_since_last_session means that the client has never
    completed a session in the supplied history as of the reporting date.
    This is activity analysis, not a prediction of client churn.
    '''
    #Verify that the data contains the columns needed for the analysis
    required_clients = {'client_id'}
    required_sessions = {'client_id', 'session_date', 'session_status'}
    if not required_clients.issubset(clients.columns):
        raise ValueError('clients must contain client_id')
    if not required_sessions.issubset(sessions.columns):
        raise ValueError('sessions must contain client_id, session_date, and session_status')

    #Use a reporting date instead of a hard-coded date
    as_of = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of):
        raise ValueError('as_of must be a valid reporting date')

    #Keep all clients, even if they have never attended a training session
    results = clients[['client_id']].copy()
    data = sessions[['client_id', 'session_date', 'session_status']].copy()

    #Convert session dates and filter to completed sessions on or before as_of
    data['session_date'] = pd.to_datetime(data['session_date'], errors='coerce').dt.normalize()
    completed = data.loc[
        data['session_status'].astype('string').str.strip().str.lower().eq('completed')
        & data['session_date'].notna()
        & (data['session_date'] <= as_of)
    ].copy()

    #Count completed sessions in each reporting window
    for days in (30, 60, 90):
        start_date = as_of - pd.Timedelta(days=days - 1)
        counts = (completed.loc[completed['session_date'] >= start_date]
                  .groupby('client_id').size())
        results[f'sessions_{days}d'] = (
            results['client_id'].map(counts).fillna(0).astype(int)
        )

    #Find each client's most recent completed training session
    last_session = completed.groupby('client_id')['session_date'].max()
    results['last_completed_session'] = results['client_id'].map(last_session)
    results['days_since_last_session'] = (
        (as_of - results['last_completed_session']).dt.days.astype('Int64')
    )

    #Keep only the columns needed for the engagement report
    return results[[
        'client_id', 'sessions_30d', 'sessions_60d',
        'sessions_90d', 'days_since_last_session'
    ]]


def attendance_consistency(clients, sessions, as_of):
    '''
    Compare completed sessions in the most recent 30 calendar days with
    the 30 calendar days immediately before that period.

    The reporting date is included in the recent period. For example,
    when as_of is 2026-11-20, recent is Oct 22-Nov 20 inclusive, and
    previous is Sep 22-Oct 21 inclusive.

    This compares activity, not attendance at scheduled appointments.
    It cannot distinguish cancellations by a trainer from missed sessions.
    '''
    #Make sure both datasets contain the necessary information
    if 'client_id' not in clients.columns:
        raise ValueError('clients must contain client_id')
    required_sessions = {'client_id', 'session_date', 'session_status'}
    if not required_sessions.issubset(sessions.columns):
        raise ValueError('sessions must contain client_id, session_date, and session_status')

    as_of = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of):
        raise ValueError('as_of must be a valid reporting date')

    #Use two equal-length, nonoverlapping periods
    recent_start = as_of - pd.Timedelta(days=29)
    previous_start = recent_start - pd.Timedelta(days=30)

    data = sessions[['client_id', 'session_date', 'session_status']].copy()
    data['session_date'] = pd.to_datetime(data['session_date'], errors='coerce').dt.normalize()
    completed = data.loc[
        data['session_status'].astype('string').str.strip().str.lower().eq('completed')
        & data['session_date'].notna()
        & data['session_date'].between(previous_start, as_of)
    ].copy()

    previous_counts = (completed.loc[completed['session_date'] < recent_start]
                       .groupby('client_id').size())
    recent_counts = (completed.loc[completed['session_date'] >= recent_start]
                     .groupby('client_id').size())

    #Keep clients who have no recorded training sessions
    results = clients[['client_id']].copy()
    results['previous_30d'] = results['client_id'].map(previous_counts).fillna(0).astype(int)
    results['recent_30d'] = results['client_id'].map(recent_counts).fillna(0).astype(int)
    results['session_change'] = results['recent_30d'] - results['previous_30d']

    #These labels describe session counts, not client churn or intent
    results['attendance_trend'] = 'steady'
    no_activity = (results['previous_30d'] == 0) & (results['recent_30d'] == 0)
    new_activity = (results['previous_30d'] == 0) & (results['recent_30d'] > 0)
    results.loc[results['session_change'] < 0, 'attendance_trend'] = 'decreased'
    results.loc[results['session_change'] > 0, 'attendance_trend'] = 'increased'
    results.loc[new_activity, 'attendance_trend'] = 'new_activity'
    results.loc[no_activity, 'attendance_trend'] = 'no_activity'

    return results[[
        'client_id', 'previous_30d', 'recent_30d',
        'session_change', 'attendance_trend'
    ]]


def client_engagement_status(activity_summary, attendance_summary,
                             inactivity_days=60, minimum_previous_sessions=2):
    '''
    Combine activity and attendance summaries into descriptive client
    engagement categories. These are rule-based labels, NOT predictions.

    Status priority:
      no_session_history : no completed sessions recorded by the as_of date
      inactive           : last completed session > inactivity_days ago
      no_recent_sessions : no completed sessions in the most recent 30 days
      declining_activity : recent sessions < previous, with a meaningful
                           previous baseline (minimum_previous_sessions)
      recently_active    : all remaining clients with recent activity

    The input summaries should be created with the same reporting date.
    '''
    if not isinstance(inactivity_days, int) or isinstance(inactivity_days, bool) or inactivity_days < 1:
        raise ValueError('inactivity_days must be a positive integer')
    if (not isinstance(minimum_previous_sessions, int)
            or isinstance(minimum_previous_sessions, bool)
            or minimum_previous_sessions < 1):
        raise ValueError('minimum_previous_sessions must be a positive integer')

    required_activity = {
        'client_id', 'sessions_30d', 'sessions_60d', 'sessions_90d',
        'days_since_last_session'
    }
    required_attendance = {
        'client_id', 'previous_30d', 'recent_30d',
        'session_change', 'attendance_trend'
    }
    if not required_activity.issubset(activity_summary.columns):
        raise ValueError('activity_summary is missing required columns')
    if not required_attendance.issubset(attendance_summary.columns):
        raise ValueError('attendance_summary is missing required columns')

    #Join one row per client and do not silently lose unmatched clients
    results = activity_summary.merge(
        attendance_summary[[
            'client_id', 'previous_30d', 'recent_30d',
            'session_change', 'attendance_trend'
        ]], on='client_id', how='left', validate='one_to_one'
    )
    if results['recent_30d'].isna().any():
        raise ValueError('attendance_summary must include every activity client')
    if (results['sessions_30d'] != results['recent_30d']).any():
        raise ValueError('activity and attendance summaries must use the same reporting date')

    #Assign more specific statuses after the default active category
    results['engagement_status'] = 'recently_active'
    declining = ((results['previous_30d'] >= minimum_previous_sessions)
                 & (results['recent_30d'] < results['previous_30d']))
    results.loc[declining, 'engagement_status'] = 'declining_activity'
    results.loc[results['recent_30d'] == 0, 'engagement_status'] = 'no_recent_sessions'
    results.loc[
        results['days_since_last_session'].gt(inactivity_days).fillna(False),
        'engagement_status'
    ] = 'inactive'
    results.loc[results['days_since_last_session'].isna(), 'engagement_status'] = 'no_session_history'

    return results[[
        'client_id', 'sessions_30d', 'sessions_60d', 'sessions_90d',
        'days_since_last_session', 'previous_30d', 'session_change',
        'attendance_trend', 'engagement_status'
    ]]


def attendance_consistency(clients, sessions, as_of):
    '''
    Compare completed sessions in the most recent 30 calendar days with
    the 30 calendar days immediately before that period.

    The reporting date is included in the recent period. For example,
    when as_of is 2026-11-20, recent is Oct 22-Nov 20 inclusive, and
    previous is Sep 22-Oct 21 inclusive.

    This compares activity, not attendance at scheduled appointments.
    It cannot distinguish cancellations by a trainer from missed sessions.
    '''
    #Make sure both datasets contain the necessary information
    if 'client_id' not in clients.columns:
        raise ValueError('clients must contain client_id')
    required_sessions = {'client_id', 'session_date', 'session_status'}
    if not required_sessions.issubset(sessions.columns):
        raise ValueError('sessions must contain client_id, session_date, and session_status')

    as_of = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of):
        raise ValueError('as_of must be a valid reporting date')

    #Use two equal-length, nonoverlapping periods
    recent_start = as_of - pd.Timedelta(days=29)
    previous_start = recent_start - pd.Timedelta(days=30)

    data = sessions[['client_id', 'session_date', 'session_status']].copy()
    data['session_date'] = pd.to_datetime(data['session_date'], errors='coerce').dt.normalize()
    completed = data.loc[
        data['session_status'].astype('string').str.strip().str.lower().eq('completed')
        & data['session_date'].notna()
        & data['session_date'].between(previous_start, as_of)
    ].copy()

    previous_counts = (completed.loc[completed['session_date'] < recent_start]
                       .groupby('client_id').size())
    recent_counts = (completed.loc[completed['session_date'] >= recent_start]
                     .groupby('client_id').size())

    #Keep clients who have no recorded training sessions
    results = clients[['client_id']].copy()
    results['previous_30d'] = results['client_id'].map(previous_counts).fillna(0).astype(int)
    results['recent_30d'] = results['client_id'].map(recent_counts).fillna(0).astype(int)
    results['session_change'] = results['recent_30d'] - results['previous_30d']

    #These labels describe session counts, not client churn or intent
    results['attendance_trend'] = 'steady'
    no_activity = (results['previous_30d'] == 0) & (results['recent_30d'] == 0)
    new_activity = (results['previous_30d'] == 0) & (results['recent_30d'] > 0)
    results.loc[results['session_change'] < 0, 'attendance_trend'] = 'decreased'
    results.loc[results['session_change'] > 0, 'attendance_trend'] = 'increased'
    results.loc[new_activity, 'attendance_trend'] = 'new_activity'
    results.loc[no_activity, 'attendance_trend'] = 'no_activity'

    return results[[
        'client_id', 'previous_30d', 'recent_30d',
        'session_change', 'attendance_trend'
    ]]


def client_engagement_status(activity_summary, attendance_summary,
                             inactivity_days=60, minimum_previous_sessions=2):
    '''
    Combine activity and attendance summaries into descriptive client
    engagement categories. These are rule-based labels, NOT predictions.

    Status priority:
      no_session_history : no completed sessions recorded by the as_of date
      inactive           : last completed session > inactivity_days ago
      no_recent_sessions : no completed sessions in the most recent 30 days
      declining_activity : recent sessions < previous, with a meaningful
                           previous baseline (minimum_previous_sessions)
      recently_active    : all remaining clients with recent activity

    The input summaries should be created with the same reporting date.
    '''
    if not isinstance(inactivity_days, int) or isinstance(inactivity_days, bool) or inactivity_days < 1:
        raise ValueError('inactivity_days must be a positive integer')
    if (not isinstance(minimum_previous_sessions, int)
            or isinstance(minimum_previous_sessions, bool)
            or minimum_previous_sessions < 1):
        raise ValueError('minimum_previous_sessions must be a positive integer')

    required_activity = {
        'client_id', 'sessions_30d', 'sessions_60d', 'sessions_90d',
        'days_since_last_session'
    }
    required_attendance = {
        'client_id', 'previous_30d', 'recent_30d',
        'session_change', 'attendance_trend'
    }
    if not required_activity.issubset(activity_summary.columns):
        raise ValueError('activity_summary is missing required columns')
    if not required_attendance.issubset(attendance_summary.columns):
        raise ValueError('attendance_summary is missing required columns')

    #Join one row per client and do not silently lose unmatched clients
    results = activity_summary.merge(
        attendance_summary[[
            'client_id', 'previous_30d', 'recent_30d',
            'session_change', 'attendance_trend'
        ]], on='client_id', how='left', validate='one_to_one'
    )
    if results['recent_30d'].isna().any():
        raise ValueError('attendance_summary must include every activity client')
    if (results['sessions_30d'] != results['recent_30d']).any():
        raise ValueError('activity and attendance summaries must use the same reporting date')

    #Assign more specific statuses after the default active category
    results['engagement_status'] = 'recently_active'
    declining = ((results['previous_30d'] >= minimum_previous_sessions)
                 & (results['recent_30d'] < results['previous_30d']))
    results.loc[declining, 'engagement_status'] = 'declining_activity'
    results.loc[results['recent_30d'] == 0, 'engagement_status'] = 'no_recent_sessions'
    results.loc[
        results['days_since_last_session'].gt(inactivity_days).fillna(False),
        'engagement_status'
    ] = 'inactive'
    results.loc[results['days_since_last_session'].isna(), 'engagement_status'] = 'no_session_history'

    return results[[
        'client_id', 'sessions_30d', 'sessions_60d', 'sessions_90d',
        'days_since_last_session', 'previous_30d', 'session_change',
        'attendance_trend', 'engagement_status'
    ]]
