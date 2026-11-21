#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Client Engagement Tests

import pandas as pd
import pytest

from src.analytics.client_engagement import (
    client_activity_summary,
    attendance_consistency,
    client_engagement_status,
)


AS_OF = '2026-11-20'


def make_clients(*ids):
    #Create a simple client dataset for testing
    return pd.DataFrame({'client_id': list(ids)})


def make_sessions(records):
    #Create session data using the columns required by the functions
    return pd.DataFrame(records, columns=['client_id', 'session_date', 'session_status'])


def client_row(report, client_id):
    #Retrieve an individual client from an analytics report
    return report.set_index('client_id').loc[client_id]


def test_activity_counts_30_60_90_day_boundaries():
    #Window start dates are inclusive; sessions before a boundary are excluded
    dates = ['2026-11-20', '2026-10-22', '2026-10-21',
             '2026-09-22', '2026-09-21', '2026-08-23',
             '2026-08-22']
    sessions = make_sessions([('C001', date, 'completed') for date in dates])

    result = client_row(client_activity_summary(make_clients('C001'), sessions, AS_OF), 'C001')

    assert result['sessions_30d'] == 2
    assert result['sessions_60d'] == 4
    assert result['sessions_90d'] == 6
    assert result['days_since_last_session'] == 0


def test_activity_ignores_canceled_no_show_future_and_invalid_dates():
    sessions = make_sessions([
        ('C001', '2026-11-19', 'completed'),
        ('C001', '2026-11-20', 'canceled'),
        ('C001', '2026-11-20', 'no_show'),
        ('C001', '2026-11-21', 'completed'),
        ('C001', 'not_a_date', 'completed'),
    ])
    result = client_row(client_activity_summary(make_clients('C001'), sessions, AS_OF), 'C001')

    assert result['sessions_30d'] == 1
    assert result['days_since_last_session'] == 1


def test_activity_includes_clients_with_no_history():
    sessions = make_sessions([('C001', '2026-09-26', 'completed')])
    report = client_activity_summary(make_clients('C001', 'C002'), sessions, AS_OF)

    assert len(report) == 2
    assert client_row(report, 'C001')['days_since_last_session'] == 55
    empty_client = client_row(report, 'C002')
    assert empty_client['sessions_30d'] == 0
    assert empty_client['sessions_60d'] == 0
    assert empty_client['sessions_90d'] == 0
    assert pd.isna(empty_client['days_since_last_session'])


def test_activity_rejects_missing_columns():
    with pytest.raises(ValueError):
        client_activity_summary(make_clients('C001'), pd.DataFrame({'client_id': ['C001']}), AS_OF)


def test_attendance_uses_nonoverlapping_periods():
    #Recent dates: Oct 22-Nov 20. Previous dates: Sep 22-Oct 21.
    sessions = make_sessions([
        ('C001', '2026-09-22', 'completed'),
        ('C001', '2026-10-21', 'completed'),
        ('C001', '2026-10-22', 'completed'),
        ('C001', '2026-11-20', 'completed'),
        ('C001', '2026-09-21', 'completed'),
        ('C001', '2026-11-21', 'completed'),
        ('C001', '2026-11-19', 'no_show'),
    ])
    result = client_row(attendance_consistency(make_clients('C001'), sessions, AS_OF), 'C001')

    assert result['previous_30d'] == 2
    assert result['recent_30d'] == 2
    assert result['session_change'] == 0
    assert result['attendance_trend'] == 'steady'


def test_attendance_trend_categories():
    sessions = make_sessions([
        ('C001', '2026-10-01', 'completed'),
        ('C001', '2026-10-07', 'completed'),
        ('C001', '2026-10-14', 'completed'),
        ('C001', '2026-11-01', 'completed'),
        ('C002', '2026-11-02', 'completed'),
        ('C002', '2026-11-05', 'completed'),
        ('C003', '2026-10-05', 'completed'),
        ('C003', '2026-11-10', 'completed'),
        ('C005', '2026-10-05', 'completed'),
        ('C005', '2026-11-05', 'completed'),
        ('C005', '2026-11-10', 'completed'),
    ])
    report = attendance_consistency(make_clients('C001', 'C002', 'C003', 'C004', 'C005'), sessions, AS_OF)
    assert client_row(report, 'C001')['attendance_trend'] == 'decreased'
    assert client_row(report, 'C002')['attendance_trend'] == 'new_activity'
    assert client_row(report, 'C003')['attendance_trend'] == 'steady'
    assert client_row(report, 'C004')['attendance_trend'] == 'no_activity'
    assert client_row(report, 'C005')['attendance_trend'] == 'increased'


def test_engagement_status_categories():
    sessions = make_sessions([
        ('C001', '2026-10-01', 'completed'),
        ('C001', '2026-10-07', 'completed'),
        ('C001', '2026-10-14', 'completed'),
        ('C001', '2026-11-18', 'completed'),
        ('C002', '2026-11-02', 'completed'),
        ('C002', '2026-11-05', 'completed'),
        ('C003', '2026-10-11', 'completed'),
        ('C004', '2026-08-22', 'completed'),
    ])
    clients = make_clients('C001', 'C002', 'C003', 'C004', 'C005')
    activity = client_activity_summary(clients, sessions, AS_OF)
    attendance = attendance_consistency(clients, sessions, AS_OF)
    report = client_engagement_status(activity, attendance)

    assert client_row(report, 'C001')['engagement_status'] == 'declining_activity'
    assert client_row(report, 'C002')['engagement_status'] == 'recently_active'
    assert client_row(report, 'C003')['engagement_status'] == 'no_recent_sessions'
    assert client_row(report, 'C004')['engagement_status'] == 'inactive'
    assert client_row(report, 'C005')['engagement_status'] == 'no_session_history'


def test_engagement_status_threshold_is_configurable():
    sessions = make_sessions([('C001', '2026-09-26', 'completed')])
    clients = make_clients('C001')
    activity = client_activity_summary(clients, sessions, AS_OF)
    attendance = attendance_consistency(clients, sessions, AS_OF)
    report = client_engagement_status(activity, attendance, inactivity_days=45)

    assert client_row(report, 'C001')['engagement_status'] == 'inactive'


def test_engagement_rejects_unmatched_client_summaries():
    clients = make_clients('C001', 'C002')
    sessions = make_sessions([('C001', '2026-11-19', 'completed')])
    activity = client_activity_summary(clients, sessions, AS_OF)
    attendance = attendance_consistency(make_clients('C001'), sessions, AS_OF)

    with pytest.raises(ValueError, match='every activity client'):
        client_engagement_status(activity, attendance)


def test_engagement_rejects_inconsistent_reporting_dates():
    clients = make_clients('C001')
    sessions = make_sessions([('C001', '2026-11-20', 'completed')])
    activity = client_activity_summary(clients, sessions, AS_OF)
    attendance = attendance_consistency(clients, sessions, '2026-11-19')

    with pytest.raises(ValueError, match='same reporting date'):
        client_engagement_status(activity, attendance)
