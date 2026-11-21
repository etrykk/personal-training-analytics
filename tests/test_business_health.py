#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 3A - Business Health Tests

import pandas as pd
import pytest

from src.analytics.business_health import (
    run_business_health,
    business_health_insights,
    dashboard_monthly_trends,
)

AS_OF = '2026-11-20'


def sample_data():
    clients = pd.DataFrame([
        {'client_id': 'C1', 'join_date': '2026-09-01'},
        {'client_id': 'C2', 'join_date': '2026-09-02'},
        {'client_id': 'C3', 'join_date': '2026-11-01'},
        {'client_id': 'C4', 'join_date': '2027-01-01'},
    ])
    memberships = pd.DataFrame([
        {'membership_id': 'M1', 'client_id': 'C1', 'start_date': '2026-10-01',
         'end_date': '2026-11-30'},
        {'membership_id': 'M2', 'client_id': 'C2', 'start_date': '2026-09-01',
         'end_date': '2026-10-31'},
        {'membership_id': 'M3', 'client_id': 'C4', 'start_date': '2027-01-01',
         'end_date': '2027-04-01'},
    ])
    sessions = pd.DataFrame([
        {'client_id': 'C1', 'session_date': '2026-10-10', 'session_status': 'completed',
         'duration_minutes': 60, 'trainer_id': 'T1'},
        {'client_id': 'C2', 'session_date': '2026-09-30', 'session_status': 'completed',
         'duration_minutes': 60, 'trainer_id': 'T2'},
        {'client_id': 'C1', 'session_date': '2026-11-01', 'session_status': 'completed',
         'duration_minutes': 60, 'trainer_id': 'T1'},
        {'client_id': 'C2', 'session_date': '2026-11-10', 'session_status': 'no_show',
         'duration_minutes': 60, 'trainer_id': 'T2'},
        {'client_id': 'C3', 'session_date': '2026-11-15', 'session_status': 'completed',
         'duration_minutes': 60, 'trainer_id': 'T2'},
        {'client_id': 'C1', 'session_date': '2026-12-05', 'session_status': 'completed',
         'duration_minutes': 60, 'trainer_id': 'T1'},
    ])
    payments = pd.DataFrame([
        {'payment_date': '2026-09-15', 'amount': 90.0, 'payment_status': 'completed',
         'transaction_type': 'payment'},
        {'payment_date': '2026-10-15', 'amount': 100.0, 'payment_status': 'completed',
         'transaction_type': 'payment'},
        {'payment_date': '2026-11-01', 'amount': 180.0, 'payment_status': 'completed',
         'transaction_type': 'payment'},
        {'payment_date': '2026-11-02', 'amount': 20.0, 'payment_status': 'completed',
         'transaction_type': 'refund'},
        {'payment_date': '2026-11-03', 'amount': 10.0, 'payment_status': 'failed',
         'transaction_type': 'payment'},
        {'payment_date': '2026-11-04', 'amount': 25.0, 'payment_status': 'pending',
         'transaction_type': 'payment'},
        {'payment_date': '2026-12-01', 'amount': 999.0, 'payment_status': 'completed',
         'transaction_type': 'payment'},
    ])
    return clients, memberships, sessions, payments


def generate(as_of=AS_OF, output_dir=None):
    return run_business_health(*sample_data(), as_of=as_of, output_dir=output_dir)


def test_overview_contains_expected_fields():
    reports = generate()
    overview = reports['business_overview']
    assert len(overview) == 1
    assert 'activity_continuation_pct' in overview.columns
    assert 'last_completed_month_net_cash' in overview.columns
    assert overview.iloc[0]['as_of'] == AS_OF


def test_recent_sessions_counts_and_rates():
    row = generate()['business_overview'].iloc[0]
    assert row['scheduled_sessions_30d'] == 3
    assert row['completed_sessions_30d'] == 2
    assert row['session_completion_rate_30d_pct'] == pytest.approx(200 / 3)
    assert row['no_show_rate_30d_pct'] == pytest.approx(100 / 3)
    assert row['completed_training_hours_30d'] == 2


def test_future_clients_and_sessions_excluded():
    row = generate()['business_overview'].iloc[0]
    assert row['total_clients'] == 3
    assert row['clients_attending_30d'] == 2
    assert row['trainers_with_completed_sessions_30d'] == 2


def test_client_activity_is_not_membership_retention():
    row = generate()['business_overview'].iloc[0]
    assert row['activity_continuation_pct'] == pytest.approx(50.0)
    assert row['membership_retention_30d_pct'] == pytest.approx(50.0)
    assert row['clients_needing_followup'] == 1
    assert row['clients_without_session_history'] == 0


def test_membership_coverage_and_expiry():
    row = generate()['business_overview'].iloc[0]
    assert row['active_membership_clients'] == 1
    assert row['memberships_expiring_30d'] == 1


def test_cash_separates_partial_and_complete_months():
    row = generate()['business_overview'].iloc[0]
    assert row['current_month_net_cash'] == pytest.approx(160)
    assert bool(row['current_month_partial'])
    assert row['last_completed_month'] == '2026-10'
    assert row['last_completed_month_net_cash'] == pytest.approx(100)
    assert row['last_completed_month_growth_pct'] == pytest.approx(100 / 9)


def test_future_payments_and_noncompleted_not_cash():
    row = generate()['business_overview'].iloc[0]
    assert row['current_month_net_cash'] == pytest.approx(160)
    assert row['failed_payment_attempts_90d'] == 1
    assert row['pending_payment_attempts_90d'] == 1


def test_month_end_is_not_partial():
    row = generate('2026-10-31')['business_overview'].iloc[0]
    assert not bool(row['current_month_partial'])
    assert row['last_completed_month'] == '2026-10'
    assert row['current_month_net_cash'] == pytest.approx(100)


def test_dashboard_trends_mark_partial_month():
    trends = generate()['dashboard_monthly_trends']
    assert trends.iloc[-1]['month'] == '2026-11'
    assert bool(trends.iloc[-1]['partial_month'])
    assert trends.iloc[-1]['net_cash_collected'] == pytest.approx(160)
    assert list(trends['month']) == sorted(trends['month'])


def test_missing_monthly_cash_stays_unavailable():
    activity = pd.DataFrame({
        'month': ['2026-10'], 'eligible_clients': [2],
        'clients_with_completed_sessions': [1], 'completed_sessions': [2],
        'monthly_activity_rate_pct': [50.0]
    })
    cash = pd.DataFrame(columns=['payment_month', 'gross_cash_collected',
                                 'refunded_cash', 'net_cash_collected', 'partial_month'])
    output = dashboard_monthly_trends(activity, cash)
    assert pd.isna(output.iloc[0]['net_cash_collected'])


def test_business_insights_are_rule_based():
    overview = generate()['business_overview']
    insights = business_health_insights(overview, no_show_threshold_pct=10.0)
    assert {'engagement', 'membership', 'payments'} <= set(insights['category'])
    assert 'sessions' not in set(insights['category'])  # only 3 sessions < 10
    assert 'churn_prediction' not in set(insights['category'])


def test_no_show_threshold_and_minimum_denominator():
    overview = generate()['business_overview'].copy()
    overview.loc[0, 'scheduled_sessions_30d'] = 12
    output = business_health_insights(overview, no_show_threshold_pct=30.0)
    assert 'sessions' in set(output['category'])
    output = business_health_insights(overview, no_show_threshold_pct=40.0)
    assert 'sessions' not in set(output['category'])


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        business_health_insights(generate()['business_overview'], no_show_threshold_pct=101)


def test_reports_export_and_originals_unchanged(tmp_path):
    sources = sample_data()
    snapshots = [frame.copy(deep=True) for frame in sources]
    result = run_business_health(*sources, as_of=AS_OF, output_dir=tmp_path)
    assert set(result) == {'business_overview', 'business_health_insights',
                           'dashboard_monthly_trends'}
    assert {f.name for f in tmp_path.glob('*.csv')} == {
        'business_overview.csv', 'business_health_insights.csv',
        'dashboard_monthly_trends.csv',
    }
    for original, snapshot in zip(sources, snapshots):
        pd.testing.assert_frame_equal(original, snapshot)


def test_invalid_reporting_date_rejected():
    with pytest.raises((ValueError, TypeError)):
        generate(as_of='not-a-date')
