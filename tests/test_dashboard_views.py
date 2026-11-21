#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Dashboard view helper tests

import pandas as pd
import pytest

from src.dashboard_views import filter_monthly_trends, overview_kpis, sorted_business_insights, search_workspace, readable_chart_label


def example_trends():
    return pd.DataFrame({
        'month': ['2026-11', '2026-09', '2026-10'],
        'partial_month': [True, False, False],
        'net_cash_collected': [500.0, 400.0, 600.0],
        'clients_with_completed_sessions': [3, 2, 4],
        'completed_sessions': [8, 4, 10],
    })


def test_trends_order_monthly():
    data = filter_monthly_trends(example_trends())
    assert list(data['month'].dt.strftime('%Y-%m')) == ['2026-09', '2026-10', '2026-11']


def test_trends_filter_completed_months():
    data = filter_monthly_trends(example_trends(), complete_only=True)
    assert list(data['net_cash_collected']) == [400.0, 600.0]
    assert not data['partial_month'].any()


def test_trends_respects_count():
    data = filter_monthly_trends(example_trends(), months=2)
    assert len(data) == 2
    assert data.iloc[-1]['net_cash_collected'] == 500.0


def test_trends_boolean_strings_after_csv_read():
    data = example_trends()
    data['partial_month'] = ['True', 'False', 'False']
    out = filter_monthly_trends(data, complete_only=True)
    assert len(out) == 2


def test_trends_does_not_change_input():
    data = example_trends()
    unchanged = data.copy(deep=True)
    filter_monthly_trends(data)
    pd.testing.assert_frame_equal(data, unchanged)


def test_trends_preserves_missing_cash():
    data = example_trends()
    data.loc[1, 'net_cash_collected'] = float('nan')
    out = filter_monthly_trends(data)
    assert pd.isna(out.iloc[0]['net_cash_collected'])


def test_trends_rejects_invalid_number_of_months():
    with pytest.raises(ValueError):
        filter_monthly_trends(example_trends(), months=0)


def test_trends_rejects_bad_partial_month_flag():
    data = example_trends()
    data['partial_month'] = data['partial_month'].astype(object)
    data.loc[0, 'partial_month'] = 'maybe'
    with pytest.raises(ValueError):
        filter_monthly_trends(data)


def test_trends_rejects_missing_columns():
    with pytest.raises(ValueError):
        filter_monthly_trends(example_trends().drop(columns='month'))


def example_overview():
    return pd.DataFrame([{
        'total_clients': 100,
        'clients_attending_30d': 62,
        'completed_sessions_30d': 206,
        'session_completion_rate_30d_pct': 81.4,
        'active_membership_clients': 27,
        'membership_retention_30d_pct': 81.5,
        'last_completed_month': '2026-10',
        'last_completed_month_net_cash': 13250.0,
        'last_completed_month_growth_pct': 42.4,
        'current_month_net_cash': 8975.0,
        'current_month_partial': True,
        'clients_needing_followup': 55,
    }])


def test_overview_kpis_preserves_values():
    out = overview_kpis(example_overview())
    assert out['completed_sessions_30d'] == 206
    assert out['last_completed_month_net_cash'] == 13250.0
    assert out['current_month_partial'] == True


def test_overview_rejects_multiple_rows():
    with pytest.raises(ValueError):
        overview_kpis(pd.concat([example_overview(), example_overview()]))


def test_overview_rejects_missing_columns():
    with pytest.raises(ValueError):
        overview_kpis(example_overview().drop(columns='total_clients'))


def test_insights_sorted_by_priority():
    insights = pd.DataFrame([
        {'priority': 'info', 'category': 'other', 'headline': 'Info', 'detail': 'd', 'observed_value': 0},
        {'priority': 'review', 'category': 'engagement', 'headline': 'Review', 'detail': 'd', 'observed_value': 5},
        {'priority': 'monitor', 'category': 'revenue', 'headline': 'Monitor', 'detail': 'd', 'observed_value': -2},
    ])
    result = sorted_business_insights(insights)
    assert list(result['priority']) == ['review', 'monitor', 'info']


def test_insights_does_not_modify_source():
    original = pd.DataFrame([{'priority': 'review', 'category': 'payments',
                              'headline': 'Test', 'detail': 'Some detail', 'observed_value': 5}])
    untouched = original.copy(deep=True)
    sorted_business_insights(original)
    pd.testing.assert_frame_equal(original, untouched)


def example_search_reports():
    return {
        'clients': {'client_engagement_status': pd.DataFrame({
            'client_id': ['C001', 'C002', 'X.3'],
            'engagement_status': ['active', 'inactive', 'active']} )},
        'sessions': {'trainer_period_comparison': pd.DataFrame({
            'trainer_id': ['T101', 'T202'], 'recent_completed': [8, 9]})},
        'health': {'business_health_insights': pd.DataFrame({
            'headline': ['Check cancellations', 'Review payments'],
            'detail': ['Two clients', 'Card issues'],
            'priority': ['review', 'monitor']})},
    }

def test_workspace_search_finds_clients_and_insights():
    result = search_workspace(example_search_reports(), 'c001')
    assert result['clients']['client_id'].tolist() == ['C001']
    assert result['trainers'].empty
    assert result['insights'].empty
    assert search_workspace(example_search_reports(), 'payments')['insights']['headline'].tolist() == ['Review payments']

def test_workspace_search_matches_literal_text_not_regex():
    result = search_workspace(example_search_reports(), 'X.3')
    assert result['clients']['client_id'].tolist() == ['X.3']
    result = search_workspace(example_search_reports(), 'X*3')
    assert result['clients'].empty

def test_workspace_search_empty_query():
    result = search_workspace(example_search_reports(), '  ')
    assert all(frame.empty for frame in result.values())

def test_workspace_search_does_not_mutate_and_checks_limits():
    source = example_search_reports()
    original = source['clients']['client_engagement_status'].copy(deep=True)
    search_workspace(source, 'C', limit=1)
    pd.testing.assert_frame_equal(source['clients']['client_engagement_status'], original)
    with pytest.raises(ValueError):
        search_workspace(source, 'C', limit=0)
    with pytest.raises(TypeError):
        search_workspace(source, 123)


def test_workspace_search_can_match_edited_trainer_names():
    result = search_workspace(example_search_reports(), 'Jordan',
                              trainer_labels={'T101': 'Jordan', 'T202': 'Alex'})
    assert result['trainers']['trainer_id'].tolist() == ['T101']
    assert result['clients'].empty


def test_readable_chart_label_replaces_underscores_for_display():
    assert readable_chart_label('completed_sessions') == 'Completed sessions'
    assert readable_chart_label('no_recent_sessions') == 'No recent sessions'
    assert readable_chart_label(' session_completion_rate_pct ') == 'Session completion rate pct'
    assert readable_chart_label('Already readable') == 'Already readable'


def test_readable_chart_label_rejects_non_string():
    with pytest.raises(TypeError):
        readable_chart_label(None)
