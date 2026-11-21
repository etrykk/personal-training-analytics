#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 2B Client Activity and Business Insights Tests

import pandas as pd
import pytest

from src.analytics.client_insights import (
    monthly_client_activity,
    activity_continuation_summary,
    generate_engagement_insights,
    run_client_insights,
)


def clients(ids, dates):
    return pd.DataFrame({'client_id': ids, 'join_date': dates})


def sessions(records):
    return pd.DataFrame(records, columns=['client_id', 'session_date', 'session_status'])


def test_monthly_activity_counts_clients_and_completed_sessions():
    roster = clients(['C1', 'C2', 'C3'], ['2026-08-01', '2026-10-10', '2026-12-01'])
    history = sessions([
        ('C1', '2026-10-01', 'completed'),
        ('C1', '2026-10-31', 'completed'),
        ('C1', '2026-11-20', 'completed'),
        ('C2', '2026-11-10', 'completed'),
        ('C2', '2026-11-11', 'no_show'),
        ('C2', '2026-11-21', 'completed'),  # future excluded
        ('C3', '2026-11-10', 'completed'),  # not joined yet
    ])
    result = monthly_client_activity(roster, history, '2026-11-20', months=2)
    assert result['month'].tolist() == ['2026-10', '2026-11']
    assert result['eligible_clients'].tolist() == [2, 2]
    assert result['clients_with_completed_sessions'].tolist() == [1, 2]
    assert result['completed_sessions'].tolist() == [2, 2]
    assert result['monthly_activity_rate_pct'].tolist() == pytest.approx([50., 100.])


def test_monthly_activity_includes_zero_activity_month():
    roster = clients(['C1'], ['2026-01-01'])
    result = monthly_client_activity(roster, sessions([]), '2026-03-15', months=3)
    assert result['month'].tolist() == ['2026-01', '2026-02', '2026-03']
    assert result['clients_with_completed_sessions'].tolist() == [0, 0, 0]
    assert result['monthly_activity_rate_pct'].tolist() == pytest.approx([0., 0., 0.])


def test_monthly_activity_undefined_without_eligible_clients():
    roster = clients(['C1'], ['2026-11-10'])
    report = monthly_client_activity(roster, sessions([]), '2026-11-20', months=2)
    assert report.iloc[0]['eligible_clients'] == 0
    assert pd.isna(report.iloc[0]['monthly_activity_rate_pct'])
    assert report.iloc[1]['eligible_clients'] == 1
    assert report.iloc[1]['monthly_activity_rate_pct'] == 0


def test_monthly_activity_rejects_invalid_inputs():
    roster = clients(['C1'], ['2026-01-01'])
    with pytest.raises(ValueError, match='positive integer'):
        monthly_client_activity(roster, sessions([]), '2026-11-20', months=0)
    with pytest.raises(ValueError, match='join_date'):
        monthly_client_activity(clients(['C1'], ['bad date']), sessions([]), '2026-11-20')
    with pytest.raises(ValueError, match='unique'):
        monthly_client_activity(clients(['C1', 'C1'], ['2026-01-01']*2), sessions([]), '2026-11-20')


def test_activity_continuation_covers_all_four_groups():
    attendance = pd.DataFrame({
        'client_id': ['C1', 'C2', 'C3', 'C4'],
        'previous_30d': [2, 1, 0, 0],
        'recent_30d': [3, 0, 1, 0],
    })
    row = activity_continuation_summary(attendance).iloc[0]
    assert row['total_clients'] == 4
    assert row['previously_active_clients'] == 2
    assert row['recently_active_clients'] == 2
    assert row['active_in_both_periods'] == 1
    assert row['stopped_attending'] == 1
    assert row['newly_attending'] == 1
    assert row['inactive_in_both_periods'] == 1
    assert row['activity_continuation_pct'] == pytest.approx(50.)


def test_activity_continuation_undefined_without_previous_activity():
    attendance = pd.DataFrame({'client_id': ['C1'], 'previous_30d': [0], 'recent_30d': [1]})
    row = activity_continuation_summary(attendance).iloc[0]
    assert pd.isna(row['activity_continuation_pct'])
    assert row['newly_attending'] == 1


def test_activity_continuation_empty_dataset():
    attendance = pd.DataFrame(columns=['client_id', 'previous_30d', 'recent_30d'])
    row = activity_continuation_summary(attendance).iloc[0]
    assert row['total_clients'] == 0
    assert pd.isna(row['activity_continuation_pct'])


def test_activity_continuation_rejects_invalid_counts():
    attendance = pd.DataFrame({'client_id': ['C1'], 'previous_30d': [-1], 'recent_30d': [1]})
    with pytest.raises(ValueError, match='nonnegative'):
        activity_continuation_summary(attendance)
    attendance['previous_30d'] = 1.5
    with pytest.raises(ValueError, match='whole numbers'):
        activity_continuation_summary(attendance)


def test_engagement_insights_are_descriptive_and_not_churn_predictions():
    engagement = pd.DataFrame({
        'client_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'engagement_status': ['inactive', 'declining_activity',
                              'no_recent_sessions', 'no_session_history', 'recently_active'],
    })
    summary = pd.DataFrame({
        'previously_active_clients': [2],
        'active_in_both_periods': [1],
        'activity_continuation_pct': [50.],
    })
    report = generate_engagement_insights(engagement, summary).set_index('category')
    assert report.loc['inactive', 'client_count'] == 1
    assert report.loc['declining_activity', 'client_count'] == 1
    assert report.loc['no_recent_sessions', 'client_count'] == 1
    assert report.loc['no_session_history', 'client_count'] == 1
    assert report.loc['activity_continuation', 'client_count'] == 1
    assert 'not membership retention' in report.loc['activity_continuation', 'description'].lower()


def test_engagement_insights_no_signals():
    engagement = pd.DataFrame({'client_id': ['C1'], 'engagement_status': ['recently_active']})
    summary = pd.DataFrame({'previously_active_clients': [0],
                            'active_in_both_periods': [0],
                            'activity_continuation_pct': [pd.NA]})
    report = generate_engagement_insights(engagement, summary)
    assert report['category'].tolist() == ['no_signals']


def test_run_client_insights_creates_four_reports_and_saves_csvs(tmp_path):
    roster = clients(['C1', 'C2'], ['2026-01-01', '2026-01-01'])
    history = sessions([
        ('C1', '2026-10-01', 'completed'),
        ('C1', '2026-11-19', 'completed'),
    ])
    report = run_client_insights(roster, history, '2026-11-20', months=2, output_dir=tmp_path)
    assert set(report) == {
        'monthly_client_activity', 'activity_continuation',
        'client_engagement_status', 'engagement_insights'
    }
    assert len(report['client_engagement_status']) == 2
    assert len(report['monthly_client_activity']) == 2
    for name in report:
        assert (tmp_path / f'{name}.csv').is_file()
