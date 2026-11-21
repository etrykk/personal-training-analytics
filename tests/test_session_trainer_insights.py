#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Session and Trainer Insights Tests

import pandas as pd
import pytest

from src.analytics.session_trainer_insights import (
    weekly_session_trends,
    weekday_session_patterns,
    session_period_comparison,
    trainer_period_comparison,
    run_session_trainer_insights,
)


AS_OF = '2026-11-20'


def example_sessions():
    return pd.DataFrame([
        {'client_id': 'C1', 'trainer_id': 'T1', 'session_date': '2026-11-20', 'session_status': 'completed', 'duration_minutes': 60},
        {'client_id': 'C1', 'trainer_id': 'T1', 'session_date': '2026-11-19', 'session_status': 'canceled', 'duration_minutes': 60},
        {'client_id': 'C2', 'trainer_id': 'T2', 'session_date': '2026-11-18', 'session_status': 'no_show', 'duration_minutes': 30},
        {'client_id': 'C1', 'trainer_id': 'T1', 'session_date': '2026-10-01', 'session_status': 'completed', 'duration_minutes': 45},
        {'client_id': 'C2', 'trainer_id': 'T2', 'session_date': '2026-10-20', 'session_status': 'completed', 'duration_minutes': 30},
        {'client_id': 'C2', 'trainer_id': 'T2', 'session_date': '2026-11-21', 'session_status': 'completed', 'duration_minutes': 100},
    ])


def test_period_comparison_correct_windows():
    result = session_period_comparison(example_sessions(), AS_OF)
    old, current = result.iloc[0], result.iloc[1]
    assert old['period_start'] == '2026-09-22'
    assert old['period_end'] == '2026-10-21'
    assert current['period_start'] == '2026-10-22'
    assert current['period_end'] == '2026-11-20'
    assert old['completed_sessions'] == 2
    assert current['scheduled_sessions'] == 3
    assert current['completed_sessions'] == 1
    assert current['canceled_sessions'] == 1
    assert current['no_shows'] == 1
    assert current['completion_rate_pct'] == pytest.approx(100 / 3)
    assert current['completed_training_hours'] == pytest.approx(1)


def test_future_sessions_are_excluded():
    result = weekly_session_trends(example_sessions(), AS_OF)
    assert result['scheduled_sessions'].sum() == 5
    assert result['completed_sessions'].sum() == 3


def test_weekly_totals_and_rates():
    result = weekly_session_trends(example_sessions(), AS_OF)
    assert result['no_shows'].sum() == 1
    assert result['canceled_sessions'].sum() == 1
    assert result['completed_training_hours'].sum() == pytest.approx(2.25)
    assert result.iloc[-1]['partial_week'] == True


def test_weekly_includes_weeks_without_sessions():
    sessions = example_sessions().iloc[:1]
    result = weekly_session_trends(sessions, AS_OF, days=30)
    assert result['scheduled_sessions'].sum() == 1
    assert (result['scheduled_sessions'] == 0).any()
    assert result.loc[result['scheduled_sessions'].eq(0), 'completion_rate_pct'].isna().all()


def test_weekday_patterns_count_correctly():
    result = weekday_session_patterns(example_sessions(), AS_OF)
    assert len(result) == 7
    assert list(result['weekday']) == ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    friday = result.loc[result['weekday'].eq('Friday')].iloc[0]
    assert friday['completed_sessions'] == 1
    assert result['scheduled_sessions'].sum() == 5


def test_trainer_comparison_includes_inactive_trainers():
    result = trainer_period_comparison(example_sessions(), AS_OF)
    assert set(result['trainer_id']) == {'T1', 'T2'}
    t1 = result.set_index('trainer_id').loc['T1']
    assert t1['previous_completed'] == 1
    assert t1['recent_completed'] == 1
    assert t1['recent_unique_clients'] == 1
    assert t1['recent_completion_rate_pct'] == pytest.approx(50)
    t2 = result.set_index('trainer_id').loc['T2']
    assert t2['previous_completed'] == 1
    assert t2['recent_completed'] == 0
    assert t2['completed_change'] == -1
    assert t2['recent_training_hours'] == 0


def test_trainer_missing_id_uses_unassigned():
    data = example_sessions().iloc[:1].drop(columns='trainer_id')
    result = trainer_period_comparison(data, AS_OF)
    assert result['trainer_id'].tolist() == ['Unassigned']


def test_no_duration_reports_unavailable_hours():
    data = example_sessions().drop(columns='duration_minutes')
    result = session_period_comparison(data, AS_OF)
    assert pd.isna(result.loc[1, 'completed_training_hours'])
    assert result.loc[1, 'completed_sessions'] == 1


def test_empty_data_preserves_report_structure():
    data = example_sessions().iloc[0:0]
    assert len(weekly_session_trends(data, AS_OF)) > 0
    assert len(weekday_session_patterns(data, AS_OF)) == 7
    assert trainer_period_comparison(data, AS_OF).empty
    assert list(trainer_period_comparison(data, AS_OF)) == [
        'trainer_id', 'previous_scheduled', 'recent_scheduled',
        'previous_completed', 'recent_completed', 'completed_change',
        'previous_no_shows', 'recent_no_shows', 'previous_unique_clients',
        'recent_unique_clients', 'previous_training_hours',
        'recent_training_hours', 'recent_completion_rate_pct'
    ]


def test_invalid_reporting_window_raises():
    with pytest.raises(ValueError):
        weekly_session_trends(example_sessions(), AS_OF, days=0)
    with pytest.raises(ValueError):
        session_period_comparison(example_sessions(), AS_OF, days=True)


def test_unknown_historical_status_raises():
    data = example_sessions()
    data.loc[0, 'session_status'] = 'maybe'
    with pytest.raises(ValueError):
        weekday_session_patterns(data, AS_OF)


def test_future_unknown_status_does_not_change_historical_reports():
    data = example_sessions()
    data.loc[5, 'session_status'] = 'scheduled'
    result = session_period_comparison(data, AS_OF)
    assert result.loc[1, 'scheduled_sessions'] == 3


def test_nonpositive_completed_duration_raises():
    data = example_sessions()
    data.loc[0, 'duration_minutes'] = 0
    with pytest.raises(ValueError):
        trainer_period_comparison(data, AS_OF)


def test_input_frame_is_not_modified():
    data = example_sessions()
    saved = data.copy(deep=True)
    run_session_trainer_insights(data, AS_OF)
    pd.testing.assert_frame_equal(data, saved)


def test_reports_write_four_csv_files(tmp_path):
    result = run_session_trainer_insights(example_sessions(), AS_OF, tmp_path)
    assert len(result) == 4
    for key, frame in result.items():
        file = tmp_path / f'{key}.csv'
        assert file.exists()
        loaded = pd.read_csv(file)
        assert len(loaded) == len(frame)


def test_monday_window_boundary_is_included():
    data = pd.DataFrame([
        {'client_id': 'C1', 'trainer_id': 'T1', 'session_date': '2026-11-16', 'session_status': 'completed', 'duration_minutes': 60}
    ])
    result = weekly_session_trends(data, '2026-11-16', days=1)
    assert len(result) == 1
    assert result.iloc[0]['completed_sessions'] == 1
