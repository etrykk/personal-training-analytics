#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Tests for Phase 4B historical membership cancellation examples

import pandas as pd
import pytest

from src.ml.historical_churn import (
    MODEL_FEATURE_COLUMNS,
    build_historical_labels,
    build_historical_features,
    build_historical_churn_dataset,
    summarize_historical_labels,
    run_analysis,
)

AS_OF = '2026-04-01'


def examples():
    clients = pd.DataFrame({
        'client_id': ['C1', 'C2', 'C3', 'C4'],
        'join_date': ['2025-01-01'] * 4,
        'status': ['inactive', 'active', 'active', 'active'],
    })
    memberships = pd.DataFrame({
        'membership_id': ['M1', 'M2', 'M3', 'M4'],
        'client_id': ['C1', 'C2', 'C3', 'C4'],
        'start_date': ['2026-01-01'] * 4,
        'end_date': ['2026-02-15', None, '2026-02-10', None],
        'status': ['canceled', 'active', 'expired', 'active'],
        'cancellation_date': ['2026-02-15', None, None, None],
        'cancellation_reason': ['price', None, None, None],
    })
    sessions = pd.DataFrame({
        'client_id': ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2'],
        'session_date': ['2026-01-10', '2026-01-30', '2026-02-01',
                         '2026-02-02', '2026-03-05', '2026-03-15', '2026-01-05'],
        'session_status': ['completed', 'no_show', 'completed', 'completed',
                           'completed', 'canceled', 'completed'],
    })
    return clients, memberships, sessions


def test_positive_label_in_30_day_horizon():
    _, members, _ = examples()
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M1') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert row.churn_within_horizon == 1
    assert row.label_status == 'positive'


def test_canceled_membership_not_active_after_event():
    _, members, _ = examples()
    result = build_historical_labels(members, AS_OF)
    assert not ((result.membership_id == 'M1') &
                (result.snapshot_date == pd.Timestamp('2026-03-01'))).any()


def test_expiry_in_horizon_is_censored_not_negative():
    _, members, _ = examples()
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M3') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert pd.isna(row.churn_within_horizon)
    assert row.label_reason == 'membership_ended_before_horizon'


def test_active_member_can_supply_negative_label():
    _, members, _ = examples()
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M2') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert row.churn_within_horizon == 0


def test_no_snapshots_without_complete_followup():
    _, members, _ = examples()
    result = build_historical_labels(members, '2026-03-15', horizon_days=30)
    assert result.horizon_end.max() <= pd.Timestamp('2026-03-15')


def test_event_exactly_at_horizon_end_is_positive():
    _, members, _ = examples()
    members.loc[0, ['end_date', 'cancellation_date']] = '2026-03-03'
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M1') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert row.churn_within_horizon == 1  # Feb 1 + 30 days = Mar 3


def test_event_after_horizon_is_negative_before_event():
    _, members, _ = examples()
    members.loc[0, ['end_date', 'cancellation_date']] = '2026-03-20'
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M1') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert row.churn_within_horizon == 0


def test_missing_cancellation_date_is_censored():
    _, members, _ = examples()
    members.loc[0, 'cancellation_date'] = None
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M1') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert pd.isna(row.churn_within_horizon)


def test_sessions_after_snapshot_and_on_snapshot_excluded():
    clients, members, sessions = examples()
    result = build_historical_churn_dataset(clients, members, sessions, AS_OF)
    # On Feb 1: only Jan 10 completed and Jan 30 no-show known.
    row = result.loc[(result.membership_id == 'M1') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert row.completed_sessions_30d == 1
    assert row.scheduled_sessions_30d == 2
    assert row.no_shows_30d == 1
    assert row.days_since_last_completed_session == 22


def test_client_status_and_future_cancellation_fields_not_used_as_features():
    assert not set(['status', 'cancellation_date', 'cancellation_reason',
                    'end_date', 'plan_name', 'monthly_fee']).intersection(MODEL_FEATURE_COLUMNS)


def test_no_completed_session_has_missing_recency():
    clients, members, sessions = examples()
    result = build_historical_churn_dataset(clients, members, sessions, AS_OF)
    row = result.loc[(result.membership_id == 'M3') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert pd.isna(row.days_since_last_completed_session)
    assert row.completed_sessions_30d == 0


def test_input_frames_unchanged():
    clients, members, sessions = examples()
    before = [frame.copy(deep=True) for frame in (clients, members, sessions)]
    build_historical_churn_dataset(clients, members, sessions, AS_OF)
    for actual, expected in zip((clients, members, sessions), before):
        pd.testing.assert_frame_equal(actual, expected)


def test_duplicate_memberships_rejected():
    _, members, _ = examples()
    members.loc[1, 'membership_id'] = 'M1'
    with pytest.raises(ValueError, match='unique'):
        build_historical_labels(members, AS_OF)


def test_missing_required_columns_rejected():
    _, members, _ = examples()
    with pytest.raises(ValueError, match='missing columns'):
        build_historical_labels(members.drop(columns=['cancellation_date']), AS_OF)


def test_invalid_horizon_rejected():
    _, members, _ = examples()
    with pytest.raises(ValueError, match='positive integer'):
        build_historical_labels(members, AS_OF, horizon_days=0)


def test_summary_counts_and_not_training_approved():
    clients, members, sessions = examples()
    result = build_historical_churn_dataset(clients, members, sessions, AS_OF)
    summary = summarize_historical_labels(result).iloc[0]
    assert summary.total_snapshots == len(result)
    assert summary.positive_unique_memberships == 1
    assert bool(summary.training_approved) is False


def test_run_analysis_writes_two_reports(tmp_path):
    clients, members, sessions = examples()
    input_dir = tmp_path / 'data'
    output_dir = tmp_path / 'reports'
    input_dir.mkdir()
    for name, frame in [('clients', clients), ('memberships', members), ('sessions', sessions)]:
        frame.to_csv(input_dir / f'{name}.csv', index=False)
    reports = run_analysis(AS_OF, data_dir=input_dir, output_dir=output_dir)
    assert len(reports['snapshots']) > 0
    assert (output_dir / 'historical_churn_snapshots.csv').exists()
    assert (output_dir / 'historical_churn_summary.csv').exists()


def test_empty_datasets_produce_empty_snapshot_table():
    clients, members, sessions = examples()
    result = build_historical_churn_dataset(clients, members.iloc[0:0], sessions, AS_OF)
    assert result.empty
    assert result.columns.tolist() == ['membership_id', 'client_id', 'snapshot_date',
                                       'horizon_end', 'churn_within_horizon',
                                       'label_status', 'label_reason'] + MODEL_FEATURE_COLUMNS


def test_missing_status_is_censored_instead_of_crashing():
    _, members, _ = examples()
    members.loc[1, 'status'] = None
    result = build_historical_labels(members, AS_OF)
    row = result.loc[(result.membership_id == 'M2') &
                     (result.snapshot_date == pd.Timestamp('2026-02-01'))].iloc[0]
    assert pd.isna(row.churn_within_horizon)
    assert row.label_reason == 'unknown_membership_status'


def test_future_sessions_do_not_change_earlier_features():
    clients, members, sessions = examples()
    before = build_historical_churn_dataset(clients, members, sessions, AS_OF)
    extra = pd.DataFrame({'client_id': ['C1'],
                          'session_date': ['2026-03-31'],
                          'session_status': ['completed']})
    after = build_historical_churn_dataset(
        clients, members, pd.concat([sessions, extra], ignore_index=True), AS_OF)
    selected = before.snapshot_date == pd.Timestamp('2026-02-01')
    pd.testing.assert_frame_equal(before.loc[selected].reset_index(drop=True),
                                  after.loc[selected].reset_index(drop=True))
