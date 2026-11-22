#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4A: Experimental ML Data Readiness Tests

import pandas as pd
import pytest

from src.ml.churn_readiness import assess_churn_readiness, run_readiness_report

AS_OF = '2026-11-20'


def make_datasets():
    clients = pd.DataFrame({
        'client_id': ['C1', 'C2', 'C3', 'C4'],
        'join_date': ['2026-01-01'] * 4,
        'status': ['active', 'inactive', 'active', 'active'],
    })
    memberships = pd.DataFrame({
        'membership_id': ['M1', 'M2', 'M3', 'M4'],
        'client_id': ['C1', 'C2', 'C3', 'C4'],
        'start_date': ['2026-01-01', '2026-11-15', '2026-01-01', '2026-05-01'],
        'status': ['canceled', 'canceled', 'expired', 'active'],
        'cancellation_date': ['2026-11-10', '2026-11-18', None, None],
    })
    sessions = pd.DataFrame({
        'client_id': ['C1', 'C2', 'C2', 'C3'],
        'session_date': ['2026-01-10', '2026-11-10', '2026-12-01', '2026-11-15'],
        'session_status': ['completed', 'completed', 'completed', 'no_show'],
    })
    return clients, memberships, sessions


def audit(*, horizon_days=30, as_of=AS_OF, min_events=30, **overrides):
    c, m, s = make_datasets()
    for name, replacement in overrides.items():
        if name == 'clients': c = replacement
        if name == 'memberships': m = replacement
        if name == 'sessions': s = replacement
    return assess_churn_readiness(c, m, s, as_of=as_of,
                                  horizon_days=horizon_days, min_events=min_events)


def met(results, key):
    metrics = results[2].set_index('metric')['value']
    return metrics.loc[key]


def check(results, key):
    return results[1].set_index('check').loc[key, 'passed']


def test_cancellations_distinct_from_expiration_and_inactivity():
    result = audit()
    assert met(result, 'recorded_cancellations_by_as_of') == 2
    assert met(result, 'expired_memberships_by_as_of_not_churn_labels') == 1
    assert met(result, 'noncanceled_membership_records_not_negative_labels') == 2
    assert result[0].iloc[0]['training_approved'] == False


def test_full_horizon_history_requires_30_day_membership_tenure():
    result = audit()
    assert met(result, 'cancellations_with_full_horizon_membership_history') == 1
    assert check(result, 'full_horizon_cancellation_examples') == False


def test_different_horizon_recalculates_full_history_count():
    result = audit(horizon_days=2)
    assert met(result, 'cancellations_with_full_horizon_membership_history') == 2


def test_future_completed_sessions_excluded():
    result = audit()
    assert met(result, 'completed_sessions_through_as_of') == 2
    assert met(result, 'clients_with_completed_sessions_through_as_of') == 2


def test_future_cancellation_excluded_from_outcomes():
    c, m, s = make_datasets()
    m.loc[0, 'cancellation_date'] = '2026-12-01'
    result = audit(memberships=m)
    assert met(result, 'recorded_cancellations_by_as_of') == 1
    assert met(result, 'cancellations_after_as_of') == 1


def test_membership_starting_after_report_date_excluded():
    c, m, s = make_datasets()
    m.loc[0, 'start_date'] = '2026-12-01'
    result = audit(memberships=m)
    assert met(result, 'membership_records_started_by_as_of') == 3
    assert met(result, 'recorded_cancellations_by_as_of') == 1


def test_missing_cancellation_date_causes_failed_check():
    c, m, s = make_datasets()
    m.loc[0, 'cancellation_date'] = None
    result = audit(memberships=m)
    assert met(result, 'canceled_memberships_missing_date') == 1
    assert check(result, 'cancellation_date_completeness') == False


def test_cancel_before_start_is_not_counted_as_valid_event():
    c, m, s = make_datasets()
    m.loc[0, 'cancellation_date'] = '2025-12-01'
    result = audit(memberships=m)
    assert met(result, 'cancellations_before_membership_start') == 1
    assert met(result, 'recorded_cancellations_by_as_of') == 1
    assert check(result, 'cancellation_date_order') == False


def test_duplicate_membership_ids_are_flagged():
    c, m, s = make_datasets()
    m.loc[1, 'membership_id'] = 'M1'
    result = audit(memberships=m)
    assert met(result, 'duplicate_membership_id_rows') == 2
    assert check(result, 'unique_membership_ids') == False


def test_unknown_client_id_is_flagged():
    c, m, s = make_datasets()
    m.loc[0, 'client_id'] = 'UNKNOWN'
    result = audit(memberships=m)
    assert met(result, 'unknown_membership_client_rows') == 1
    assert check(result, 'member_client_relationships') == False


def test_invalid_required_date_is_flagged():
    c, m, s = make_datasets()
    s.loc[0, 'session_date'] = 'not-a-date'
    result = audit(sessions=s)
    assert met(result, 'invalid_session_date_rows') == 1
    assert check(result, 'required_event_dates_parse') == False


def test_inputs_are_not_mutated():
    c, m, s = make_datasets()
    before = (c.copy(deep=True), m.copy(deep=True), s.copy(deep=True))
    assess_churn_readiness(c, m, s, AS_OF)
    for actual, expected in zip((c, m, s), before):
        pd.testing.assert_frame_equal(actual, expected)


def test_missing_required_schema_is_informative():
    c, m, s = make_datasets()
    with pytest.raises(ValueError, match='cancellation_date'):
        assess_churn_readiness(c, m.drop(columns=['cancellation_date']), s, AS_OF)


@pytest.mark.parametrize('bad_horizon', [0, -1, 1.5, '30'])
def test_invalid_horizon_rejected(bad_horizon):
    with pytest.raises(ValueError, match='horizon_days'):
        audit(horizon_days=bad_horizon)


def test_empty_frames_are_audited_without_crashing():
    c, m, s = make_datasets()
    result = audit(clients=c.iloc[0:0], memberships=m.iloc[0:0], sessions=s.iloc[0:0])
    assert met(result, 'recorded_cancellations_by_as_of') == 0
    assert result[0].iloc[0]['basic_readiness_screen'] == 'insufficient_data_or_quality'


def test_report_writes_three_csv_files(tmp_path):
    c, m, s = make_datasets()
    for name, frame in [('clients', c), ('memberships', m), ('sessions', s)]:
        frame.to_csv(tmp_path / f'{name}.csv', index=False)
    output = tmp_path / 'generated'
    result = run_readiness_report(as_of=AS_OF, processed_dir=tmp_path, output_dir=output)
    assert set(result) == {'summary', 'checks', 'metrics'}
    assert sorted(path.name for path in output.glob('*.csv')) == [
        'churn_readiness_checks.csv',
        'churn_readiness_decision.csv',
        'churn_readiness_metrics.csv',
    ]


def test_small_sample_not_training_approved_even_with_screen_parameters_lowered():
    result = audit(min_events=1)
    assert result[0].iloc[0]['training_approved'] == False
