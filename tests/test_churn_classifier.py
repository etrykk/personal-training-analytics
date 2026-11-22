#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4C: Experimental Churn Classification Tests

import pandas as pd
import pytest

from src.ml.churn_classifier import (
    _validate_snapshots,
    chronological_client_disjoint_split,
    evaluate_training_readiness,
    evaluate_experimental_classifiers,
    run_analysis,
)
from src.ml.historical_churn import MODEL_FEATURE_COLUMNS

CUTOFF = '2026-01-01'


def example_snapshots():
    """Synthetic examples with enough independent clients to exercise the gate.

    The fixture is intentionally a software test, not a business dataset.
    """
    rows = []
    #Earlier cohort: 60 cancellation outcomes, 80 negative outcomes.
    #Later, new cohort: 20 cancellation outcomes, 30 negative outcomes.
    for part, snapshot_date, horizon_end, positive, negative in [
        ('TR', '2025-01-01', '2025-01-31', 60, 80),
        ('TE', '2026-01-01', '2026-01-31', 20, 30),
    ]:
        for i in range(positive + negative):
            is_churn = i < positive
            row = {
                'membership_id': f'M_{part}_{i}',
                'client_id': f'C_{part}_{i}',
                'snapshot_date': snapshot_date,
                'horizon_end': horizon_end,
                'churn_within_horizon': int(is_churn),
                'label_status': 'positive' if is_churn else 'negative',
            }
            for col in MODEL_FEATURE_COLUMNS:
                row[col] = 0
            row.update({
                'membership_tenure_days': 40 + i % 12,
                'client_tenure_days': 90 + i % 40,
                'completed_sessions_30d': 0 if is_churn else 6,
                'completed_sessions_90d': 1 if is_churn else 15,
                'scheduled_sessions_30d': 1 if is_churn else 7,
                'no_shows_30d': 1 if is_churn else 0,
                'canceled_sessions_30d': 0,
                'days_since_last_completed_session': 72 if is_churn else 3,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_chronological_holdout_no_overlap():
    train, test, details = chronological_client_disjoint_split(example_snapshots(), CUTOFF)
    assert len(train) == 140
    assert len(test) == 50
    assert set(train.client_id).isdisjoint(set(test.client_id))
    assert (train.horizon_end < pd.Timestamp(CUTOFF)).all()
    assert (test.snapshot_date >= pd.Timestamp(CUTOFF)).all()
    assert details['train_client_overlap_purged'] == 0


def test_censored_rows_excluded():
    data = example_snapshots()
    extra = data.iloc[[0]].copy()
    extra['membership_id'] = 'MC'
    extra['client_id'] = 'CC'
    extra['label_status'] = 'censored'
    extra['churn_within_horizon'] = pd.NA
    train, _, details = chronological_client_disjoint_split(
        pd.concat([data, extra], ignore_index=True), CUTOFF)
    assert len(train) == 140
    assert details['censored_excluded'] == 1


def test_horizon_purge_and_client_overlap_purge():
    data = example_snapshots()
    #Earlier row is leaked into test if a single client appears on both sides.
    overlap = data.iloc[[0]].copy()
    overlap['client_id'] = 'C_TE_0'
    overlap['membership_id'] = 'M_overlap'
    data = pd.concat([data, overlap], ignore_index=True)
    #Changing one earlier cohort horizon to reach into holdout removes it.
    data.loc[1, 'horizon_end'] = '2026-01-01'
    train, _, details = chronological_client_disjoint_split(data, CUTOFF)
    assert details['train_client_overlap_purged'] == 1
    assert details['train_horizon_purged'] == 1
    assert not (train.client_id == 'C_TE_0').any()


def test_gate_approves_independent_synthetic_fixture():
    checks, decision, train, test = evaluate_training_readiness(example_snapshots(), CUTOFF)
    assert checks.passed.all()
    assert bool(decision.training_approved.iloc[0])
    assert len(train) == 140 and len(test) == 50


def test_small_fixture_never_fits_model():
    data = example_snapshots().iloc[:20]
    out = evaluate_experimental_classifiers(data, CUTOFF)
    assert not bool(out['decision'].training_approved.iloc[0])
    assert out['metrics'].empty


def test_models_evaluated_only_after_approval():
    out = evaluate_experimental_classifiers(example_snapshots(), CUTOFF)
    metrics = out['metrics'].set_index('model')
    assert set(metrics.index) == {'majority_class_baseline', 'logistic_regression'}
    assert metrics.loc['majority_class_baseline', 'recall'] == 0
    assert metrics.loc['logistic_regression', 'recall'] > metrics.loc['majority_class_baseline', 'recall']
    assert metrics.loc['logistic_regression', 'precision'] > 0
    assert (metrics[['accuracy', 'recall', 'f1']] <= 1).all().all()


def test_real_uploaded_report_stays_blocked():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'outputs' / 'predictive' / 'historical' / 'historical_churn_snapshots.csv'
    if not path.exists():
        pytest.skip('Historical report has not yet been generated')
    report = pd.read_csv(path)
    decision = evaluate_experimental_classifiers(report, '2026-09-01')['decision']
    assert not bool(decision.training_approved.iloc[0])


def test_missing_feature_raises():
    data = example_snapshots().drop(columns=['completed_sessions_30d'])
    with pytest.raises(ValueError, match='missing columns'):
        evaluate_experimental_classifiers(data, CUTOFF)


def test_inconsistent_censored_label_rejected():
    data = example_snapshots()
    data.loc[0, 'label_status'] = 'censored'
    with pytest.raises(ValueError, match='Inconsistent'):
        _validate_snapshots(data)


def test_duplicate_member_snapshot_rejected():
    data = example_snapshots()
    data.loc[1, 'membership_id'] = data.loc[0, 'membership_id']
    with pytest.raises(ValueError, match='Duplicate'):
        _validate_snapshots(data)


def test_negative_feature_rejected():
    data = example_snapshots()
    data.loc[1, 'membership_tenure_days'] = -2
    with pytest.raises(ValueError, match='negative'):
        _validate_snapshots(data)


def test_original_input_unmodified():
    data = example_snapshots()
    original = data.copy(deep=True)
    evaluate_experimental_classifiers(data, CUTOFF)
    pd.testing.assert_frame_equal(data, original)


def test_cli_report_exports_and_no_identifying_predictions(tmp_path):
    path = tmp_path / 'snapshots.csv'
    example_snapshots().to_csv(path, index=False)
    results = run_analysis(path, CUTOFF, tmp_path / 'report')
    names = {p.name for p in (tmp_path / 'report').glob('*.csv')}
    assert names == {'churn_classification_checks.csv',
                     'churn_classification_decision.csv',
                     'churn_classification_metrics.csv'}
    assert 'client_id' not in results['metrics'].columns
    assert len(results['metrics']) == 2


def test_blocked_report_outputs_no_fake_scores(tmp_path):
    path = tmp_path / 'snapshots.csv'
    example_snapshots().iloc[:15].to_csv(path, index=False)
    result = run_analysis(path, CUTOFF, tmp_path / 'report')
    assert result['metrics'].empty
    assert not bool(result['decision'].training_approved.iloc[0])
