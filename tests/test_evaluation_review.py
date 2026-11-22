#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4D: Experimental Model Evaluation Review Tests

import pandas as pd
import pytest

from src.ml.evaluation_review import (
    wilson_interval, review_model_evaluation, run_analysis, REVIEW_COLUMNS,
)


def screening(approved=False):
    conditions = [
        ('training cancellation memberships', 40 if approved else 6, 30),
        ('testing cancellation memberships', 20 if approved else 1, 10),
        ('training non-cancellation memberships', 60 if approved else 11, 30),
        ('testing non-cancellation memberships', 30 if approved else 24, 10),
    ]
    checks = pd.DataFrame([{
        'check': name, 'observed': value,
        'minimum_screening_requirement': required,
        'passed': value >= required,
    } for name, value, required in conditions])
    decision = pd.DataFrame([{
        'cutoff_date': '2026-01-01', 'train_rows': 100 if approved else 34,
        'test_rows': 50 if approved else 43,
        'training_approved': approved,
        'status': ('experimental_evaluation_permitted' if approved
                   else 'blocked_insufficient_independent_events'),
    }])
    return checks, decision


def sample_scores(recall=.75):
    if recall == .75:
        #Two independently labeled groups, 50 test examples total.
        tp, fp, tn, fn = 15, 9, 21, 5
    else:
        #Below the initial 60% recall goal.
        tp, fp, tn, fn = 10, 10, 20, 10
    def row(name, a, b, c, d):
        precision = a / (a + b) if a + b else 0.0
        recall_ = a / (a + d)
        f1 = 2*a / (2*a+b+d) if 2*a+b+d else 0
        return {
            'model': name,
            'accuracy': (a + c) / 50,
            'balanced_accuracy': (recall_ + c/(c+b))/2,
            'precision': precision,
            'recall': recall_,
            'f1': f1,
            'roc_auc': .50 if name == 'majority_class_baseline' else .80,
            'average_precision': .40 if name == 'majority_class_baseline' else .70,
            'true_positives': a,
            'false_positives': b,
            'true_negatives': c,
            'false_negatives': d,
        }
    return pd.DataFrame([
        row('majority_class_baseline', 0, 0, 30, 20),
        row('logistic_regression', tp, fp, tn, fn),
    ])


def blocked_scores():
    return pd.DataFrame(columns=sample_scores().columns)


def test_wilson_interval_zero_denominator():
    lo, hi = wilson_interval(0, 0)
    assert pd.isna(lo) and pd.isna(hi)


def test_wilson_interval_basic():
    lo, hi = wilson_interval(6, 10)
    assert 0 < lo < .6 < hi < 1


def test_wilson_interval_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_blocked_training_produces_no_scores():
    checks, decision = screening()
    r = review_model_evaluation(checks, decision, blocked_scores())
    assert r['comparison'].empty
    assert r['comparison'].columns.tolist() == REVIEW_COLUMNS
    assert r['decision'].iloc[0]['status'] == 'blocked_insufficient_independent_events'
    assert not bool(r['decision'].iloc[0]['business_predictions_enabled'])
    assert not r['checks']['evaluated'].any()


def test_boolean_strings_are_read_safely():
    checks, decision = screening()
    checks['passed'] = checks['passed'].map(str)
    decision['training_approved'] = 'False'
    r = review_model_evaluation(checks, decision, blocked_scores())
    assert not bool(r['decision'].iloc[0]['classifier_screening_approved'])


def test_approved_synthetic_example_comparison():
    checks, decision = screening(True)
    r = review_model_evaluation(checks, decision, sample_scores())
    assert bool(r['decision'].iloc[0]['exploratory_targets_met'])
    assert r['decision'].iloc[0]['status'] == 'experimental_metric_targets_met'
    assert not bool(r['decision'].iloc[0]['business_predictions_enabled'])
    comparison = r['comparison'].set_index('model')
    assert comparison.loc['logistic_regression', 'recall_change_vs_baseline'] == .75
    assert comparison.loc['logistic_regression', 'recall_ci_lower_95'] < .75
    assert comparison.loc['logistic_regression', 'recall_ci_upper_95'] > .75
    assert r['checks']['passed'].all()


def test_low_recall_does_not_meet_target():
    checks, decision = screening(True)
    r = review_model_evaluation(checks, decision, sample_scores(.50))
    assert not bool(r['decision'].iloc[0]['exploratory_targets_met'])
    assert r['decision'].iloc[0]['status'] == 'experimental_metric_targets_not_met'


def test_does_not_modify_inputs():
    checks, decision = screening(True)
    metrics = sample_scores()
    originals = [frame.copy(deep=True) for frame in (checks, decision, metrics)]
    review_model_evaluation(checks, decision, metrics)
    for actual, expected in zip((checks, decision, metrics), originals):
        pd.testing.assert_frame_equal(actual, expected)


def test_blocked_model_with_scores_is_rejected():
    checks, decision = screening()
    with pytest.raises(ValueError, match='Blocked training'):
        review_model_evaluation(checks, decision, sample_scores())


def test_inconsistent_upstream_approval_rejected():
    checks, decision = screening()
    decision['training_approved'] = True
    with pytest.raises(ValueError, match='disagrees'):
        review_model_evaluation(checks, decision, blocked_scores())


def test_missing_baseline_rejected():
    checks, decision = screening(True)
    with pytest.raises(ValueError, match='one baseline'):
        review_model_evaluation(checks, decision, sample_scores().iloc[[1]])


def test_inconsistent_confusion_matrix_rejected():
    checks, decision = screening(True)
    metrics = sample_scores()
    metrics.loc[1, 'true_positives'] = 16
    with pytest.raises(ValueError, match='totals'):
        review_model_evaluation(checks, decision, metrics)


def test_inconsistent_metric_rejected():
    checks, decision = screening(True)
    metrics = sample_scores()
    metrics.loc[1, 'recall'] = .90
    with pytest.raises(ValueError, match='Inconsistent recall'):
        review_model_evaluation(checks, decision, metrics)


def test_bad_threshold_rejected():
    checks, decision = screening(True)
    with pytest.raises(ValueError, match='min_recall'):
        review_model_evaluation(checks, decision, sample_scores(), min_recall=1.5)


def test_writes_three_aggregate_reports(tmp_path):
    checks, decision = screening()
    path = tmp_path / 'classification'
    path.mkdir()
    checks.to_csv(path/'churn_classification_checks.csv', index=False)
    decision.to_csv(path/'churn_classification_decision.csv', index=False)
    blocked_scores().to_csv(path/'churn_classification_metrics.csv', index=False)
    r = run_analysis(path, tmp_path/'evaluation')
    files = {p.name for p in (tmp_path/'evaluation').glob('*.csv')}
    assert files == {'churn_evaluation_comparison.csv', 'churn_evaluation_checks.csv',
                     'churn_evaluation_decision.csv'}
    assert r['comparison'].empty
    for table in r.values():
        assert 'client_id' not in table.columns
        assert 'membership_id' not in table.columns


def test_approved_but_missing_scores_not_successful():
    checks, decision = screening(True)
    r = review_model_evaluation(checks, decision, blocked_scores())
    assert r['decision'].iloc[0]['status'] == 'approved_but_metrics_missing'
    assert not bool(r['decision'].iloc[0]['business_predictions_enabled'])
