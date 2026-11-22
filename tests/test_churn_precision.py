#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4G: Research-Only Precision Optimization Tests

import numpy as np
import pandas as pd
import pytest

from src.ml.churn_precision import (
    CHECK_COLUMNS, DECISION_COLUMNS, METRIC_COLUMNS, THRESHOLD_COLUMNS,
    EXPERIMENT_FEATURES, TARGET, _confusion, _metric_row,
    evaluate_precision_optimization, run_analysis,
)


def sample_external(n=450, seed=46):
    """Synthetic TEST FIXTURE. These values are not claimed as real metrics."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        'Near_Location': rng.integers(0, 2, n),
        'Partner': rng.integers(0, 2, n),
        'Promo_friends': rng.integers(0, 2, n),
        'Phone': rng.integers(0, 2, n),
        'Contract_period': rng.choice([1, 6, 12], n),
        'Group_visits': rng.integers(0, 2, n),
    })
    p = np.where(x['Contract_period'] == 1, 0.66, 0.24)
    p = np.clip(p + 0.09 * (1 - x['Near_Location']) - 0.09 * x['Partner'], 0.03, 0.94)
    x['Churn'] = (rng.random(n) < p).astype(int)
    return x


def test_reports_have_expected_schemas():
    reports = evaluate_precision_optimization(sample_external())
    assert list(reports['checks']) == CHECK_COLUMNS
    assert list(reports['metrics']) == METRIC_COLUMNS
    assert list(reports['thresholds']) == THRESHOLD_COLUMNS
    assert list(reports['decision']) == DECISION_COLUMNS


def test_three_models_compared_on_same_test_set():
    reports = evaluate_precision_optimization(sample_external())
    assert set(reports['metrics']['model']) == {
        'majority_class_baseline', 'default_logistic_regression',
        'tuned_logistic_regression'}
    n_test = int(reports['decision'].loc[0, 'test_rows'])
    for _, row in reports['metrics'].iterrows():
        assert sum(int(row[name]) for name in (
            'true_positives', 'false_positives', 'true_negatives',
            'false_negatives')) == n_test


def test_holdout_splits_are_disjoint_in_size():
    d = evaluate_precision_optimization(sample_external())['decision'].iloc[0]
    assert int(d['train_rows'] + d['validation_rows'] + d['test_rows']) == int(d['input_rows'])
    assert (d['train_rows'], d['validation_rows'], d['test_rows']) == (270, 90, 90)


def test_optimization_is_reproducible():
    x = sample_external()
    a = evaluate_precision_optimization(x)
    b = evaluate_precision_optimization(x)
    for key in ('metrics', 'thresholds', 'decision'):
        pd.testing.assert_frame_equal(a[key], b[key])


def test_chosen_threshold_is_on_validation_curve():
    reports = evaluate_precision_optimization(sample_external())
    d = reports['decision'].iloc[0]
    threshold = d['best_threshold']
    row = reports['thresholds'].loc[reports['thresholds']['threshold'] == threshold].iloc[0]
    assert bool(row['meets_recall_goal'])
    assert 0 <= row['precision'] <= 1
    assert d['selected_on'] == 'validation_only'


def test_test_data_not_used_for_model_selection(monkeypatch):
    """Changes to untouched test outcomes cannot affect chosen settings.

    This fixture intercepts the random splitter so the final partition stays
    fixed while we flip ONLY test labels. It explicitly tests split isolation.
    """
    import sklearn.model_selection as ms
    orig = ms.train_test_split
    rng = np.random.default_rng(4)
    x = sample_external(450)
    #Inject a fixed 90-row test partition while retaining stratification
    test_indices = np.arange(90)
    train_val_indices = np.arange(90, 450)
    original_test_labels = x.loc[test_indices, 'Churn'].copy()
    calls = [0]

    def split_override(indices, *args, **kwargs):
        calls[0] += 1
        if calls[0] % 2 == 1:
            return train_val_indices.copy(), test_indices.copy()
        return orig(indices, *args, **kwargs)

    monkeypatch.setattr(ms, 'train_test_split', split_override)
    a = evaluate_precision_optimization(x)
    #Flip some final test labels while keeping the original class mix and
    #the exact training/validation split unaffected.
    switched = x.copy()
    perm = rng.permutation(len(test_indices))
    switched.loc[test_indices, 'Churn'] = original_test_labels.iloc[perm].to_numpy()
    b = evaluate_precision_optimization(switched)
    da, db = a['decision'].iloc[0], b['decision'].iloc[0]
    assert da['best_threshold'] == db['best_threshold']
    assert da['best_class_weight'] == db['best_class_weight']
    assert da['best_regularization_c'] == db['best_regularization_c']
    pd.testing.assert_frame_equal(a['thresholds'], b['thresholds'])


def test_missing_data_blocks(tmp_path):
    reports = run_analysis(tmp_path / 'missing.csv', tmp_path / 'out')
    assert reports['metrics'].empty
    assert reports['thresholds'].empty
    assert reports['decision'].loc[0, 'status'] == 'external_dataset_missing'
    assert not bool(reports['decision'].loc[0, 'business_predictions_enabled'])
    assert len(list((tmp_path / 'out').glob('churn_precision_*.csv'))) == 4


def test_insufficient_data_blocks():
    reports = evaluate_precision_optimization(sample_external(40))
    assert reports['metrics'].empty
    assert reports['decision'].loc[0, 'status'] == 'blocked_insufficient_external_examples'


def test_one_class_blocks():
    x = sample_external()
    x[TARGET] = 0
    reports = evaluate_precision_optimization(x)
    assert reports['metrics'].empty
    assert not bool(reports['decision'].loc[0, 'business_predictions_enabled'])


def test_writes_only_aggregate_reports(tmp_path):
    csv = tmp_path / 'gym_churn_us.csv'
    sample_external().to_csv(csv, index=False)
    reports = run_analysis(csv, tmp_path / 'reports')
    assert bool(reports['decision'].loc[0, 'test_evaluated'])
    assert {p.name for p in (tmp_path / 'reports').glob('*.csv')} == {
        'churn_precision_checks.csv', 'churn_precision_metrics.csv',
        'churn_precision_thresholds.csv', 'churn_precision_decision.csv'}
    for table in reports.values():
        assert 'client_id' not in table.columns
        assert 'membership_id' not in table.columns
        assert 'Churn' not in table.columns
        assert not any('probability' in c.lower() for c in table.columns)


def test_excluded_leakage_features_are_never_used():
    x = sample_external()
    x['Lifetime'] = 1000 * x['Churn']
    x['Churn_Probability'] = x['Churn'].astype(float)
    reports = evaluate_precision_optimization(x)
    assert 'Lifetime' not in EXPERIMENT_FEATURES
    assert 'Churn_Probability' not in EXPERIMENT_FEATURES
    assert bool(reports['decision'].loc[0, 'test_evaluated'])


def test_original_input_frame_unmodified():
    x = sample_external()
    before = x.copy(deep=True)
    evaluate_precision_optimization(x)
    pd.testing.assert_frame_equal(x, before)


def test_missing_required_columns_rejected():
    with pytest.raises(ValueError, match='missing columns'):
        evaluate_precision_optimization(sample_external().drop(columns=['Partner']))


def test_non_binary_target_rejected():
    x = sample_external()
    x.loc[0, TARGET] = 12
    with pytest.raises(ValueError, match='binary labels'):
        evaluate_precision_optimization(x)


def test_confusion_counts_match_examples():
    scores = _confusion(np.array([0, 0, 1, 1]), np.array([0.2, 0.8, 0.7, 0.1]), 0.5)
    assert scores['true_positives'] == 1
    assert scores['false_positives'] == 1
    assert scores['true_negatives'] == 1
    assert scores['false_negatives'] == 1


def test_precision_metric_confidence_intervals():
    out = _metric_row('model', np.array([0, 0, 1, 1]),
                      np.array([0.2, 0.8, 0.7, 0.1]), 0.5)
    assert 0 <= out['precision_ci_lower_95'] <= out['precision_ci_upper_95'] <= 1
    assert 0 <= out['recall_ci_lower_95'] <= out['recall_ci_upper_95'] <= 1


def test_business_predictions_never_enabled():
    decision = evaluate_precision_optimization(sample_external())['decision'].iloc[0]
    assert not bool(decision['business_predictions_enabled'])
    assert not bool(decision['chronological_validation'])
    assert not bool(decision['verified_pre_outcome_features'])


def test_forecast_target_not_conflated_with_30_day_churn():
    decision = evaluate_precision_optimization(sample_external())['decision'].iloc[0]
    assert 'no verified feature timestamps' in decision['note'].lower()


def test_thresholds_do_not_contain_test_predictions():
    reports = evaluate_precision_optimization(sample_external())
    assert 'threshold' in reports['thresholds'].columns
    assert 'client_id' not in reports['thresholds'].columns
    assert 'test' not in ' '.join(reports['thresholds'].columns).lower()
