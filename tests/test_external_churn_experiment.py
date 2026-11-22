#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4E: External Research Classifier Tests

import numpy as np
import pandas as pd
import pytest

from src.ml.external_churn_experiment import (
    EXPERIMENT_FEATURES, CHECK_COLUMNS, METRIC_COLUMNS, DECISION_COLUMNS,
    evaluate_external_experiment, run_analysis, validate_external_data,
)


def sample_external(n=480):
    #Synthetic TEST FIXTURE only. Not a real external dataset or proof of quality.
    rng = np.random.default_rng(46)
    frame = pd.DataFrame({
        'Near_Location': rng.integers(0, 2, n),
        'Partner': rng.integers(0, 2, n),
        'Promo_friends': rng.integers(0, 2, n),
        'Phone': rng.integers(0, 2, n),
        'Contract_period': rng.choice([1, 6, 12], size=n),
        'Group_visits': rng.integers(0, 2, n),
    })
    #Deliberately artificial binary outcomes to test pipeline calculations.
    prob = np.where(frame['Contract_period'].eq(1), 0.48, 0.19)
    frame['Churn'] = (rng.random(n) < prob).astype(int)
    return frame


def test_schema_validates_correct_features():
    out = validate_external_data(sample_external())
    assert list(out) == EXPERIMENT_FEATURES + ['Churn']
    assert set(out['Churn']) == {0, 1}


def test_input_is_not_mutated():
    x = sample_external()
    before = x.copy(deep=True)
    evaluate_external_experiment(x)
    pd.testing.assert_frame_equal(x, before)


def test_missing_column_is_rejected():
    x = sample_external().drop(columns=['Partner'])
    with pytest.raises(ValueError, match='missing columns'):
        validate_external_data(x)


def test_non_dataframe_is_rejected():
    with pytest.raises(TypeError):
        validate_external_data([])


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError):
        validate_external_data(sample_external().iloc[:0])


def test_non_binary_target_is_rejected():
    x = sample_external()
    x.loc[0, 'Churn'] = 2
    with pytest.raises(ValueError, match='binary labels'):
        validate_external_data(x)


def test_missing_target_is_rejected():
    x = sample_external()
    x.loc[0, 'Churn'] = np.nan
    with pytest.raises(ValueError, match='binary labels'):
        validate_external_data(x)


def test_non_numeric_feature_is_rejected():
    x = sample_external()
    x['Group_visits'] = x['Group_visits'].astype(object)
    x.loc[0, 'Group_visits'] = 'not a number'
    with pytest.raises(ValueError, match='nonnumeric'):
        validate_external_data(x)


def test_negative_features_rejected():
    x = sample_external()
    x.loc[0, 'Near_Location'] = -1
    with pytest.raises(ValueError, match='negative'):
        validate_external_data(x)


def test_all_missing_feature_rejected():
    x = sample_external()
    x['Promo_friends'] = float('nan')
    with pytest.raises(ValueError, match='no usable'):
        validate_external_data(x)


def test_duplicate_ids_rejected_when_present():
    x = sample_external()
    x['client_id'] = ['C1'] * len(x)
    with pytest.raises(ValueError, match='Repeated client_id'):
        validate_external_data(x)


def test_insufficient_data_blocks_training():
    out = evaluate_external_experiment(sample_external(60))
    assert not bool(out['decision'].loc[0, 'models_evaluated'])
    assert out['metrics'].empty
    assert out['decision'].loc[0, 'status'] == 'blocked_insufficient_external_examples'


def test_one_class_blocks_training():
    x = sample_external()
    x['Churn'] = 0
    out = evaluate_external_experiment(x)
    assert out['metrics'].empty
    assert not bool(out['decision'].loc[0, 'business_predictions_enabled'])


def test_external_models_evaluated_with_aggregate_metrics():
    out = evaluate_external_experiment(sample_external())
    assert set(out['metrics']['model']) == {'majority_class_baseline', 'logistic_regression'}
    assert list(out['metrics']) == METRIC_COLUMNS
    assert list(out['checks']) == CHECK_COLUMNS
    assert list(out['decision']) == DECISION_COLUMNS
    assert out['decision'].loc[0, 'status'] == 'research_metrics_only'
    assert bool(out['decision'].loc[0, 'models_evaluated'])
    assert not bool(out['decision'].loc[0, 'business_predictions_enabled'])
    assert not bool(out['decision'].loc[0, 'chronological_validation'])
    assert not bool(out['decision'].loc[0, 'verified_pre_outcome_features'])


def test_confusion_counts_reconcile_with_test_examples():
    out = evaluate_external_experiment(sample_external())
    test_rows = int(out['decision'].loc[0, 'test_rows'])
    for _, row in out['metrics'].iterrows():
        total = sum(row[k] for k in (
            'true_positives', 'false_positives', 'true_negatives', 'false_negatives'))
        assert total == test_rows
        assert 0 <= row['recall_ci_lower_95'] <= row['recall_ci_upper_95'] <= 1


def test_runs_repeatably_with_same_seed():
    x = sample_external()
    a = evaluate_external_experiment(x, random_state=42)
    b = evaluate_external_experiment(x, random_state=42)
    pd.testing.assert_frame_equal(a['metrics'], b['metrics'])


def test_invalid_test_fraction_rejected():
    with pytest.raises(ValueError, match='test_size'):
        evaluate_external_experiment(sample_external(), test_size=0.8)


def test_nan_feature_imputed_from_training_data():
    x = sample_external()
    x.loc[0:19, 'Partner'] = np.nan
    out = evaluate_external_experiment(x)
    assert len(out['metrics']) == 2


def test_duplicate_input_index_is_safe():
    x = sample_external()
    x.index = [0] * len(x)
    out = evaluate_external_experiment(x)
    assert len(out['metrics']) == 2


def test_missing_csv_writes_safe_reports(tmp_path):
    results = run_analysis(tmp_path / 'missing.csv', tmp_path / 'reports')
    assert results['decision'].loc[0, 'status'] == 'external_dataset_missing'
    assert not bool(results['decision'].loc[0, 'business_predictions_enabled'])
    assert len(list((tmp_path / 'reports').glob('external_churn_*.csv'))) == 3


def test_existing_csv_writes_aggregate_only_reports(tmp_path):
    path = tmp_path / 'gym_churn_us.csv'
    sample_external().to_csv(path, index=False)
    reports = run_analysis(path, tmp_path / 'out')
    assert bool(reports['decision'].loc[0, 'models_evaluated'])
    filenames = {p.name for p in (tmp_path / 'out').glob('*.csv')}
    assert filenames == {'external_churn_checks.csv',
                         'external_churn_metrics.csv',
                         'external_churn_decision.csv'}
    for frame in reports.values():
        assert 'client_id' not in frame.columns
        assert 'membership_id' not in frame.columns


def test_future_leakage_columns_are_excluded():
    x = sample_external()
    x['Churn_Probability'] = x['Churn'].astype(float)
    x['Lifetime'] = 100 * x['Churn']
    x['Avg_class_frequency_current_month'] = 10 * x['Churn']
    clean = validate_external_data(x)
    assert 'Lifetime' not in clean.columns
    assert 'Churn_Probability' not in clean.columns
    assert 'Avg_class_frequency_current_month' not in clean.columns
