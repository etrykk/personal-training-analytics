#Author: etrykk
#Date: 11/21/2026
#V2 - Predictive Lab aggregate presentation tests

import pandas as pd
import pytest

from src.ml.predictive_lab_reports import (
    load_experimental_summary, research_comparison_table
)


def make_reports(tmp_path):
    pd.DataFrame([{
        'dataset_kind': 'external_cross_sectional_research',
        'source_url': 'https://example.com/research',
        'input_rows': 400, 'train_rows': 300, 'test_rows': 100,
        'chronological_validation': False,
        'verified_pre_outcome_features': False,
        'business_predictions_enabled': False,
        'status': 'research_metrics_only'
    }]).to_csv(tmp_path / 'external_churn_decision.csv', index=False)
    pd.DataFrame([
        {'model': 'majority_class_baseline', 'accuracy': .75, 'balanced_accuracy': .5,
         'precision': 0., 'recall': 0., 'f1': 0., 'roc_auc': .5,
         'true_positives': 0, 'false_positives': 0, 'true_negatives': 75, 'false_negatives': 25},
        {'model': 'logistic_regression', 'accuracy': .69, 'balanced_accuracy': .72,
         'precision': .45, 'recall': .8, 'f1': .58, 'roc_auc': .79,
         'true_positives': 20, 'false_positives': 24, 'true_negatives': 51, 'false_negatives': 5},
    ]).to_csv(tmp_path / 'external_churn_metrics.csv', index=False)
    pd.DataFrame([{'check': 'dataset rows', 'passed': True}]).to_csv(
        tmp_path / 'external_churn_checks.csv', index=False)
    return tmp_path


def test_missing_reports_blocks_display(tmp_path):
    assert load_experimental_summary(tmp_path)['status'] == 'missing'


def test_valid_research_only_reports(tmp_path):
    result = load_experimental_summary(make_reports(tmp_path))
    assert result['status'] == 'research_only'
    assert result['test_rows'] == 100
    assert result['models']['logistic_regression']['true_positives'] == 20


def test_business_predictions_never_enabled(tmp_path):
    result = load_experimental_summary(make_reports(tmp_path))
    assert result['business_predictions_enabled'] is False


def test_refuse_decision_with_prediction_enabled(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_decision.csv'
    d = pd.read_csv(p)
    d['business_predictions_enabled'] = True
    d.to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_wrong_dataset_kind(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_decision.csv'
    d = pd.read_csv(p)
    d['dataset_kind'] = 'business_data'
    d.to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_wrong_status(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_decision.csv'
    d = pd.read_csv(p)
    d['status'] = 'production'
    d.to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_missing_columns(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_metrics.csv'
    d = pd.read_csv(p).drop(columns=['true_positives'])
    d.to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_inconsistent_counts(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_metrics.csv'
    d = pd.read_csv(p)
    d.loc[d['model']=='logistic_regression','false_positives'] = 25
    d.to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_failed_checks(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_checks.csv'
    pd.DataFrame([{'check': 'holdout', 'passed': False}]).to_csv(p, index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_refuse_duplicate_model_entries(tmp_path):
    make_reports(tmp_path)
    p = tmp_path / 'external_churn_metrics.csv'
    d = pd.read_csv(p)
    d = pd.concat([d,d.iloc[[1]]],ignore_index=True)
    d.to_csv(p,index=False)
    assert load_experimental_summary(tmp_path)['status'] == 'incomplete'


def test_comparison_table_has_only_aggregate_metrics(tmp_path):
    d = research_comparison_table(load_experimental_summary(make_reports(tmp_path)))
    assert len(d) == 2
    assert all('client' not in str(c).lower() for c in d.columns)
    assert d.loc[1, 'Recall'] == pytest.approx(.8)


def test_incomplete_comparison_is_empty():
    assert research_comparison_table({'status': 'incomplete'}).empty


def test_no_unwanted_data_leakage_in_dashboard_file():
    #Check actual Predictive Lab dependencies rather than exact display wording.
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'app' / 'dashboard_v2.py'
    module = ast.parse(path.read_text(encoding='utf-8'))
    page = next((node for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == 'experimental_page'), None)
    assert page is not None, 'Predictive Lab page is missing from dashboard_v2.py'

    called_functions = set()
    for node in ast.walk(page):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_functions.add(node.func.attr)

    assert 'load_experimental_summary' in called_functions, (
        'Expected the Phase 4F dashboard that loads aggregate experiment reports')
    assert 'research_comparison_table' in called_functions
    assert not called_functions.intersection({
        'fit', 'predict', 'predict_proba', 'read_csv', 'load_processed_data'
    }), 'Predictive Lab must not load client records or train/score individual clients'
