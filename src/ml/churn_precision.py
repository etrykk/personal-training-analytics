#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4G: Experimental Churn Precision Optimization

"""Retrospective precision/recall tuning using a genuinely untouched holdout.

Research ONLY. This is NOT the business-specific 30-day churn model.
The public cross-sectional dataset lacks verified pre-outcome feature timestamps.

A stratified 60/20/20 train/validation/test split is used:
- Fit candidate Logistic Regression pipelines only on train data.
- Select class weight, regularization and threshold using validation data.
- Evaluate the chosen settings ONCE on the untouched test partition.
- Compare the majority baseline, original balanced logistic model at 0.50,
  and optimized logistic model on the SAME final test partition.

The new partition differs from Phase 4E's 75/25 split, so results from the
old 68.7%/44.9% study are NOT directly comparable to this new test.

Never mix external examples with individual-business training. No client-level
records, predictions, training data, or fitted models are exported.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .churn_classifier import _scores
from .evaluation_review import wilson_interval
from .external_churn_experiment import (
    DEFAULT_SOURCE, EXPERIMENT_FEATURES, MIN_NEGATIVE, MIN_POSITIVE,
    MIN_ROWS, SOURCE_URL, TARGET, validate_external_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'precision_optimization'
RANDOM_STATE = 42
MIN_SPLIT_CLASS = 10
RECALL_GOAL = 0.60
PRECISION_GOAL = 0.60
THRESHOLDS = tuple(round(float(v), 2) for v in np.arange(0.30, 0.851, 0.05))
REGULARIZATION_C = (0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, 'balanced')

CHECK_COLUMNS = ['check', 'observed', 'minimum', 'passed']
METRIC_COLUMNS = ['model', 'split', 'threshold', 'class_weight', 'regularization_c',
                  'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
                  'roc_auc', 'average_precision', 'true_positives', 'false_positives',
                  'true_negatives', 'false_negatives', 'recall_ci_lower_95',
                  'recall_ci_upper_95', 'precision_ci_lower_95',
                  'precision_ci_upper_95']
THRESHOLD_COLUMNS = ['threshold', 'validation_predicted_churn', 'precision', 'recall',
                     'f1', 'true_positives', 'false_positives', 'true_negatives',
                     'false_negatives', 'meets_recall_goal', 'meets_precision_goal']
DECISION_COLUMNS = ['dataset_kind', 'source_url', 'input_rows', 'train_rows',
                    'validation_rows', 'test_rows', 'best_threshold',
                    'best_class_weight', 'best_regularization_c', 'selected_on',
                    'test_evaluated', 'validation_recall_goal_met',
                    'test_recall_goal_met', 'test_precision_goal_met',
                    'chronological_validation', 'verified_pre_outcome_features',
                    'business_predictions_enabled', 'status', 'note']


def _blocked(status, note, checks=None, *, input_rows=0, train_rows=0,
             validation_rows=0, test_rows=0):
    if checks is None:
        checks = pd.DataFrame(columns=CHECK_COLUMNS)
    decision = {
        'dataset_kind': 'external_cross_sectional_research',
        'source_url': SOURCE_URL,
        'input_rows': int(input_rows), 'train_rows': int(train_rows),
        'validation_rows': int(validation_rows), 'test_rows': int(test_rows),
        'best_threshold': float('nan'), 'best_class_weight': '',
        'best_regularization_c': float('nan'), 'selected_on': 'validation_only',
        'test_evaluated': False, 'validation_recall_goal_met': False,
        'test_recall_goal_met': False, 'test_precision_goal_met': False,
        'chronological_validation': False, 'verified_pre_outcome_features': False,
        'business_predictions_enabled': False, 'status': status, 'note': note,
    }
    return {'checks': checks,
            'metrics': pd.DataFrame(columns=METRIC_COLUMNS),
            'thresholds': pd.DataFrame(columns=THRESHOLD_COLUMNS),
            'decision': pd.DataFrame([decision], columns=DECISION_COLUMNS)}


def _check_rows(clean):
    n_positive = int(clean[TARGET].sum())
    n_negative = int(len(clean) - n_positive)
    rows = [
        ('dataset rows', len(clean), MIN_ROWS),
        ('positive churn labels', n_positive, MIN_POSITIVE),
        ('negative churn labels', n_negative, MIN_NEGATIVE),
    ]
    return pd.DataFrame([{'check': label, 'observed': int(value),
                          'minimum': int(minimum), 'passed': bool(value >= minimum)}
                         for label, value, minimum in rows], columns=CHECK_COLUMNS)


def _add_split_checks(checks, parts):
    rows = []
    for name, part in parts.items():
        positives = int(part[TARGET].sum())
        negatives = int(len(part) - positives)
        rows += [
            {'check': f'{name} positive labels', 'observed': positives,
             'minimum': MIN_SPLIT_CLASS, 'passed': positives >= MIN_SPLIT_CLASS},
            {'check': f'{name} negative labels', 'observed': negatives,
             'minimum': MIN_SPLIT_CLASS, 'passed': negatives >= MIN_SPLIT_CLASS},
        ]
    return pd.concat([checks, pd.DataFrame(rows, columns=CHECK_COLUMNS)], ignore_index=True)


def _make_logistic(c, weight):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(C=c, class_weight=weight, random_state=RANDOM_STATE,
                           max_iter=1500),
    )


def _probabilities(model, x):
    #Label 1 corresponds to churn; never assume a fixed column ordering.
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def _confusion(y, prob, threshold):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    predicted = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'true_positives': int(tp), 'false_positives': int(fp),
            'true_negatives': int(tn), 'false_negatives': int(fn)}


def _metric_row(name, y, prob, threshold, weight='', c=float('nan')):
    pred = (prob >= threshold).astype(int)
    scores = _scores(name, y, pred, prob)
    recall_ci = wilson_interval(scores['true_positives'],
                                scores['true_positives'] + scores['false_negatives'])
    precision_ci = wilson_interval(scores['true_positives'],
                                   scores['true_positives'] + scores['false_positives'])
    return {'model': name, 'split': 'untouched_test', 'threshold': threshold,
            'class_weight': 'none' if weight is None else str(weight),
            'regularization_c': c, **scores,
            'recall_ci_lower_95': recall_ci[0], 'recall_ci_upper_95': recall_ci[1],
            'precision_ci_lower_95': precision_ci[0],
            'precision_ci_upper_95': precision_ci[1]}


def evaluate_precision_optimization(data, random_state=RANDOM_STATE):
    """Choose settings on validation data; inspect test exactly once afterward.

    Returns only aggregate reporting DataFrames, never a fitted model.
    An unmet precision goal is reported faithfully; it does not imply a defect.
    """
    clean = validate_external_data(data).reset_index(drop=True)
    checks = _check_rows(clean)
    n_total = len(clean)
    if not bool(checks['passed'].all()):
        return _blocked('blocked_insufficient_external_examples',
                        'Not enough independent labeled examples for validation.',
                        checks, input_rows=n_total)

    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier

    #No time index exists, so random splitting is ONLY exploratory.
    train_val_idx, test_idx = train_test_split(
        np.arange(n_total), test_size=0.20, stratify=clean[TARGET],
        random_state=random_state)
    train_idx, validation_idx = train_test_split(
        train_val_idx, test_size=0.25,
        stratify=clean.loc[train_val_idx, TARGET], random_state=random_state)
    train, val, test = (clean.iloc[indices].reset_index(drop=True)
                        for indices in (train_idx, validation_idx, test_idx))
    parts = {'training': train, 'validation': val, 'test': test}
    checks = _add_split_checks(checks, parts)
    sizes = {'input_rows': n_total, 'train_rows': len(train),
             'validation_rows': len(val), 'test_rows': len(test)}
    if not bool(checks['passed'].all()):
        return _blocked('blocked_weak_split',
                        'At least one split lacks sufficient churn and non-churn cases.',
                        checks, **sizes)

    x_train, y_train = train[EXPERIMENT_FEATURES], train[TARGET]
    x_val, y_val = val[EXPERIMENT_FEATURES], val[TARGET]
    x_test, y_test = test[EXPERIMENT_FEATURES], test[TARGET]

    #All tuning uses VALIDATION only. No test labels/probabilities participate.
    candidates = []
    for weight in CLASS_WEIGHTS:
        for c in REGULARIZATION_C:
            estimator = _make_logistic(c, weight)
            estimator.fit(x_train, y_train)
            val_prob = _probabilities(estimator, x_val)
            for threshold in THRESHOLDS:
                s = _confusion(y_val, val_prob, threshold)
                if s['recall'] >= RECALL_GOAL:
                    candidates.append((s['precision'], s['f1'], s['recall'],
                                       threshold, c, weight, estimator, val_prob))

    if not candidates:
        return _blocked('blocked_no_validation_threshold',
                        'No validation threshold attained the minimum recall goal.',
                        checks, **sizes)

    #Validation objective: maximize precision subject to recall >=60%,
    #with F1 then recall as tie breakers. No test-set tuning.
    selected = max(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
    val_precision, _val_f1, val_recall, threshold, c, weight, tuned_model, val_prob = selected

    curve = []
    for t in THRESHOLDS:
        score = _confusion(y_val, val_prob, t)
        curve.append({'threshold': t, 'validation_predicted_churn':
                      score['true_positives'] + score['false_positives'],
                      **score, 'meets_recall_goal': score['recall'] >= RECALL_GOAL,
                      'meets_precision_goal': score['precision'] >= PRECISION_GOAL})
    thresholds = pd.DataFrame(curve, columns=THRESHOLD_COLUMNS)

    #Untouched test split starts here; no reselection after viewing results.
    baseline = DummyClassifier(strategy='prior')
    baseline.fit(x_train, y_train)
    baseline_prob = _probabilities(baseline, x_test)
    default_model = _make_logistic(1.0, 'balanced')
    default_model.fit(x_train, y_train)
    default_prob = _probabilities(default_model, x_test)
    tuned_prob = _probabilities(tuned_model, x_test)
    metrics = pd.DataFrame([
        _metric_row('majority_class_baseline', y_test, baseline_prob, 0.5,
                    weight='majority_baseline'),
        _metric_row('default_logistic_regression', y_test, default_prob, 0.5,
                    weight='balanced', c=1.0),
        _metric_row('tuned_logistic_regression', y_test, tuned_prob, threshold,
                    weight=weight, c=c),
    ], columns=METRIC_COLUMNS)
    tuned = metrics.loc[metrics['model'] == 'tuned_logistic_regression'].iloc[0]
    decision = pd.DataFrame([{
        'dataset_kind': 'external_cross_sectional_research', 'source_url': SOURCE_URL,
        **sizes, 'best_threshold': threshold,
        'best_class_weight': 'none' if weight is None else str(weight),
        'best_regularization_c': c, 'selected_on': 'validation_only',
        'test_evaluated': True, 'validation_recall_goal_met': bool(val_recall >= RECALL_GOAL),
        'test_recall_goal_met': bool(tuned['recall'] >= RECALL_GOAL),
        'test_precision_goal_met': bool(tuned['precision'] >= PRECISION_GOAL),
        'chronological_validation': False, 'verified_pre_outcome_features': False,
        'business_predictions_enabled': False, 'status': 'research_metrics_only',
        'note': ('Retrospective 60/20/20 random splits; selected on validation only. '
                 'Untouched test assessed once. No verified feature timestamps or '
                 '30-day labels; do not deploy or compare directly with the Phase 4E split.'),
    }], columns=DECISION_COLUMNS)
    return {'checks': checks, 'metrics': metrics, 'thresholds': thresholds,
            'decision': decision}


def run_analysis(source_path=DEFAULT_SOURCE, output_dir=OUTPUT_DIR):
    """Generate research-only CSV reports; missing external file blocks safely."""
    source_path = Path(source_path)
    if source_path.is_file():
        reports = evaluate_precision_optimization(pd.read_csv(source_path))
    else:
        reports = _blocked('external_dataset_missing',
                           f'Place the independent research CSV at {source_path}.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in reports.items():
        frame.to_csv(output_dir / f'churn_precision_{key}.csv', index=False)
    print('Phase 4G research reports saved to:', output_dir)
    print(reports['decision'].to_string(index=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize precision in a research-only churn experiment')
    parser.add_argument('--input', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    arguments = parser.parse_args()
    run_analysis(arguments.input, arguments.output_dir)
