#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4D: Experimental Model Evaluation Review

"""Review aggregate Phase 4C classification reports without training a model.

This is an experimental, descriptive evaluation gate. It does not turn on
business-specific predictions, and it never exports client-level records.
Metric thresholds are exploratory targets, NOT a deployment approval.
"""

import argparse
from math import sqrt
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'classification'
EVALUATION_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'evaluation'

CHECK_COLUMNS = {'check', 'observed', 'minimum_screening_requirement', 'passed'}
DECISION_COLUMNS = {'cutoff_date', 'train_rows', 'test_rows', 'training_approved', 'status'}
METRIC_COLUMNS = {
    'model', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
    'roc_auc', 'average_precision', 'true_positives', 'false_positives',
    'true_negatives', 'false_negatives',
}
MODEL_NAMES = {'majority_class_baseline', 'logistic_regression'}
SCORE_COLUMNS = ('accuracy', 'balanced_accuracy', 'precision', 'recall',
                 'f1', 'roc_auc', 'average_precision')
COUNT_COLUMNS = ('true_positives', 'false_positives', 'true_negatives',
                 'false_negatives')
REVIEW_COLUMNS = [
    'model', 'test_examples', 'actual_cancellations', 'accuracy',
    'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc',
    'average_precision', 'recall_ci_lower_95', 'recall_ci_upper_95',
    'precision_ci_lower_95', 'precision_ci_upper_95',
    'recall_change_vs_baseline', 'f1_change_vs_baseline',
    'balanced_accuracy_change_vs_baseline',
]
AUDIT_COLUMNS = ['check', 'evaluated', 'observed', 'requirement', 'passed']


def _require_columns(frame, required, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{name} must be a pandas DataFrame')
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'{name} missing required columns: {missing}')


def _as_bool(value):
    """Interpret CSV booleans safely (bool('False') would be True)."""
    word = str(value).strip().lower()
    if word in ('true', '1'):
        return True
    if word in ('false', '0'):
        return False
    raise ValueError(f'Invalid boolean value: {value!r}')


def _as_nonnegative_int(value, label):
    number = pd.to_numeric(value, errors='coerce')
    if pd.isna(number) or number < 0 or number % 1 != 0:
        raise ValueError(f'{label} must be a nonnegative whole number')
    return int(number)


def wilson_interval(successes, trials, z=1.96):
    """Return a 95% Wilson binomial interval; NaN when denominator is zero."""
    successes = _as_nonnegative_int(successes, 'successes')
    trials = _as_nonnegative_int(trials, 'trials')
    if successes > trials:
        raise ValueError('successes cannot exceed trials')
    if trials == 0:
        return float('nan'), float('nan')
    p = successes / trials
    d = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / d
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return max(0.0, center - half), min(1.0, center + half)


def _check_metric_rows(metrics, test_rows):
    """Check consistency before displaying model performance summaries."""
    _require_columns(metrics, METRIC_COLUMNS, 'metrics')
    if len(metrics) != 2 or set(metrics['model']) != MODEL_NAMES:
        raise ValueError('metrics must contain one baseline and one Logistic Regression row')
    validated = metrics.copy(deep=True)
    for col in SCORE_COLUMNS:
        values = pd.to_numeric(validated[col], errors='coerce')
        if values.isna().any() or (~values.between(0, 1)).any():
            raise ValueError(f'Invalid score values in {col}')
        validated[col] = values.astype(float)
    for col in COUNT_COLUMNS:
        validated[col] = validated[col].map(lambda v: _as_nonnegative_int(v, col))
    totals = validated[list(COUNT_COLUMNS)].sum(axis=1)
    if not totals.eq(test_rows).all():
        raise ValueError('Confusion matrix totals must equal test_rows')
    pos = validated['true_positives'] + validated['false_negatives']
    neg = validated['true_negatives'] + validated['false_positives']
    if not (pos.nunique() == 1 and neg.nunique() == 1 and pos.iloc[0] > 0 and neg.iloc[0] > 0):
        raise ValueError('The models must use the same test set containing both outcomes')
    for _, row in validated.iterrows():
        tp, fp, tn, fn = (row['true_positives'], row['false_positives'],
                          row['true_negatives'], row['false_negatives'])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        accuracy = (tp + tn) / test_rows
        balanced = (recall + tn / (tn + fp)) / 2
        for col, expected in [('precision', precision), ('recall', recall),
                              ('f1', f1), ('accuracy', accuracy),
                              ('balanced_accuracy', balanced)]:
            if abs(row[col] - expected) > 1e-6:
                raise ValueError(f'Inconsistent {col} for {row["model"]}')
    return validated


def review_model_evaluation(checks, decision, metrics, min_recall=0.60,
                            min_precision=0.30):
    """Create aggregate model comparison, experimental checks, and decision.

    min_recall/min_precision are chosen exploratory thresholds, not clinical,
    financial, or commercial standards. The business prediction flag remains
    False regardless of these metrics.
    """
    _require_columns(checks, CHECK_COLUMNS, 'checks')
    _require_columns(decision, DECISION_COLUMNS, 'decision')
    _require_columns(metrics, METRIC_COLUMNS, 'metrics')
    if len(decision) != 1:
        raise ValueError('decision must have exactly one row')
    for name, threshold in [('min_recall', min_recall), ('min_precision', min_precision)]:
        if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError(f'{name} must be between zero and one')

    approved = _as_bool(decision.loc[decision.index[0], 'training_approved'])
    test_rows = _as_nonnegative_int(decision.iloc[0]['test_rows'], 'test_rows')
    train_rows = _as_nonnegative_int(decision.iloc[0]['train_rows'], 'train_rows')
    screen_checks_pass = checks['passed'].map(_as_bool).all()
    if approved != bool(screen_checks_pass):
        raise ValueError('Upstream training approval disagrees with screening checks')
    if not approved and not metrics.empty:
        raise ValueError('Blocked training must not produce model metrics')

    cutoff = str(decision.iloc[0]['cutoff_date'])
    rows = []
    audit = []
    exploratory_targets_met = False
    status = 'blocked_insufficient_independent_events'

    if approved and metrics.empty:
        status = 'approved_but_metrics_missing'
    elif approved:
        model_metrics = _check_metric_rows(metrics, test_rows)
        baseline = model_metrics.loc[model_metrics['model'] == 'majority_class_baseline'].iloc[0]
        model = model_metrics.loc[model_metrics['model'] == 'logistic_regression'].iloc[0]
        for _, score in model_metrics.iterrows():
            tp, fp, fn = (score['true_positives'], score['false_positives'],
                          score['false_negatives'])
            rec_low, rec_high = wilson_interval(tp, tp + fn)
            pre_low, pre_high = wilson_interval(tp, tp + fp)
            rows.append({
                'model': score['model'], 'test_examples': test_rows,
                'actual_cancellations': tp + fn,
                **{col: score[col] for col in SCORE_COLUMNS},
                'recall_ci_lower_95': rec_low,
                'recall_ci_upper_95': rec_high,
                'precision_ci_lower_95': pre_low,
                'precision_ci_upper_95': pre_high,
                'recall_change_vs_baseline': score['recall'] - baseline['recall'],
                'f1_change_vs_baseline': score['f1'] - baseline['f1'],
                'balanced_accuracy_change_vs_baseline': (
                    score['balanced_accuracy'] - baseline['balanced_accuracy']),
            })
        conditions = [
            ('Recall target (exploratory)', model['recall'], f'>= {min_recall:.0%}',
             model['recall'] >= min_recall),
            ('Precision target (exploratory)', model['precision'], f'>= {min_precision:.0%}',
             model['precision'] >= min_precision),
            ('F1 better than baseline', model['f1'] - baseline['f1'], '> 0',
             model['f1'] > baseline['f1']),
            ('Balanced accuracy better than baseline',
             model['balanced_accuracy'] - baseline['balanced_accuracy'], '> 0',
             model['balanced_accuracy'] > baseline['balanced_accuracy']),
        ]
        audit = [{'check': n, 'evaluated': True, 'observed': float(v),
                  'requirement': rule, 'passed': bool(ok)} for n, v, rule, ok in conditions]
        exploratory_targets_met = all(a['passed'] for a in audit)
        status = ('experimental_metric_targets_met' if exploratory_targets_met
                  else 'experimental_metric_targets_not_met')
    else:
        audit = [{'check': name, 'evaluated': False, 'observed': None,
                  'requirement': rule, 'passed': False} for name, rule in [
            ('Recall target (exploratory)', f'>= {min_recall:.0%}'),
            ('Precision target (exploratory)', f'>= {min_precision:.0%}'),
            ('F1 better than baseline', '> 0'),
            ('Balanced accuracy better than baseline', '> 0'),
        ]]

    summary = [{
        'cutoff_date': cutoff,
        'train_examples': train_rows,
        'test_examples': test_rows,
        'classifier_screening_approved': approved,
        'metrics_evaluated': approved and not metrics.empty,
        'exploratory_targets_met': bool(exploratory_targets_met),
        'business_predictions_enabled': False,
        'status': status,
        'note': ('No predictions enabled. Passing exploratory targets does not prove '
                 'real-world utility: external validation, calibrated probabilities, '
                 'and sufficient independent events are still required.'),
    }]
    return {
        'comparison': pd.DataFrame(rows, columns=REVIEW_COLUMNS),
        'checks': pd.DataFrame(audit, columns=AUDIT_COLUMNS),
        'decision': pd.DataFrame(summary),
    }


def run_analysis(input_dir=CLASSIFICATION_DIR, output_dir=EVALUATION_DIR,
                 min_recall=0.60, min_precision=0.30):
    """Read Phase 4C aggregate CSVs and write privacy-preserving evaluations."""
    input_dir = Path(input_dir)
    source = {
        'checks': pd.read_csv(input_dir / 'churn_classification_checks.csv'),
        'decision': pd.read_csv(input_dir / 'churn_classification_decision.csv'),
        'metrics': pd.read_csv(input_dir / 'churn_classification_metrics.csv'),
    }
    reports = review_model_evaluation(**source, min_recall=min_recall,
                                      min_precision=min_precision)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.to_csv(output_dir / f'churn_evaluation_{name}.csv', index=False)
    print('Experimental evaluation reports saved to:', output_dir)
    print(reports['decision'].to_string(index=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Review aggregate experimental churn scores')
    parser.add_argument('--input-dir', default=str(CLASSIFICATION_DIR))
    parser.add_argument('--output-dir', default=str(EVALUATION_DIR))
    parser.add_argument('--min-recall', type=float, default=0.60)
    parser.add_argument('--min-precision', type=float, default=0.30)
    args = parser.parse_args()
    run_analysis(args.input_dir, args.output_dir, args.min_recall, args.min_precision)
