#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Predictive Lab research report presentation (aggregate metrics only)

from pathlib import Path
import pandas as pd

REQUIRED_DECISION = {
    'dataset_kind', 'source_url', 'input_rows', 'train_rows', 'test_rows',
    'chronological_validation', 'verified_pre_outcome_features',
    'business_predictions_enabled', 'status'
}
REQUIRED_METRICS = {
    'model', 'accuracy', 'balanced_accuracy', 'precision', 'recall',
    'f1', 'roc_auc', 'true_positives', 'false_positives',
    'true_negatives', 'false_negatives'
}
REQUIRED_CHECKS = {'check', 'passed'}
MODELS = ('majority_class_baseline', 'logistic_regression')


def _is_true(value):
    # CSV booleans may be parsed as bools or strings.
    return str(value).strip().lower() == 'true'


def load_experimental_summary(output_dir):
    """Read research-only reports; never return individual client records.

    status: missing | incomplete | research_only
    """
    directory = Path(output_dir)
    paths = {kind: directory / f'external_churn_{kind}.csv'
             for kind in ('decision', 'metrics', 'checks')}
    if any(not file.is_file() for file in paths.values()):
        return {'status': 'missing', 'business_predictions_enabled': False}
    try:
        decisions = pd.read_csv(paths['decision'])
        metrics = pd.read_csv(paths['metrics'])
        checks = pd.read_csv(paths['checks'])
    except (OSError, ValueError, pd.errors.ParserError):
        return {'status': 'incomplete', 'business_predictions_enabled': False}
    if (len(decisions) != 1 or not REQUIRED_DECISION.issubset(decisions.columns)
            or not REQUIRED_METRICS.issubset(metrics.columns)
            or not REQUIRED_CHECKS.issubset(checks.columns)):
        return {'status': 'incomplete', 'business_predictions_enabled': False}

    record = decisions.iloc[0]
    if (str(record['dataset_kind']) != 'external_cross_sectional_research'
            or str(record['status']) != 'research_metrics_only'
            or _is_true(record['business_predictions_enabled'])):
        return {'status': 'incomplete', 'business_predictions_enabled': False}
    by_model = metrics.set_index('model', drop=False)
    if not all(name in by_model.index for name in MODELS) or not by_model.index.is_unique:
        return {'status': 'incomplete', 'business_predictions_enabled': False}
    if checks.empty or not checks['passed'].map(_is_true).all():
        return {'status': 'incomplete', 'business_predictions_enabled': False}
    try:
        n_train, n_test = int(record['train_rows']), int(record['test_rows'])
        n_total = int(record['input_rows'])
        m = {}
        for name in MODELS:
            row = by_model.loc[name]
            m[name] = {field: float(row[field]) for field in
                       ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc')}
            m[name].update({field: int(row[field]) for field in
                            ('true_positives', 'false_positives',
                             'true_negatives', 'false_negatives')})
            if (any(not 0 <= m[name][f] <= 1 for f in
                    ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
                    or any(m[name][f] < 0 for f in
                           ('true_positives', 'false_positives', 'true_negatives', 'false_negatives'))
                    or sum(m[name][f] for f in
                           ('true_positives', 'false_positives', 'true_negatives', 'false_negatives')) != n_test):
                return {'status': 'incomplete', 'business_predictions_enabled': False}
        if n_total < n_train+n_test or min(n_train, n_test) <= 0:
            return {'status': 'incomplete', 'business_predictions_enabled': False}
    except (ValueError, TypeError, KeyError, OverflowError):
        return {'status': 'incomplete', 'business_predictions_enabled': False}
    return {
        'status': 'research_only',
        'dataset_kind': str(record['dataset_kind']),
        'source_url': str(record['source_url']),
        'input_rows': n_total,
        'train_rows': n_train,
        'test_rows': n_test,
        'chronological_validation': _is_true(record['chronological_validation']),
        'verified_pre_outcome_features': _is_true(record['verified_pre_outcome_features']),
        'business_predictions_enabled': False,
        'models': m,
    }


def research_comparison_table(summary):
    """Return an aggregate display table. Never expose client-level information."""
    if summary.get('status') != 'research_only':
        return pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Precision',
                                     'F1', 'Balanced accuracy', 'ROC-AUC'])
    labels = [('majority_class_baseline', 'Majority-class baseline'),
              ('logistic_regression', 'Logistic Regression')]
    rows = []
    for name, display in labels:
        entry = summary['models'][name]
        rows.append({'Model': display,
                     'Accuracy': entry['accuracy'],
                     'Recall': entry['recall'],
                     'Precision': entry['precision'],
                     'F1': entry['f1'],
                     'Balanced accuracy': entry['balanced_accuracy'],
                     'ROC-AUC': entry['roc_auc']})
    return pd.DataFrame(rows)
