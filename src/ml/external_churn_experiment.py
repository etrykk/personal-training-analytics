#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4E: External Gym Churn Research Experiment

"""Evaluate a separate public gym-membership dataset as an ML research exercise.

IMPORTANT: This is a retrospective, cross-sectional experiment. The dataset
has no verified prediction-date / feature-availability timestamps, so results
MUST NOT be described as prospective 30-day churn accuracy. No data is combined
with the user's business. No client predictions or model files are saved.

Obtain gym_churn_us.csv independently from the linked Kaggle dataset and save
it to data/external/gym_churn_us.csv. Do not commit the raw CSV to GitHub until
its redistribution rights and privacy properties have been verified.
"""

import argparse
from pathlib import Path

import pandas as pd

from .churn_classifier import _scores
from .evaluation_review import wilson_interval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / 'data' / 'external' / 'gym_churn_us.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'external_experiment'
SOURCE_URL = 'https://www.kaggle.com/datasets/adrianvinueza/gym-customers-features-and-churn'

#Only pre-specified structural/enrollment-related fields. These are NOT
#verified as timestamped pre-cancellation observations, so this experiment
#cannot support deployment or claims about future prediction performance.
EXPERIMENT_FEATURES = [
    'Near_Location', 'Partner', 'Promo_friends', 'Phone',
    'Contract_period', 'Group_visits',
]
TARGET = 'Churn'
MIN_ROWS = 200
MIN_POSITIVE = 30
MIN_NEGATIVE = 30
MIN_HOLDOUT_CLASS = 10
CHECK_COLUMNS = ['check', 'observed', 'minimum', 'passed']
METRIC_COLUMNS = ['model', 'accuracy', 'balanced_accuracy', 'precision',
                  'recall', 'f1', 'roc_auc', 'average_precision',
                  'true_positives', 'false_positives', 'true_negatives',
                  'false_negatives', 'recall_ci_lower_95',
                  'recall_ci_upper_95', 'precision_ci_lower_95',
                  'precision_ci_upper_95']
DECISION_COLUMNS = ['dataset_kind', 'source_url', 'input_rows', 'positive_rows',
                    'negative_rows', 'train_rows', 'test_rows', 'models_evaluated',
                    'exploratory_targets_met', 'chronological_validation',
                    'verified_pre_outcome_features',
                    'business_predictions_enabled', 'status', 'note']


def validate_external_data(data):
    """Check the narrowly specified source schema and numeric values.

    Never infer features from arbitrary columns: the target or downstream
    outcome signals could otherwise leak into model features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('External data must be a pandas DataFrame')
    if not data.columns.is_unique:
        raise ValueError('External CSV contains duplicate column names')
    required = EXPERIMENT_FEATURES + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'External CSV missing columns: {missing}')
    if data.empty:
        raise ValueError('External CSV is empty')
    if 'client_id' in data.columns and data['client_id'].duplicated().any():
        raise ValueError('Repeated client_id values cannot use random row splitting')
    if 'customer_id' in data.columns and data['customer_id'].duplicated().any():
        raise ValueError('Repeated customer_id values cannot use random row splitting')

    cleaned = data[required].copy(deep=True)
    target = pd.to_numeric(cleaned[TARGET], errors='coerce')
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError('Churn must contain only complete binary labels 0 and 1')
    cleaned[TARGET] = target.astype(int)
    for feature in EXPERIMENT_FEATURES:
        numeric = pd.to_numeric(cleaned[feature], errors='coerce')
        if (cleaned[feature].notna() & numeric.isna()).any():
            raise ValueError(f'{feature} contains nonnumeric values')
        if (numeric.dropna() < 0).any():
            raise ValueError(f'{feature} contains negative values')
        if numeric.notna().sum() == 0:
            raise ValueError(f'{feature} contains no usable numeric values')
        cleaned[feature] = numeric.astype(float)
    return cleaned


def _checks(data):
    """Use counts as a technical screen, not an approval to deploy ML."""
    positives = int(data[TARGET].sum())
    negatives = int(len(data) - positives)
    pairs = [('dataset rows', len(data), MIN_ROWS),
             ('positive churn labels', positives, MIN_POSITIVE),
             ('negative churn labels', negatives, MIN_NEGATIVE)]
    return pd.DataFrame([{'check': name, 'observed': int(value),
                          'minimum': int(minimum), 'passed': bool(value >= minimum)}
                         for name, value, minimum in pairs], columns=CHECK_COLUMNS)


def _empty_reports(status, note, source_rows=0, positives=0, negatives=0,
                   checks=None, train_rows=0, test_rows=0):
    if checks is None:
        checks = pd.DataFrame(columns=CHECK_COLUMNS)
    decision = pd.DataFrame([{
        'dataset_kind': 'external_cross_sectional_research', 'source_url': SOURCE_URL,
        'input_rows': int(source_rows), 'positive_rows': int(positives),
        'negative_rows': int(negatives), 'train_rows': int(train_rows),
        'test_rows': int(test_rows), 'models_evaluated': False,
        'exploratory_targets_met': False, 'chronological_validation': False,
        'verified_pre_outcome_features': False,
        'business_predictions_enabled': False, 'status': status, 'note': note,
    }], columns=DECISION_COLUMNS)
    return {'checks': checks, 'metrics': pd.DataFrame(columns=METRIC_COLUMNS),
            'decision': decision}


def evaluate_external_experiment(data, test_size=0.25, random_state=42):
    """Run a stratified retrospective holdout with training-only preprocessing.

    Separate from 30-day future cancellation: even high numbers here would not
    imply that the business-specific prediction model is ready or accurate.
    """
    clean = validate_external_data(data).reset_index(drop=True)
    checks = _checks(clean)
    positive = int(clean[TARGET].sum())
    negative = int(len(clean) - positive)
    if not isinstance(test_size, (int, float)) or not 0.1 <= test_size <= 0.4:
        raise ValueError('test_size must be between 0.1 and 0.4')
    if not bool(checks['passed'].all()):
        return _empty_reports('blocked_insufficient_external_examples',
                              'Insufficient independent labeled records for even a research holdout.',
                              len(clean), positive, negative, checks)

    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_index, test_index = train_test_split(
        clean.index.to_numpy(), test_size=test_size, random_state=random_state,
        stratify=clean[TARGET].to_numpy(), shuffle=True)
    train, test = clean.loc[train_index], clean.loc[test_index]
    pos_test, neg_test = int(test[TARGET].sum()), int(len(test) - test[TARGET].sum())
    checks = pd.concat([checks, pd.DataFrame([
        {'check': 'holdout positive labels', 'observed': pos_test,
         'minimum': MIN_HOLDOUT_CLASS, 'passed': pos_test >= MIN_HOLDOUT_CLASS},
        {'check': 'holdout negative labels', 'observed': neg_test,
         'minimum': MIN_HOLDOUT_CLASS, 'passed': neg_test >= MIN_HOLDOUT_CLASS},
    ])], ignore_index=True)
    if not bool(checks['passed'].all()):
        return _empty_reports('blocked_weak_holdout',
                              'A stratified holdout lacks enough examples of both classes.',
                              len(clean), positive, negative, checks, len(train), len(test))

    x_train, y_train = train[EXPERIMENT_FEATURES], train[TARGET]
    x_test, y_test = test[EXPERIMENT_FEATURES], test[TARGET]
    models = {
        'majority_class_baseline': DummyClassifier(strategy='prior'),
        'logistic_regression': make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    }
    rows = []
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_test)[:, list(estimator.classes_).index(1)]
        predicted = (probabilities >= 0.5).astype(int)
        scores = _scores(name, y_test, predicted, probabilities)
        tp = scores['true_positives']
        fp = scores['false_positives']
        fn = scores['false_negatives']
        recall_ci = wilson_interval(tp, tp + fn)
        precision_ci = wilson_interval(tp, tp + fp)
        rows.append({**scores,
                     'recall_ci_lower_95': recall_ci[0], 'recall_ci_upper_95': recall_ci[1],
                     'precision_ci_lower_95': precision_ci[0],
                     'precision_ci_upper_95': precision_ci[1]})
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    base = metrics.loc[metrics['model'].eq('majority_class_baseline')].iloc[0]
    logreg = metrics.loc[metrics['model'].eq('logistic_regression')].iloc[0]
    explored = bool(logreg['recall'] >= 0.60 and logreg['precision'] >= 0.30 and
                    logreg['f1'] > base['f1'] and
                    logreg['balanced_accuracy'] > base['balanced_accuracy'])
    return {'checks': checks, 'metrics': metrics,
            'decision': pd.DataFrame([{
                'dataset_kind': 'external_cross_sectional_research',
                'source_url': SOURCE_URL, 'input_rows': len(clean),
                'positive_rows': positive, 'negative_rows': negative,
                'train_rows': len(train), 'test_rows': len(test),
                'models_evaluated': True, 'exploratory_targets_met': explored,
                'chronological_validation': False,
                'verified_pre_outcome_features': False,
                'business_predictions_enabled': False,
                'status': 'research_metrics_only',
                'note': ('Retrospective random holdout; no reliable feature timestamps or '
                         '30-day prediction labels. Not evidence of real-world churn prediction.'),
            }], columns=DECISION_COLUMNS)}


def run_analysis(source_path=DEFAULT_SOURCE, output_dir=OUTPUT_DIR):
    """Generate aggregate CSV reports; missing external file is a safe block."""
    source_path = Path(source_path)
    if source_path.exists():
        reports = evaluate_external_experiment(pd.read_csv(source_path))
    else:
        reports = _empty_reports('external_dataset_missing',
            f'Download the source CSV from {SOURCE_URL} and place it at {source_path}.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in reports.items():
        frame.to_csv(output_dir / f'external_churn_{key}.csv', index=False)
    print('Phase 4E research reports saved to:', output_dir)
    print(reports['decision'].to_string(index=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='External gym churn research only')
    parser.add_argument('--input', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_analysis(args.input, args.output_dir)
