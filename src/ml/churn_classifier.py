#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform - V2
#Phase 4C: Experimental Churn Classification

"""Safely evaluate membership-cancellation classification *only if data permits*.

This is an EXPERIMENTAL ML module, not a deployed prediction feature.
We deliberately prevent training on the small V1/V2 sample data.

The outcome is a specific membership's recorded cancellation in the 30-day
future horizon. Inactivity, non-renewal, and expiry are NOT treated as churn.
Only confirmed positive/negative examples are eligible for model training.

A chronological holdout is used. Training examples whose LABEL HORIZONS
reach the holdout cutoff are purged. Clients in the holdout are removed
entirely from the training set. This is intentionally conservative and may
reduce training data substantially, especially for smaller businesses.

No client-level predictions, trained models, or probabilities are exported.
"""

import argparse
from pathlib import Path

import pandas as pd

from .historical_churn import MODEL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_PATH = (PROJECT_ROOT / 'outputs' / 'predictive' / 'historical' /
                  'historical_churn_snapshots.csv')
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'predictive' / 'classification'

#Conservative screening thresholds; these are NOT guarantees of usefulness.
MIN_TRAIN_POSITIVE_MEMBERSHIPS = 30
MIN_TEST_POSITIVE_MEMBERSHIPS = 10
MIN_TRAIN_NEGATIVE_MEMBERSHIPS = 30
MIN_TEST_NEGATIVE_MEMBERSHIPS = 10

REQUIRED_COLUMNS = [
    'membership_id', 'client_id', 'snapshot_date', 'horizon_end',
    'churn_within_horizon', 'label_status',
] + MODEL_FEATURE_COLUMNS


def _validate_snapshots(snapshots):
    """Validate the feature/label contract without modifying input data."""
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError('snapshots must be a pandas DataFrame')
    missing = sorted(set(REQUIRED_COLUMNS) - set(snapshots.columns))
    if missing:
        raise ValueError(f'snapshots missing columns: {missing}')

    data = snapshots.copy(deep=True)
    for field in ('snapshot_date', 'horizon_end'):
        data[field] = pd.to_datetime(data[field], errors='coerce')
        if data[field].isna().any():
            raise ValueError(f'{field} contains invalid or missing dates')
    if (data['horizon_end'] <= data['snapshot_date']).any():
        raise ValueError('Each horizon_end must occur after snapshot_date')

    for field in ('membership_id', 'client_id'):
        data[field] = data[field].astype('string').str.strip()
        if data[field].isna().any() or data[field].eq('').any():
            raise ValueError(f'{field} contains missing identifiers')
    if data.duplicated(['membership_id', 'snapshot_date']).any():
        raise ValueError('Duplicate membership_id/snapshot_date observations')

    statuses = data['label_status'].astype('string').str.lower().str.strip()
    if not statuses.isin(['positive', 'negative', 'censored']).all():
        raise ValueError('Unexpected label_status')
    target = pd.to_numeric(data['churn_within_horizon'], errors='coerce')
    raw_target = data['churn_within_horizon']
    if (raw_target.notna() & target.isna()).any():
        raise ValueError('Invalid churn target values')
    if ((statuses.eq('positive') & target.ne(1).fillna(True)) |
        (statuses.eq('negative') & target.ne(0).fillna(True)) |
        (statuses.eq('censored') & target.notna())).any():
        raise ValueError('Inconsistent label_status and churn target')
    data['label_status'] = statuses
    data['churn_within_horizon'] = target

    for feature in MODEL_FEATURE_COLUMNS:
        values = pd.to_numeric(data[feature], errors='coerce')
        if (data[feature].notna() & values.isna()).any():
            raise ValueError(f'{feature} contains invalid numeric values')
        if (values.dropna() < 0).any():
            raise ValueError(f'{feature} contains negative values')
        data[feature] = values
    return data


def chronological_client_disjoint_split(snapshots, cutoff_date):
    """Get time-ordered, label-mature, client-disjoint train/test examples.

    A training observation is eligible only if its ENTIRE future label
    horizon precedes the cutoff. Holdout clients are excluded from training
    entirely, even when an earlier observation belongs to that client.
    Censored observations are never used.
    """
    data = _validate_snapshots(snapshots)
    cutoff = pd.Timestamp(cutoff_date)
    if pd.isna(cutoff):
        raise ValueError('cutoff_date must be valid')

    labeled = data.loc[data['label_status'].isin(['positive', 'negative'])].copy()
    early = labeled.loc[labeled['snapshot_date'] < cutoff]
    train_before_purge = early.loc[early['horizon_end'] < cutoff].copy()
    test = labeled.loc[labeled['snapshot_date'] >= cutoff].copy()
    holdout_clients = set(test['client_id'])
    overlap = train_before_purge['client_id'].isin(holdout_clients)
    train = train_before_purge.loc[~overlap].copy()

    train = train.sort_values(['snapshot_date', 'membership_id']).reset_index(drop=True)
    test = test.sort_values(['snapshot_date', 'membership_id']).reset_index(drop=True)
    details = {
        'cutoff_date': cutoff.date().isoformat(),
        'censored_excluded': int(len(data) - len(labeled)),
        'train_horizon_purged': int(len(early) - len(train_before_purge)),
        'train_client_overlap_purged': int(overlap.sum()),
        'train_rows': len(train),
        'test_rows': len(test),
    }
    return train, test, details


def _unique_outcome_memberships(data, label):
    return int(data.loc[data['churn_within_horizon'].eq(label),
                        'membership_id'].nunique())


def evaluate_training_readiness(snapshots, cutoff_date):
    """Produce a human-readable gate; do NOT fit models here."""
    train, test, details = chronological_client_disjoint_split(snapshots, cutoff_date)
    checks = [
        ('training cancellation memberships',
         _unique_outcome_memberships(train, 1), MIN_TRAIN_POSITIVE_MEMBERSHIPS),
        ('testing cancellation memberships',
         _unique_outcome_memberships(test, 1), MIN_TEST_POSITIVE_MEMBERSHIPS),
        ('training non-cancellation memberships',
         _unique_outcome_memberships(train, 0), MIN_TRAIN_NEGATIVE_MEMBERSHIPS),
        ('testing non-cancellation memberships',
         _unique_outcome_memberships(test, 0), MIN_TEST_NEGATIVE_MEMBERSHIPS),
    ]
    report = pd.DataFrame([{
        'check': name,
        'observed': observed,
        'minimum_screening_requirement': required,
        'passed': bool(observed >= required),
    } for name, observed, required in checks])
    approved = bool(report['passed'].all())
    decision = pd.DataFrame([{**details,
        'training_approved': approved,
        'status': 'experimental_evaluation_permitted' if approved else 'blocked_insufficient_independent_events',
        'note': ('Screening approval is not proof of reliability. Validation metrics must be '
                 'evaluated before considering display of results.'),
    }])
    return report, decision, train, test


def _scores(model_name, truth, predicted, probabilities):
    """Aggregate holdout metrics; no personal identifiers are exposed."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score,
                                 confusion_matrix)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        'model': model_name,
        'accuracy': float(accuracy_score(truth, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
        'precision': float(precision_score(truth, predicted, zero_division=0)),
        'recall': float(recall_score(truth, predicted, zero_division=0)),
        'f1': float(f1_score(truth, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(truth, probabilities)),
        'average_precision': float(average_precision_score(truth, probabilities)),
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
    }


def evaluate_experimental_classifiers(snapshots, cutoff_date):
    """Evaluate baseline and Logistic Regression, or return blocked decision.

    No threshold optimization on the holdout: both use fixed threshold 0.5.
    This never returns client-level predictions or fitted estimators.
    """
    checks, decision, train, test = evaluate_training_readiness(snapshots, cutoff_date)
    metric_columns = ['model', 'accuracy', 'balanced_accuracy', 'precision',
                      'recall', 'f1', 'roc_auc', 'average_precision',
                      'true_positives', 'false_positives', 'true_negatives',
                      'false_negatives']
    if not bool(decision.loc[0, 'training_approved']):
        return {'checks': checks, 'decision': decision,
                'metrics': pd.DataFrame(columns=metric_columns)}

    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = train[MODEL_FEATURE_COLUMNS].astype(float)
    x_test = test[MODEL_FEATURE_COLUMNS].astype(float)
    y_train = train['churn_within_horizon'].astype(int)
    y_test = test['churn_within_horizon'].astype(int)

    models = {
        'majority_class_baseline': DummyClassifier(strategy='prior'),
        'logistic_regression': make_pipeline(
            SimpleImputer(strategy='median', add_indicator=False),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=42)),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        positive_index = list(model.classes_).index(1) if hasattr(model, 'classes_') else 1
        probabilities = model.predict_proba(x_test)[:, positive_index]
        predicted = (probabilities >= 0.5).astype(int)
        rows.append(_scores(name, y_test, predicted, probabilities))
    return {'checks': checks, 'decision': decision,
            'metrics': pd.DataFrame(rows, columns=metric_columns)}


def run_analysis(snapshots_path=SNAPSHOTS_PATH, cutoff_date='2026-09-01',
                 output_dir=OUTPUT_DIR):
    """Write aggregate assessment reports; never export personal predictions."""
    snapshots = pd.read_csv(snapshots_path)
    result = evaluate_experimental_classifiers(snapshots, cutoff_date)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result['checks'].to_csv(out / 'churn_classification_checks.csv', index=False)
    result['decision'].to_csv(out / 'churn_classification_decision.csv', index=False)
    result['metrics'].to_csv(out / 'churn_classification_metrics.csv', index=False)
    print('Experimental classification checks written to:', out)
    print(result['decision'].to_string(index=False))
    if result['metrics'].empty:
        print('Model fitting SKIPPED: insufficient independent cancellation events.')
    else:
        print(result['metrics'].to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate experimental membership churn readiness')
    parser.add_argument('--snapshots', default=str(SNAPSHOTS_PATH))
    parser.add_argument('--cutoff', default='2026-09-01')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_analysis(args.snapshots, args.cutoff, args.output_dir)
