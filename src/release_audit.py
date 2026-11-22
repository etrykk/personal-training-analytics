#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Final release audit (read-only checks)

"""Run a release-readiness audit without changing business data or model outputs.

This checks project structure, Python syntax, dataset headers, dependency declarations,
privacy-sensitive file names, and optional aggregate research reports. It cannot
replace browser testing, the automated test suite, or a security review.
"""

import argparse
import ast
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORE_FILES = (
    'app/dashboard.py', 'app/dashboard_v2.py',
    'src/data_schema.py', 'src/validation.py',
    'src/data_input/import_pipeline.py',
    'src/dashboard_data.py', 'src/dashboard_views.py',
)
ANALYTICS_FILES = (
    'src/analytics/client_engagement.py',
    'src/analytics/client_insights.py',
    'src/analytics/financial_membership_insights.py',
    'src/analytics/session_trainer_insights.py',
    'src/analytics/business_health.py',
)
EXPERIMENT_FILES = (
    'src/ml/churn_readiness.py',
    'src/ml/historical_churn.py',
    'src/ml/churn_classifier.py',
    'src/ml/evaluation_review.py',
    'src/ml/external_churn_experiment.py',
    'src/ml/churn_precision.py',
    'src/ml/predictive_lab_reports.py',
    'src/ml/revenue_forecast_readiness.py',
    'src/ml/revenue_forecast_models.py',
    'src/ml/revenue_forecast_evaluation.py',
    'src/ml/revenue_lab_reports.py',
)
TEST_FILES = (
    'tests/test_dashboard_data.py',
    'tests/test_studio_layout.py',
    'tests/test_predictive_lab_reports.py',
    'tests/test_revenue_lab_reports.py',
)
SCHEMA_HEADERS = {
    'clients': {'client_id', 'join_date', 'status'},
    'memberships': {'membership_id', 'client_id', 'start_date', 'monthly_fee', 'status'},
    'packages': {'package_id', 'client_id', 'purchase_date', 'expiration_date',
                 'package_sessions', 'package_price'},
    'sessions': {'session_id', 'client_id', 'session_date', 'session_status',
                 'pricing_model', 'session_fee'},
    'payments': {'payment_id', 'client_id', 'payment_date', 'amount',
                 'payment_status', 'transaction_type', 'payment_type'},
}
RESEARCH_REPORTS = (
    'outputs/predictive/external_experiment/external_churn_checks.csv',
    'outputs/predictive/external_experiment/external_churn_metrics.csv',
    'outputs/predictive/external_experiment/external_churn_decision.csv',
    'outputs/predictive/revenue_evaluation/forecast_backtest_decision.csv',
    'outputs/predictive/revenue_evaluation/forecast_backtest_checks.csv',
    'outputs/predictive/revenue_evaluation/forecast_validation_scores.csv',
    'outputs/predictive/revenue_evaluation/forecast_holdout_scores.csv',
    'outputs/predictive/revenue_evaluation/forecast_backtest_predictions.csv',
)
DEPENDENCIES = {'pandas', 'plotly', 'streamlit', 'numpy', 'scikit-learn'}
EXCLUDED_DIRS = {'__pycache__', '.git', '.pytest_cache', '.venv', 'venv', 'env',
                 'node_modules', '.tox', 'dist', 'build'}


def _result(area, check, status, details):
    return {'area': area, 'check': check, 'status': status, 'details': details}


def _files_check(root, area, paths):
    missing = [p for p in paths if not (root / p).is_file()]
    return _result(area, 'Required source files', 'FAIL' if missing else 'PASS',
                   'Missing: ' + ', '.join(missing) if missing
                   else f'{len(paths)} required files present')


def _syntax_check(root):
    failures, count = [], 0
    for folder in ('src', 'app', 'tests'):
        base = root / folder
        if not base.exists():
            continue
        for file in base.rglob('*.py'):
            if any(part in EXCLUDED_DIRS for part in file.relative_to(root).parts):
                continue
            count += 1
            try:
                ast.parse(file.read_text(encoding='utf-8-sig'), filename=file.name)
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f'{file.relative_to(root)} ({type(exc).__name__})')
    if not count:
        return _result('Code', 'Python syntax', 'FAIL', 'No Python files found')
    return _result('Code', 'Python syntax', 'FAIL' if failures else 'PASS',
                   'Invalid: ' + ', '.join(failures) if failures else f'{count} files parsed')


def _requirements_check(root):
    path = root / 'requirements.txt'
    if not path.exists():
        return _result('Deployment', 'Dependencies', 'FAIL', 'requirements.txt missing')
    installed = set()
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        stripped = line.strip().split('#', 1)[0].strip()
        if not stripped or stripped.startswith(('-', '.')):
            continue
        name = stripped.split('[', 1)[0].split('=', 1)[0].split('<', 1)[0].split('>', 1)[0]
        installed.add(name.replace('_', '-').lower())
    missing = DEPENDENCIES - installed
    return _result('Deployment', 'Dependencies', 'WARNING' if missing else 'PASS',
                   'Review missing declarations: ' + ', '.join(sorted(missing)) if missing
                   else 'Core application and ML dependencies declared')


def _dataset_check(root, dataset):
    file = root / 'data' / 'processed' / (dataset + '.csv')
    if not file.is_file():
        return _result('Data', dataset + ' dataset', 'WARNING',
                       'Processed sample CSV missing (dashboard may not load sample data)')
    try:
        with file.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            has_row = bool(next(reader, []))
    except (OSError, UnicodeError, csv.Error) as exc:
        return _result('Data', dataset + ' dataset', 'FAIL', type(exc).__name__)
    missing = SCHEMA_HEADERS[dataset] - set(header)
    if missing:
        return _result('Data', dataset + ' dataset', 'FAIL',
                       'Missing headers: ' + ', '.join(sorted(missing)))
    return _result('Data', dataset + ' dataset', 'PASS' if has_row else 'WARNING',
                   'Required headers present' if has_row else 'CSV has no data rows')


def _privacy_check(root):
    # Check names only; never inspect or print credential contents.
    unsafe = []
    for folder in (root, root / '.streamlit'):
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and (path.name == '.env' or path.name.startswith('.env.')
                                   or path.name == 'secrets.toml'
                                   or path.suffix.lower() in {'.pem', '.key'}):
                unsafe.append(str(path.relative_to(root)))
    if unsafe:
        return _result('Privacy', 'Local secrets check', 'WARNING',
                       'Never upload these local files: ' + ', '.join(sorted(unsafe)))
    return _result('Privacy', 'Local secrets check', 'PASS',
                   'No obvious credential filenames in project root or .streamlit')


def _gitignore_check(root):
    path = root / '.gitignore'
    if not path.is_file():
        return _result('Privacy', 'Git ignore rules', 'WARNING', '.gitignore is missing')
    content = path.read_text(encoding='utf-8-sig')
    required = ('__pycache__/', '.env', 'outputs/', 'data/private/',
                'data/uploads/', '.streamlit/secrets.toml')
    missing = [rule for rule in required if rule not in content]
    return _result('Privacy', 'Git ignore rules', 'WARNING' if missing else 'PASS',
                   'Review missing patterns: ' + ', '.join(missing) if missing
                   else 'Sensitive and generated file patterns listed')


def _research_check(root):
    missing = [p for p in RESEARCH_REPORTS if not (root / p).is_file()]
    return _result('Research', 'Aggregate experiment reports',
                   'WARNING' if missing else 'PASS',
                   f'{len(RESEARCH_REPORTS)-len(missing)}/{len(RESEARCH_REPORTS)} local reports present; '
                   'these are optional locally but must be supplied separately for hosted research panels'
                   if missing else f'All {len(RESEARCH_REPORTS)} local aggregate reports present')


def audit_project(root):
    """Return a checklist of release readiness findings; does not modify input files."""
    root = Path(root).resolve()
    rows = [
        _files_check(root, 'Architecture', CORE_FILES),
        _files_check(root, 'Analytics', ANALYTICS_FILES),
        _files_check(root, 'Experimental ML', EXPERIMENT_FILES),
        _files_check(root, 'Tests', TEST_FILES),
        _syntax_check(root),
        _requirements_check(root),
        _privacy_check(root),
        _gitignore_check(root),
    ]
    rows.extend(_dataset_check(root, dataset) for dataset in SCHEMA_HEADERS)
    rows.append(_research_check(root))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='V2 release readiness checks')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output', type=Path, default=None)
    options = parser.parse_args(argv)
    rows = audit_project(options.root)
    output = options.output or (options.root / 'outputs' / 'release' / 'v2_release_audit.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('area', 'check', 'status', 'details'))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['status']:7} | {row['area']:18} | {row['check']}: {row['details']}")
    counts = {key: sum(r['status'] == key for r in rows)
              for key in ('PASS', 'WARNING', 'FAIL')}
    print(f"Release audit: {counts['PASS']} passed, {counts['WARNING']} warning(s), "
          f"{counts['FAIL']} failure(s)")
    print(f'Report: {output}')
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
