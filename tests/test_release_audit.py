#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Release audit tests

import csv
from pathlib import Path

from src.release_audit import (
    CORE_FILES, ANALYTICS_FILES, EXPERIMENT_FILES, TEST_FILES,
    SCHEMA_HEADERS, RESEARCH_REPORTS, audit_project, main,
)


def make_project(root):
    for name in (*CORE_FILES, *ANALYTICS_FILES, *EXPERIMENT_FILES, *TEST_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('pass\n', encoding='utf-8')
    for name, required in SCHEMA_HEADERS.items():
        path = root / 'data' / 'processed' / (name + '.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(','.join(sorted(required)) + '\n' +
                        ','.join('x' for _ in required) + '\n', encoding='utf-8')
    (root / 'requirements.txt').write_text('pandas\nstreamlit\nplotly\nnumpy\nscikit-learn\n')
    (root / '.gitignore').write_text('\n'.join([
        '__pycache__/', '.env', 'outputs/', 'data/private/',
        'data/uploads/', '.streamlit/secrets.toml']))
    for report in RESEARCH_REPORTS:
        path = root / report
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('report\n', encoding='utf-8')


def row_for(rows, name):
    return next(row for row in rows if row['check'] == name)


def test_complete_fixture_passes(tmp_path):
    make_project(tmp_path)
    rows = audit_project(tmp_path)
    assert all(row['status'] == 'PASS' for row in rows)


def test_missing_core_file_fails(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'app/dashboard_v2.py').unlink()
    assert row_for(audit_project(tmp_path), 'Required source files')['status'] == 'FAIL'


def test_missing_ml_file_fails(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'src/ml/churn_classifier.py').unlink()
    rows = audit_project(tmp_path)
    assert any(row['area'] == 'Experimental ML' and row['status'] == 'FAIL' for row in rows)


def test_python_syntax_failure_detected(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'src/dashboard_views.py').write_text('def bad(:\n')
    assert row_for(audit_project(tmp_path), 'Python syntax')['status'] == 'FAIL'


def test_missing_dependency_is_warning(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'requirements.txt').write_text('pandas\nstreamlit\nplotly\n')
    assert row_for(audit_project(tmp_path), 'Dependencies')['status'] == 'WARNING'


def test_missing_processed_csv_is_warning(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'data/processed/payments.csv').unlink()
    assert row_for(audit_project(tmp_path), 'payments dataset')['status'] == 'WARNING'


def test_invalid_processed_header_fails(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'data/processed/clients.csv').write_text('client_id\nC1\n')
    assert row_for(audit_project(tmp_path), 'clients dataset')['status'] == 'FAIL'


def test_empty_processed_csv_warns(tmp_path):
    make_project(tmp_path)
    file = tmp_path / 'data/processed/clients.csv'
    with file.open(encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream))
    file.write_text(','.join(header) + '\n')
    assert row_for(audit_project(tmp_path), 'clients dataset')['status'] == 'WARNING'


def test_private_secrets_warn_without_reading_contents(tmp_path):
    make_project(tmp_path)
    secret = tmp_path / '.env'
    secret.write_text('SENSITIVE_FAKE_SECRET_FOR_TEST')
    status = row_for(audit_project(tmp_path), 'Local secrets check')
    assert status['status'] == 'WARNING'
    assert 'SENSITIVE_FAKE_SECRET_FOR_TEST' not in status['details']


def test_missing_ignore_rules_warn(tmp_path):
    make_project(tmp_path)
    (tmp_path / '.gitignore').unlink()
    assert row_for(audit_project(tmp_path), 'Git ignore rules')['status'] == 'WARNING'


def test_missing_optional_reports_warn(tmp_path):
    make_project(tmp_path)
    (tmp_path / RESEARCH_REPORTS[0]).unlink()
    assert row_for(audit_project(tmp_path), 'Aggregate experiment reports')['status'] == 'WARNING'


def test_cli_writes_report(tmp_path):
    make_project(tmp_path)
    path = tmp_path / 'audit.csv'
    assert main(['--root', str(tmp_path), '--output', str(path)]) == 0
    with path.open(newline='', encoding='utf-8') as stream:
        assert all('status' in row for row in csv.DictReader(stream))


def test_cli_returns_failure_status(tmp_path):
    make_project(tmp_path)
    (tmp_path / 'app/dashboard_v2.py').unlink()
    assert main(['--root', str(tmp_path), '--output', str(tmp_path / 'audit.csv')]) == 1
