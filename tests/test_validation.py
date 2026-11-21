#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Automated Data Validation Tests

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from validation import validate_dataset, validate_all_datasets


@pytest.fixture
def clients():
    return pd.DataFrame({
        'client_id': ['C001', 'C002'],
        'join_date': ['2026-01-01', '2026-01-02'],
        'status': ['active', 'inactive']
    })


@pytest.fixture
def payments():
    return pd.DataFrame({
        'payment_id': ['P001', 'P002'],
        'client_id': ['C001', 'C002'],
        'payment_date': ['2026-01-03', '2026-01-04'],
        'amount': [100.0, 50.0],
        'payment_status': ['completed', 'completed'],
        'transaction_type': ['payment', 'payment'],
        'payment_type': ['session', 'membership']
    })


def has_issue(report, check, column=None):
    matches = report['check'].eq(check)
    if column is not None:
        matches &= report['column'].eq(column)
    return bool(matches.any())


def test_valid_clients_have_no_issues(clients):
    report = validate_dataset(clients, 'clients')
    assert report.empty


def test_valid_datasets_have_no_issues(clients, payments):
    report = validate_all_datasets({'clients': clients, 'payments': payments})
    assert report.empty


def test_missing_required_column(clients):
    data = clients.drop(columns='join_date')
    report = validate_dataset(data, 'clients')
    assert has_issue(report, 'required_columns', 'join_date')


def test_missing_required_value(clients):
    data = clients.copy()
    data.loc[0, 'client_id'] = None
    report = validate_dataset(data, 'clients')
    assert has_issue(report, 'missing_values', 'client_id')


def test_blank_required_value(clients):
    data = clients.copy()
    data.loc[0, 'status'] = '  '
    report = validate_dataset(data, 'clients')
    assert has_issue(report, 'missing_values', 'status')


def test_duplicate_record_id(clients):
    data = clients.copy()
    data.loc[1, 'client_id'] = 'C001'
    report = validate_dataset(data, 'clients')
    assert has_issue(report, 'duplicate_ids', 'client_id')


def test_invalid_date(clients):
    data = clients.copy()
    data.loc[0, 'join_date'] = 'not a date'
    report = validate_dataset(data, 'clients')
    assert has_issue(report, 'invalid_date', 'join_date')


def test_invalid_number(payments):
    data = payments.copy()
    data['amount'] = data['amount'].astype(object)
    data.loc[0, 'amount'] = 'abc'
    report = validate_dataset(data, 'payments')
    assert has_issue(report, 'invalid_number', 'amount')


def test_negative_payment_amount(payments):
    data = payments.copy()
    data.loc[0, 'amount'] = -25
    report = validate_dataset(data, 'payments')
    assert has_issue(report, 'negative_amount', 'amount')


def test_zero_payment_amount(payments):
    data = payments.copy()
    data.loc[0, 'amount'] = 0
    report = validate_dataset(data, 'payments')
    assert has_issue(report, 'zero_payment', 'amount')


def test_invalid_payment_status(payments):
    data = payments.copy()
    data.loc[0, 'payment_status'] = 'unknown'
    report = validate_dataset(data, 'payments')
    assert has_issue(report, 'invalid_category', 'payment_status')


def test_unknown_client_reference(clients, payments):
    data = payments.copy()
    data.loc[1, 'client_id'] = 'C999'
    report = validate_all_datasets({'clients': clients, 'payments': data})
    assert has_issue(report, 'unknown_client', 'client_id')


def test_unknown_session_reference(clients, payments):
    data = payments.copy()
    data['session_id'] = ['S999', None]
    sessions = pd.DataFrame({'session_id': ['S001']})
    report = validate_all_datasets({
        'clients': clients, 'payments': data, 'sessions': sessions
    })
    assert has_issue(report, 'unknown_reference', 'session_id')


def test_unknown_dataset_raises_error(clients):
    with pytest.raises(ValueError, match='Unknown dataset'):
        validate_dataset(clients, 'unknown')


def test_non_dataframe_raises_error():
    with pytest.raises(TypeError, match='DataFrame'):
        validate_dataset([], 'clients')
