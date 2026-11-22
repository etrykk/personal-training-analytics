#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 6B - Date Parsing Regression Tests

import sys
from pathlib import Path
import warnings

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from validation import validate_dataset


def _clients(dates):
    return pd.DataFrame({
        'client_id': [f'C{i:03d}' for i in range(len(dates))],
        'join_date': dates,
        'status': ['active'] * len(dates),
    })


def test_invalid_date_still_generates_validation_error():
    report = validate_dataset(_clients(['not a date', '2026-01-02']), 'clients')
    rows = report.loc[report['check'].eq('invalid_date')]
    assert len(rows) == 1
    assert rows.iloc[0]['affected_rows'] == 1


def test_invalid_date_does_not_emit_date_inference_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        validate_dataset(_clients(['not a date', '2026-01-02']), 'clients')
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_mixed_valid_import_dates_are_accepted():
    report = validate_dataset(_clients(['2026-01-02', '01/03/2026']), 'clients')
    assert not report['check'].eq('invalid_date').any()


def test_two_bad_dates_are_counted():
    report = validate_dataset(_clients(['invalid', 'another bad date', '2026-01-02']), 'clients')
    rows = report.loc[report['check'].eq('invalid_date')]
    assert len(rows) == 1
    assert rows.iloc[0]['affected_rows'] == 2


def test_input_dataframe_is_not_changed():
    clients = _clients(['2026-01-02', 'not a date'])
    original = clients.copy(deep=True)
    validate_dataset(clients, 'clients')
    pd.testing.assert_frame_equal(clients, original)
