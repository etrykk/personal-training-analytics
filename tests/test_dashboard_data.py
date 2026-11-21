#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Imported data and trainer-name tests

from pathlib import Path

import pandas as pd
import pytest

from src.dashboard_data import (prepare_dashboard_data, import_signature,
                                trainer_name_mapping, labeled_trainers)
from src.data_input.import_pipeline import import_datasets, DATASET_NAMES
from src.analytics.business_health import run_business_health

SAMPLE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'sample'


def example_import():
    return {name: (SAMPLE_DIR / f'{name}.csv').read_bytes() for name in DATASET_NAMES}


def example_prepared():
    result = import_datasets(example_import())
    assert result['valid'], result['validation_report'].to_string(index=False)
    return result, prepare_dashboard_data(result['datasets'])


def test_v1_csv_import_pipeline_works_with_all_five_datasets():
    result, prepared = example_prepared()
    assert result['validation_report'].empty
    assert set(prepared) == set(DATASET_NAMES)
    assert len(prepared['clients']) == 100


def test_imported_data_runs_v2_business_analytics():
    _, data = example_prepared()
    overview = run_business_health(data['clients'], data['memberships'],
                                    data['sessions'], data['payments'], '2026-11-20')
    assert len(overview['business_overview']) == 1
    assert overview['business_overview'].iloc[0]['total_clients'] == 100


def test_import_validation_rejects_missing_required_fields():
    uploaded = example_import()
    clients = pd.read_csv(SAMPLE_DIR / 'clients.csv').drop(columns='join_date')
    uploaded['clients'] = clients.to_csv(index=False).encode('utf-8')
    result = import_datasets(uploaded)
    assert not result['valid']
    assert result['datasets'] == {}
    assert (result['validation_report']['severity'] == 'error').any()


def test_prepare_dashboard_data_rejects_missing_dataset():
    _, data = example_prepared()
    data.pop('payments')
    with pytest.raises(ValueError, match='Missing datasets'):
        prepare_dashboard_data(data)


def test_prepare_dashboard_data_checks_columns_needed_for_analytics():
    _, data = example_prepared()
    data['sessions'] = data['sessions'].drop(columns='trainer_id')
    with pytest.raises(ValueError, match='trainer_id'):
        prepare_dashboard_data(data)


def test_prepare_dashboard_data_does_not_change_original_frames():
    result = import_datasets(example_import())
    original = result['datasets']['payments'].copy(deep=True)
    prepared = prepare_dashboard_data(result['datasets'])
    pd.testing.assert_frame_equal(result['datasets']['payments'], original)
    assert 'net_amount' in prepared['payments']


def test_prepare_dashboard_data_uses_refunds_as_negative_cash():
    _, data = example_prepared()
    payments = data['payments']
    completed_refunds = payments.loc[(payments['transaction_type'] == 'refund') &
                                     (payments['payment_status'] == 'completed')]
    assert not completed_refunds.empty
    assert (completed_refunds['net_amount'] < 0).all()


def test_import_signature_detects_change_to_file():
    uploads = example_import()
    first = import_signature(uploads, {})
    uploads['clients'] += b'\n'
    assert import_signature(uploads, {}) != first


def test_import_signature_detects_mapping_changes_and_is_order_independent():
    uploads = example_import()
    x = import_signature(uploads, {'clients': {'a': 'client_id', 'b': 'join_date'}})
    y = import_signature(uploads, {'clients': {'b': 'join_date', 'a': 'client_id'}})
    z = import_signature(uploads, {'clients': {'a': 'status', 'b': 'join_date'}})
    assert x == y
    assert x != z


def test_import_signature_rejects_nonbyte_inputs():
    with pytest.raises(TypeError):
        import_signature({'clients': 'not bytes'}, {})


def test_trainer_names_default_and_edited():
    sessions = pd.DataFrame({'trainer_id': ['T002', 'T001', 'T001']})
    names = trainer_name_mapping(sessions, edits={'T002': 'Alex', 'not_present': 'Ignore'})
    assert names == {'T001': 'Trainer 01', 'T002': 'Alex'}


def test_trainer_names_loaded_from_optional_csv(tmp_path):
    path = tmp_path / 'trainer_names.csv'
    path.write_text('trainer_id,trainer_name\nT001,Jordan\nT002,Sam\n')
    sessions = pd.DataFrame({'trainer_id': ['T001', 'T002']})
    names = trainer_name_mapping(sessions, names_file=path, edits={'T002': 'Pat'})
    assert names == {'T001': 'Jordan', 'T002': 'Pat'}


def test_trainer_names_ignore_empty_edit_and_unmatched_id():
    sessions = pd.DataFrame({'trainer_id': ['T001']})
    assert trainer_name_mapping(sessions, edits={'T001': ' ', 'T099': 'Other'}) == {'T001': 'Trainer 01'}


def test_labeled_trainers_preserves_source_and_original_ids():
    frame = pd.DataFrame({'trainer_id': ['T001', 'T002'], 'recent_completed': [3, 4]})
    original = frame.copy(deep=True)
    result = labeled_trainers(frame, {'T001': 'Taylor', 'T002': 'Jordan'})
    pd.testing.assert_frame_equal(frame, original)
    assert result['trainer_name'].tolist() == ['Taylor', 'Jordan']
    assert result['trainer_id'].tolist() == ['T001', 'T002']


def test_duplicate_trainer_names_are_disambiguated_but_repeated_ids_are_not():
    frame = pd.DataFrame({'trainer_id': ['T001', 'T001', 'T002'], 'month': [1, 2, 1]})
    labels = {'T001': 'Chris', 'T002': 'Chris'}
    assert labeled_trainers(frame, labels)['trainer_name'].tolist() == [
        'Chris (T001)', 'Chris (T001)', 'Chris (T002)'
    ]
    assert labeled_trainers(frame, {'T001': 'Chris', 'T002': 'Pat'})['trainer_name'].tolist() == [
        'Chris', 'Chris', 'Pat'
    ]


def test_trainer_table_requires_trainer_ids():
    with pytest.raises(ValueError):
        labeled_trainers(pd.DataFrame({'x': [1]}), {})
