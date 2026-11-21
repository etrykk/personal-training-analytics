#Author: etrykk
#Personal Training Analytics Platform - V2
#Tests for Dataset Analytics Readiness

import pandas as pd

from src.analytics.data_readiness import analyze_data_readiness
from src.data_schema import REQUIRED_COLUMNS


def make_valid_datasets():
    return {
        name: pd.DataFrame([{column: 'sample' for column in columns}])
        for name, columns in REQUIRED_COLUMNS.items()
    }


def get_result(report, dataset_name):
    return report.set_index('dataset').loc[dataset_name]


def test_all_datasets_ready():
    report = analyze_data_readiness(make_valid_datasets())

    assert len(report) == len(REQUIRED_COLUMNS)
    assert report['ready'].all()
    assert report['present'].all()
    assert (report['row_count'] == 1).all()
    assert report['missing_columns'].apply(lambda missing: len(missing) == 0).all()


def test_missing_dataset():
    datasets = make_valid_datasets()
    del datasets['payments']

    result = get_result(analyze_data_readiness(datasets), 'payments')

    assert not result['present']
    assert not result['ready']
    assert result['row_count'] == 0
    assert result['missing_columns'] == REQUIRED_COLUMNS['payments']


def test_missing_required_column():
    datasets = make_valid_datasets()
    datasets['clients'] = datasets['clients'].drop(columns=['status'])

    result = get_result(analyze_data_readiness(datasets), 'clients')

    assert result['present']
    assert not result['ready']
    assert result['row_count'] == 1
    assert result['missing_columns'] == ['status']


def test_empty_dataset():
    datasets = make_valid_datasets()
    datasets['sessions'] = datasets['sessions'].iloc[0:0]

    result = get_result(analyze_data_readiness(datasets), 'sessions')

    assert result['present']
    assert not result['ready']
    assert result['row_count'] == 0
    assert result['missing_columns'] == []
