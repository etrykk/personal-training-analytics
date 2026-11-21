#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#CSV Import Pipeline

import sys
from pathlib import Path
from io import BytesIO

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .csv_loader import load_csv
    from .column_mapper import map_columns, mapping_report
except ImportError:
    from csv_loader import load_csv
    from column_mapper import map_columns, mapping_report

from validation import (REQUIRED_COLUMNS, DATE_COLUMNS, NUMERIC_COLUMNS,
                        ALLOWED_VALUES, REPORT_COLUMNS, validate_all_datasets)

DATASET_NAMES = tuple(REQUIRED_COLUMNS)


def _read_uploaded_csv(upload):
    try:
        if isinstance(upload, bytes):
            data = pd.read_csv(BytesIO(upload))
        else:
            if hasattr(upload, 'seek'):
                upload.seek(0)
            data = pd.read_csv(upload)
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeError, OSError, ValueError) as error:
        raise ValueError(f'Unable to read uploaded CSV: {error}') from error
    if data.empty:
        raise ValueError('Uploaded CSV contains no data rows')
    return data


def _prepare_types(data, dataset_name):
    result = data.copy()
    for column in DATE_COLUMNS.get(dataset_name, []):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors='coerce')
    for column in NUMERIC_COLUMNS.get(dataset_name, []):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors='coerce')
    for column in ALLOWED_VALUES.get(dataset_name, {}):
        if column in result.columns:
            result[column] = result[column].astype('string').str.strip().str.lower()
    return result


def import_datasets(file_sources, manual_mappings=None, auto_match=True):
    '''
    Load, map, validate, and prepare CSV datasets.

    Parameters
    ----------
    file_sources : dict
        Dataset name -> CSV path, uploaded file, or raw CSV bytes.
    manual_mappings : dict, optional
        Dataset name -> {original_column: platform_column}.
    auto_match : bool
        Automatically recognize common alternative column names.

    Returns
    -------
    dict with:
        'datasets': prepared DataFrames if there are no validation errors;
                    otherwise an empty dictionary
        'mapped_datasets': mapped DataFrames for review (not approved for analytics)
        'mapping_reports': dictionary of column mapping reports
        'validation_report': DataFrame of validation issues
        'valid': True only when there are no errors

    The function does not write files or modify existing project data.
    '''
    if not isinstance(file_sources, dict) or not file_sources:
        raise ValueError('file_sources must be a nonempty dictionary')
    if manual_mappings is None:
        manual_mappings = {}
    if not isinstance(manual_mappings, dict):
        raise TypeError('manual_mappings must be a dictionary')

    unknown = set(file_sources) - set(DATASET_NAMES)
    if unknown:
        raise ValueError(f'Unknown dataset names: {sorted(unknown)}')
    unknown_mappings = set(manual_mappings) - set(file_sources)
    if unknown_mappings:
        raise ValueError(f'Mappings provided for datasets not uploaded: {sorted(unknown_mappings)}')

    mapped_datasets = {}
    mapping_reports = {}

    for name, source in file_sources.items():
        if isinstance(source, (str, Path)):
            original = load_csv(source)
        else:
            original = _read_uploaded_csv(source)

        mapped = map_columns(original,
                             column_mapping=manual_mappings.get(name),
                             auto_match=auto_match)
        mapped_datasets[name] = mapped
        mapping_reports[name] = mapping_report(original, mapped)

    report = validate_all_datasets(mapped_datasets)
    has_errors = bool((report['severity'] == 'error').any())

    prepared = ({name: _prepare_types(data, name)
                 for name, data in mapped_datasets.items()}
                if not has_errors else {})

    return {
        'datasets': prepared,
        'mapped_datasets': mapped_datasets,
        'mapping_reports': mapping_reports,
        'validation_report': report,
        'valid': not has_errors,
    }


if __name__ == '__main__':
    sources = {name: f'data/sample/{name}.csv' for name in DATASET_NAMES}
    result = import_datasets(sources)
    print(f"Import valid: {result['valid']}")
    print(f"Validation issues: {len(result['validation_report'])}")
    if not result['validation_report'].empty:
        print(result['validation_report'].to_string(index=False))
    if result['valid']:
        for name, data in result['datasets'].items():
            print(f'{name.title()}: {len(data)} rows, {len(data.columns)} columns')
