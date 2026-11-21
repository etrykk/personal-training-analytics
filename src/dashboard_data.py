#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Reusable data import and trainer display helpers

import hashlib
from pathlib import Path

import pandas as pd

from src.data_input.import_pipeline import DATASET_NAMES
from src.data_schema import REQUIRED_COLUMNS


#Fields used by the V1 and V2 business analytics pages, beyond basic validation.
ANALYTICS_COLUMNS = {
    'clients': {'client_id', 'join_date', 'status'},
    'memberships': {'membership_id', 'client_id', 'start_date', 'end_date',
                    'cancellation_date', 'status'},
    'packages': {'package_id', 'client_id', 'package_sessions',
                 'sessions_redeemed', 'sessions_remaining', 'status'},
    'sessions': {'session_id', 'client_id', 'trainer_id', 'session_date',
                 'session_status', 'pricing_model', 'duration_minutes'},
    'payments': {'payment_id', 'client_id', 'payment_date', 'amount',
                 'payment_status', 'transaction_type', 'payment_type', 'session_id'},
}


def prepare_dashboard_data(datasets):
    '''Check the fields needed for reporting, without modifying uploaded data.

    The V1 import pipeline performs column mapping, types and data validation
    first. This second check ensures the analytical reports can also run.
    '''
    if not isinstance(datasets, dict):
        raise TypeError('datasets must be a dictionary')
    missing_files = set(DATASET_NAMES) - set(datasets)
    if missing_files:
        raise ValueError(f'Missing datasets: {sorted(missing_files)}')
    result = {}
    for name in DATASET_NAMES:
        data = datasets[name]
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'{name} must be a pandas DataFrame')
        required = ANALYTICS_COLUMNS[name] | set(REQUIRED_COLUMNS[name])
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f'{name}: missing analytics columns: {sorted(missing)}')
        result[name] = data.copy(deep=True)

    payments = result['payments']
    payments['amount'] = pd.to_numeric(payments['amount'], errors='raise')
    if payments['amount'].isna().any() or (payments['amount'] <= 0).any():
        raise ValueError('Payment amounts must be positive and present')
    if (payments['transaction_type'].eq('refund') &
            ~payments['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions')
    complete = payments['payment_status'].eq('completed')
    payments['net_amount'] = 0.0
    payments.loc[complete & payments['transaction_type'].eq('payment'), 'net_amount'] = payments['amount']
    payments.loc[complete & payments['transaction_type'].eq('refund'), 'net_amount'] = -payments['amount']
    payments['payment_date'] = pd.to_datetime(payments['payment_date'], errors='raise')
    payments['payment_month'] = payments['payment_date'].dt.to_period('M')
    return result


def import_signature(uploads, manual_mappings):
    '''Fingerprint the files and mapping choices for the current import review.'''
    digest = hashlib.sha256()
    for name in DATASET_NAMES:
        blob = uploads.get(name, b'')
        if not isinstance(blob, bytes):
            raise TypeError('uploaded file content must be bytes')
        digest.update(name.encode('utf-8'))
        digest.update(len(blob).to_bytes(8, 'big'))
        digest.update(blob)
        mapping = manual_mappings.get(name, {})
        digest.update(repr(sorted(mapping.items())).encode('utf-8'))
    return digest.hexdigest()


def trainer_name_mapping(sessions, names_file=None, edits=None):
    '''Build default labels and apply optional saved and session-only names.'''
    if not isinstance(sessions, pd.DataFrame) or 'trainer_id' not in sessions.columns:
        raise ValueError('sessions must contain trainer_id')
    ids = sorted(sessions['trainer_id'].dropna().astype(str).unique().tolist())
    mapping = {identifier: f'Trainer {index:02d}'
               for index, identifier in enumerate(ids, start=1)}

    if names_file is not None and Path(names_file).is_file():
        saved = pd.read_csv(names_file, dtype='string')
        if {'trainer_id', 'trainer_name'}.issubset(saved.columns):
            for _, row in saved.dropna(subset=['trainer_id', 'trainer_name']).iterrows():
                identifier = str(row['trainer_id']).strip()
                label = str(row['trainer_name']).strip()
                if identifier in mapping and label:
                    mapping[identifier] = label

    if edits:
        for identifier, label in edits.items():
            identifier = str(identifier).strip()
            label = str(label).strip()
            if identifier in mapping and label:
                mapping[identifier] = label
    return mapping


def labeled_trainers(table, labels):
    '''Add display names while retaining trainer IDs for joins and exports.

    When different trainers share a name, add the ID so chart categories never
    accidentally collapse two trainers into the same group.
    '''
    if 'trainer_id' not in table.columns:
        raise ValueError('table must contain trainer_id')
    result = table.copy(deep=True)
    result['trainer_name'] = result['trainer_id'].astype('string').map(
        lambda value: labels.get(str(value), str(value)) if pd.notna(value) else 'Unknown')
    unique = result[['trainer_id', 'trainer_name']].drop_duplicates()
    shared_names = set(unique.loc[unique['trainer_name'].duplicated(keep=False), 'trainer_name'])
    if shared_names:
        dup = result['trainer_name'].isin(shared_names)
        result.loc[dup, 'trainer_name'] = (
            result.loc[dup, 'trainer_name'].astype(str) + ' (' +
            result.loc[dup, 'trainer_id'].astype(str) + ')')
    return result
