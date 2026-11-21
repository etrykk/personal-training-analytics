#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#CSV Column Mapper

import re
import pandas as pd


#Common names used by businesses for the columns in our platform.
#These suggestions can be expanded as we encounter new CSV formats.
COLUMN_ALIASES = {
    'client_id': ['member id', 'member_id', 'customer id', 'customer_id', 'client number'],
    'join_date': ['signup date', 'sign up date', 'registration date', 'enrollment date'],
    'trainer_id': ['employee id', 'employee_id', 'coach id', 'instructor id'],
    'session_id': ['appointment id', 'booking id', 'training session id'],
    'session_date': ['appointment date', 'booking date', 'training date'],
    'session_status': ['appointment status', 'booking status', 'attendance status'],
    'duration_minutes': ['duration', 'session duration', 'duration mins', 'minutes'],
    'pricing_model': ['billing model', 'pricing type', 'service model'],
    'payment_id': ['transaction id', 'receipt id', 'charge id'],
    'payment_date': ['transaction date', 'date paid', 'paid date'],
    'amount': ['transaction amount', 'payment amount', 'total amount'],
    'payment_status': ['transaction status', 'charge status'],
    'payment_type': ['purchase type', 'charge type', 'revenue type'],
    'transaction_type': ['transaction category', 'payment or refund'],
    'membership_id': ['subscription id', 'contract id'],
    'start_date': ['membership start', 'contract start', 'start date'],
    'end_date': ['membership end', 'contract end', 'end date'],
    'cancellation_date': ['date canceled', 'date cancelled', 'canceled date'],
    'package_id': ['bundle id', 'session package id'],
    'purchase_date': ['date purchased', 'package purchase date'],
    'expiration_date': ['expiry date', 'expires on', 'package expiry'],
    'package_sessions': ['sessions purchased', 'sessions included', 'total sessions'],
    'sessions_redeemed': ['sessions used', 'used sessions'],
    'sessions_remaining': ['remaining sessions', 'sessions left'],
}


def normalize_column_name(column_name):
    name = str(column_name).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


def suggest_column_mapping(data, aliases=None):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')

    aliases = COLUMN_ALIASES if aliases is None else aliases
    if not isinstance(aliases, dict):
        raise TypeError('aliases must be a dictionary')

    lookup = {}
    for target, alternatives in aliases.items():
        for name in [target, *alternatives]:
            normalized = normalize_column_name(name)
            if normalized in lookup and lookup[normalized] != target:
                raise ValueError(f'Ambiguous alias: {name}')
            lookup[normalized] = target

    mapping = {}
    used_targets = {}
    for source in data.columns:
        target = lookup.get(normalize_column_name(source))
        if target is None:
            continue
        if target in used_targets:
            raise ValueError(
                f"Columns '{used_targets[target]}' and '{source}' both match '{target}'. "
                'Provide a manual mapping or remove the duplicate column.'
            )
        mapping[source] = target
        used_targets[target] = source

    return mapping


def map_columns(data, column_mapping=None, required_columns=None, auto_match=True):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if not data.columns.is_unique:
        raise ValueError('Source CSV contains duplicate column names')
    if column_mapping is None:
        column_mapping = {}
    if not isinstance(column_mapping, dict):
        raise TypeError('column_mapping must be a dictionary')

    unknown = set(column_mapping) - set(data.columns)
    if unknown:
        raise ValueError(f'Mapping refers to missing source columns: {sorted(unknown)}')

    mapping = {}
    if auto_match:
        suggestions = suggest_column_mapping(data)
        reserved = set(column_mapping.values())
        mapping = {source: target for source, target in suggestions.items()
                   if source not in column_mapping and target not in reserved}
    mapping.update(column_mapping)

    if any(not isinstance(target, str) or not target.strip() for target in mapping.values()):
        raise ValueError('Every target column name must be a nonempty string')

    renamed_columns = [mapping.get(source, source) for source in data.columns]
    if len(renamed_columns) != len(set(renamed_columns)):
        raise ValueError('Column mapping would create duplicate output column names')

    result = data.rename(columns=mapping).copy()

    if required_columns is not None:
        if isinstance(required_columns, str):
            raise TypeError('required_columns must be a collection, not a string')
        missing = set(required_columns) - set(result.columns)
        if missing:
            raise ValueError(f'Missing required columns after mapping: {sorted(missing)}')

    return result


def mapping_report(original_data, mapped_data):
    if not isinstance(original_data, pd.DataFrame) or not isinstance(mapped_data, pd.DataFrame):
        raise TypeError('Both inputs must be pandas DataFrames')
    if len(original_data.columns) != len(mapped_data.columns):
        raise ValueError('The datasets must have the same number of columns')

    return pd.DataFrame({
        'original_column': list(original_data.columns),
        'mapped_column': list(mapped_data.columns),
        'renamed': [a != b for a, b in zip(original_data.columns, mapped_data.columns)]
    })


if __name__ == '__main__':
    sample = pd.DataFrame({
        'Member ID': [101, 102],
        'Signup Date': ['2026-01-05', '2026-01-06'],
        'Favorite Trainer': ['Alex', 'Jordan']
    })

    mapped = map_columns(
        sample,
        column_mapping={'Favorite Trainer': 'preferred_trainer'},
        required_columns=['client_id', 'join_date']
    )

    print(mapped.head())
    print('\nColumn mapping report:')
    print(mapping_report(sample, mapped).to_string(index=False))
