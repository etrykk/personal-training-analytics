#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Imported Data Validation

import pandas as pd

#Columns required for the existing analytics pipeline
REQUIRED_COLUMNS = {
    'clients': ['client_id', 'join_date', 'status'],
    'memberships': ['membership_id', 'client_id', 'start_date', 'monthly_fee', 'status'],
    'packages': ['package_id', 'client_id', 'purchase_date', 'expiration_date',
                 'package_sessions', 'package_price'],
    'sessions': ['session_id', 'client_id', 'session_date', 'session_status',
                 'pricing_model', 'session_fee'],
    'payments': ['payment_id', 'client_id', 'payment_date', 'amount',
                 'payment_status', 'transaction_type', 'payment_type']
}

ID_COLUMNS = {'clients': 'client_id', 'memberships': 'membership_id',
              'packages': 'package_id', 'sessions': 'session_id',
              'payments': 'payment_id'}

DATE_COLUMNS = {
    'clients': ['join_date'],
    'memberships': ['start_date', 'end_date', 'cancellation_date'],
    'packages': ['purchase_date', 'expiration_date'],
    'sessions': ['session_date'],
    'payments': ['payment_date']
}

NUMERIC_COLUMNS = {
    'memberships': ['monthly_fee'],
    'packages': ['package_sessions', 'package_price', 'sessions_redeemed', 'sessions_remaining'],
    'sessions': ['duration_minutes', 'session_fee'],
    'payments': ['amount']
}

ALLOWED_VALUES = {
    'clients': {'status': {'active', 'inactive'}},
    'memberships': {'status': {'active', 'canceled', 'expired'}},
    'packages': {'status': {'active', 'expired', 'depleted'}},
    'sessions': {'session_status': {'completed', 'canceled', 'no_show'},
                 'pricing_model': {'monthly', 'package', 'per_session'}},
    'payments': {'payment_status': {'completed', 'pending', 'failed'},
                 'transaction_type': {'payment', 'refund'},
                 'payment_type': {'membership', 'session', 'package'}}
}

REPORT_COLUMNS = ['dataset', 'severity', 'check', 'column', 'affected_rows', 'message']


def _issue(dataset, severity, check, column, rows, message):
    return {'dataset': dataset, 'severity': severity, 'check': check,
            'column': column, 'affected_rows': int(rows), 'message': message}


def validate_required_columns(data, dataset_name):
    #Check for missing columns needed by the analytics platform
    required = REQUIRED_COLUMNS[dataset_name]
    return [_issue(dataset_name, 'error', 'required_columns', column, len(data),
                   f'Missing required column: {column}')
            for column in required if column not in data.columns]


def validate_missing_values(data, dataset_name):
    #Find blank values in required columns
    issues = []
    for column in REQUIRED_COLUMNS[dataset_name]:
        if column not in data.columns:
            continue
        blank = data[column].isna()
        if pd.api.types.is_string_dtype(data[column]) or data[column].dtype == object:
            blank = blank | data[column].astype('string').str.strip().eq('').fillna(False)
        count = int(blank.sum())
        if count:
            issues.append(_issue(dataset_name, 'error', 'missing_values', column,
                                 count, f'{count} row(s) have missing {column}'))
    return issues


def validate_duplicates(data, dataset_name):
    #Record identifiers must be unique within each dataset
    column = ID_COLUMNS[dataset_name]
    if column not in data.columns:
        return []
    ids = data[column].astype('string').str.strip()
    duplicates = ids.notna() & ids.duplicated(keep=False)
    count = int(duplicates.sum())
    return ([_issue(dataset_name, 'error', 'duplicate_ids', column, count,
                    f'{count} row(s) share a duplicate {column}')]
            if count else [])


def validate_data_types(data, dataset_name):
    #Check whether dates and numeric values can be converted
    issues = []
    for column in DATE_COLUMNS.get(dataset_name, []):
        if column in data.columns:
            values = data[column].replace(r'^\s*$', pd.NA, regex=True)
            converted = pd.to_datetime(values, errors='coerce')
            count = int((values.notna() & converted.isna()).sum())
            if count:
                issues.append(_issue(dataset_name, 'error', 'invalid_date', column,
                                     count, f'{count} invalid date value(s) in {column}'))
    for column in NUMERIC_COLUMNS.get(dataset_name, []):
        if column in data.columns:
            values = data[column].replace(r'^\s*$', pd.NA, regex=True)
            converted = pd.to_numeric(values, errors='coerce')
            count = int((values.notna() & converted.isna()).sum())
            if count:
                issues.append(_issue(dataset_name, 'error', 'invalid_number', column,
                                     count, f'{count} nonnumeric value(s) in {column}'))
            #Negative monetary amounts are not permitted; refunds use transaction_type
            valid = converted.dropna()
            if column in {'amount', 'monthly_fee', 'package_price', 'session_fee'}:
                negative = int((valid < 0).sum())
                if negative:
                    issues.append(_issue(dataset_name, 'error', 'negative_amount', column,
                                         negative, f'{negative} negative value(s) in {column}'))
            if column == 'amount':
                zero = int((valid == 0).sum())
                if zero:
                    issues.append(_issue(dataset_name, 'error', 'zero_payment', column,
                                         zero, 'Payment amounts must be greater than zero'))
            if column in {'package_sessions', 'sessions_redeemed', 'sessions_remaining'}:
                bad = int(((valid < 0) | (valid % 1 != 0)).sum())
                if bad:
                    issues.append(_issue(dataset_name, 'error', 'invalid_session_count', column,
                                         bad, 'Session counts must be nonnegative whole numbers'))
            if column == 'duration_minutes':
                bad = int((valid <= 0).sum())
                if bad:
                    issues.append(_issue(dataset_name, 'error', 'invalid_duration', column,
                                         bad, 'Session duration must be greater than zero'))
    return issues


def validate_allowed_values(data, dataset_name):
    #Compare category values to the statuses used by the analytics functions
    issues = []
    for column, allowed in ALLOWED_VALUES.get(dataset_name, {}).items():
        if column not in data.columns:
            continue
        values = data[column].astype('string').str.strip().str.lower()
        invalid = values.notna() & values.ne('') & ~values.isin(allowed)
        count = int(invalid.sum())
        if count:
            issues.append(_issue(dataset_name, 'error', 'invalid_category', column,
                                 count, f'Unexpected {column}; expected one of {sorted(allowed)}'))
    return issues


def validate_relationships(datasets):
    #Check references across the supplied datasets, when both datasets are available
    issues = []
    clients = datasets.get('clients')
    if clients is not None and 'client_id' in clients.columns:
        known = set(clients['client_id'].dropna().astype('string').str.strip())
        for name in ('memberships', 'packages', 'sessions', 'payments'):
            data = datasets.get(name)
            if data is None or 'client_id' not in data.columns:
                continue
            refs = data['client_id'].astype('string').str.strip()
            count = int((refs.notna() & refs.ne('') & ~refs.isin(known)).sum())
            if count:
                issues.append(_issue(name, 'error', 'unknown_client', 'client_id',
                                     count, f'{count} client ID(s) not found in clients'))

    #Check optional foreign keys against the matching source dataset
    for name, column, source_name, source_column in [
        ('sessions', 'package_id', 'packages', 'package_id'),
        ('payments', 'membership_id', 'memberships', 'membership_id'),
        ('payments', 'package_id', 'packages', 'package_id'),
        ('payments', 'session_id', 'sessions', 'session_id'),
        ('payments', 'original_payment_id', 'payments', 'payment_id')
    ]:
        data, source = datasets.get(name), datasets.get(source_name)
        if data is None or source is None or column not in data or source_column not in source:
            continue
        known = set(source[source_column].dropna().astype('string').str.strip())
        refs = data[column].astype('string').str.strip()
        count = int((refs.notna() & refs.ne('') & ~refs.isin(known)).sum())
        if count:
            issues.append(_issue(name, 'error', 'unknown_reference', column,
                                 count, f'{count} reference(s) not found in {source_name}'))
    return issues


def validate_dataset(data, dataset_name):
    #Run all dataset-level checks and return a readable issue report
    if dataset_name not in REQUIRED_COLUMNS:
        raise ValueError(f'Unknown dataset: {dataset_name}. Expected {list(REQUIRED_COLUMNS)}')
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')

    issues = []
    if not data.columns.is_unique:
        issues.append(_issue(dataset_name, 'error', 'duplicate_columns', '', 0,
                             'The CSV contains duplicate column names'))
    issues.extend(validate_required_columns(data, dataset_name))
    issues.extend(validate_missing_values(data, dataset_name))
    issues.extend(validate_duplicates(data, dataset_name))
    issues.extend(validate_data_types(data, dataset_name))
    issues.extend(validate_allowed_values(data, dataset_name))
    if data.empty:
        issues.append(_issue(dataset_name, 'warning', 'empty_dataset', '', 0,
                             'No records were supplied'))
    return pd.DataFrame(issues, columns=REPORT_COLUMNS)


def validate_all_datasets(datasets):
    #Combine dataset checks with cross-dataset relationship checks
    if not isinstance(datasets, dict):
        raise TypeError('datasets must be a dictionary')
    reports = [validate_dataset(data, name) for name, data in datasets.items()]
    reports.append(pd.DataFrame(validate_relationships(datasets), columns=REPORT_COLUMNS))
    return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame(columns=REPORT_COLUMNS)


if __name__ == '__main__':
    #Test with a small dataset containing an invalid status and unknown client
    sample_clients = pd.DataFrame({
        'client_id': ['101', '102'],
        'join_date': ['2026-01-05', '2026-01-06'],
        'status': ['active', 'inactive']
    })
    sample_payments = pd.DataFrame({
        'payment_id': ['P1', 'P2'], 'client_id': ['101', '999'],
        'payment_date': ['2026-01-07', 'invalid date'],
        'amount': [150, -50], 'payment_status': ['completed', 'unknown'],
        'transaction_type': ['payment', 'payment'],
        'payment_type': ['session', 'session']
    })
    report = validate_all_datasets({'clients': sample_clients, 'payments': sample_payments})
    print(report.to_string(index=False))
    print(f'\nValidation complete: {len(report)} issue(s) found')
