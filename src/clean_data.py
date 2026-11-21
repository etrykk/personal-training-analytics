#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Data Import, Cleaning, Validation, and Feature Preparation

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
SAMPLE_DIR = DATA_DIR / 'sample'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DATE = pd.Timestamp('2026-11-20')

SCHEMA = {
    'clients': ['client_id', 'join_date', 'status', 'trainer_id'],
    'memberships': ['membership_id', 'client_id', 'plan_name', 'start_date', 'end_date',
                    'monthly_fee', 'status', 'cancellation_date', 'cancellation_reason'],
    'packages': ['package_id', 'client_id', 'purchase_date', 'expiration_date',
                 'package_sessions', 'package_price', 'sessions_redeemed',
                 'sessions_remaining', 'status'],
    'sessions': ['session_id', 'client_id', 'trainer_id', 'session_date', 'session_time',
                 'session_status', 'duration_minutes', 'pricing_model', 'session_fee', 'package_id'],
    'payments': ['payment_id', 'client_id', 'payment_date', 'amount', 'payment_status',
                 'transaction_type', 'payment_type', 'membership_id', 'session_id',
                 'package_id', 'original_payment_id'],
}

REQUIRED = {
    'clients': ['client_id', 'join_date'],
    'memberships': ['membership_id', 'client_id', 'start_date', 'monthly_fee', 'status'],
    'packages': ['package_id', 'client_id', 'purchase_date', 'expiration_date',
                 'package_sessions', 'package_price'],
    'sessions': ['session_id', 'client_id', 'session_date', 'session_status', 'pricing_model', 'session_fee'],
    'payments': ['payment_id', 'client_id', 'payment_date', 'amount',
                 'payment_status', 'transaction_type', 'payment_type'],
}
DATE_COLUMNS = {
    'clients': ['join_date'],
    'memberships': ['start_date', 'end_date', 'cancellation_date'],
    'packages': ['purchase_date', 'expiration_date'],
    'sessions': ['session_date'],
    'payments': ['payment_date'],
}
NUMERIC_COLUMNS = {
    'clients': [],
    'memberships': ['monthly_fee'],
    'packages': ['package_sessions', 'package_price', 'sessions_redeemed', 'sessions_remaining'],
    'sessions': ['duration_minutes', 'session_fee'],
    'payments': ['amount'],
}
CATEGORIES = {
    'clients': {'status': {'active', 'inactive'}},
    'memberships': {'status': {'active', 'canceled', 'expired'}},
    'packages': {'status': {'active', 'expired', 'depleted'}},
    'sessions': {'session_status': {'completed', 'canceled', 'no_show'},
                 'pricing_model': {'monthly', 'package', 'per_session'}},
    'payments': {'payment_status': {'completed', 'pending', 'failed'},
                 'transaction_type': {'payment', 'refund'},
                 'payment_type': {'membership', 'session', 'package'}},
}
#Add common business export names here. Ambiguous mappings must be resolved manually.
COLUMN_ALIASES = {
    'clients': {'customer_id': 'client_id', 'member_number': 'client_id',
                'signup_date': 'join_date', 'registration_date': 'join_date',
                'coach_id': 'trainer_id'},
    'memberships': {'customer_id': 'client_id', 'member_number': 'client_id',
                    'price': 'monthly_fee', 'membership_start': 'start_date'},
    'packages': {'customer_id': 'client_id', 'purchase_amount': 'package_price'},
    'sessions': {'appointment_id': 'session_id', 'appointment_date': 'session_date',
                 'training_date': 'session_date', 'customer_id': 'client_id',
                 'coach_id': 'trainer_id'},
    'payments': {'transaction_id': 'payment_id', 'customer_id': 'client_id',
                 'total_paid': 'amount', 'transaction_amount': 'amount',
                 'transaction_date': 'payment_date'},
}


def load_data(source_dir=SAMPLE_DIR, allow_missing=False):
    """Read CSV or XLSX exports. Returns the same five DataFrames as V1."""
    source_dir = Path(source_dir)
    datasets = {}
    for name, columns in SCHEMA.items():
        csv_path = source_dir / f'{name}.csv'
        xlsx_path = source_dir / f'{name}.xlsx'
        if csv_path.exists():
            datasets[name] = pd.read_csv(csv_path, dtype='str')
        elif xlsx_path.exists():
            datasets[name] = pd.read_excel(xlsx_path, dtype='str')
        elif allow_missing and name != 'clients':
            datasets[name] = pd.DataFrame(columns=columns)
            print(f'Optional dataset not supplied: {name}')
        else:
            raise FileNotFoundError(f'Missing required file: {csv_path} or {xlsx_path}')
    return tuple(datasets[name] for name in SCHEMA)


def inspect_data(client_data, membership_data, package_data, session_data, payment_data):
    datasets = dict(zip(SCHEMA, (client_data, membership_data, package_data,
                                 session_data, payment_data)))
    for name, data in datasets.items():
        print(f'\n{name.upper()} | Rows: {len(data)} | Columns: {len(data.columns)}')
        print(data.head())
        print('Data types:\n', data.dtypes)
        print('Missing:\n', data.isna().sum())
        print('Duplicate rows:', data.duplicated().sum())
        print(data.describe(include='all'))


def clean_data(client_data, membership_data, package_data, session_data, payment_data):
    datasets = dict(zip(SCHEMA, (client_data, membership_data, package_data,
                                 session_data, payment_data)))
    cleaned = {}
    for name, original in datasets.items():
        data = original.copy()
        data.columns = (pd.Index(data.columns).astype(str).str.strip().str.lower()
                        .str.replace(r'\s+', '_', regex=True))
        data = data.rename(columns=COLUMN_ALIASES[name])
        if data.columns.duplicated().any():
            raise ValueError(f'{name}: duplicate columns after mapping: '
                             f'{data.columns[data.columns.duplicated()].tolist()}')
        missing = set(REQUIRED[name]) - set(data.columns)
        if missing:
            raise ValueError(f'{name}: missing required columns: {sorted(missing)}')
        for column in SCHEMA[name]:
            if column not in data.columns:
                data[column] = pd.NA
        for column in data.columns:
            if pd.api.types.is_string_dtype(data[column]):
                data[column] = data[column].astype('string').str.strip()
                data[column] = data[column].replace('', pd.NA)
        for column in DATE_COLUMNS[name]:
            before = data[column].notna()
            converted = pd.to_datetime(data[column], errors='coerce', format='mixed')
            if (before & converted.isna()).any():
                raise ValueError(f'{name}.{column}: invalid date values')
            data[column] = converted
        for column in NUMERIC_COLUMNS[name]:
            before = data[column].notna()
            converted = pd.to_numeric(data[column], errors='coerce')
            if (before & converted.isna()).any():
                raise ValueError(f'{name}.{column}: invalid numeric values')
            data[column] = converted
        for column in CATEGORIES[name]:
            data[column] = data[column].astype('string').str.lower().str.strip()
        cleaned[name] = data
    return tuple(cleaned[name] for name in SCHEMA)


def validate_cleaned_data(client_data, membership_data, package_data,
                          session_data, payment_data):
    """Fail before exporting incomplete, inconsistent, or misleading records."""
    datasets = dict(zip(SCHEMA, (client_data, membership_data, package_data,
                                 session_data, payment_data)))
    keys = {'clients': 'client_id', 'memberships': 'membership_id',
            'packages': 'package_id', 'sessions': 'session_id', 'payments': 'payment_id'}
    for name, data in datasets.items():
        for column in REQUIRED[name]:
            if data[column].isna().any():
                raise ValueError(f'{name}.{column}: missing required values')
        if data[keys[name]].duplicated().any():
            raise ValueError(f'{name}: duplicate {keys[name]}')
        for column, allowed in CATEGORIES[name].items():
            bad = data[column].dropna().loc[~data[column].dropna().isin(allowed)]
            if not bad.empty:
                raise ValueError(f'{name}.{column}: invalid categories {bad.unique().tolist()}')
        for column in DATE_COLUMNS[name]:
            if (data[column].dropna() > REFERENCE_DATE).any() and not (
                    name == 'packages' and column == 'expiration_date'):
                raise ValueError(f'{name}.{column}: future dates beyond reference date')
    client_ids = set(client_data.client_id)
    for name in ('memberships', 'packages', 'sessions', 'payments'):
        if not set(datasets[name].client_id).issubset(client_ids):
            raise ValueError(f'{name}: unknown client_id')
    if (client_data.join_date.isna()).any():
        raise ValueError('clients: missing join dates')
    for name, column in [('memberships', 'start_date'), ('packages', 'purchase_date'),
                         ('sessions', 'session_date'), ('payments', 'payment_date')]:
        data = datasets[name]
        if data.empty:
            continue
        joined = data[['client_id', column]].merge(client_data[['client_id', 'join_date']],
                                                   on='client_id', validate='many_to_one')
        if (joined[column] < joined.join_date).any():
            raise ValueError(f'{name}: activity before client join date')
    if (membership_data.monthly_fee.dropna() < 0).any():
        raise ValueError('memberships: negative fee')
    if not membership_data.empty:
        valid_end = membership_data.end_date.isna() | (membership_data.end_date >= membership_data.start_date)
        if not valid_end.all():
            raise ValueError('memberships: end before start')
    if not package_data.empty:
        if (package_data.expiration_date < package_data.purchase_date).any():
            raise ValueError('packages: expiration before purchase')
        for col in ('package_sessions', 'sessions_redeemed', 'sessions_remaining'):
            if (package_data[col].dropna() < 0).any():
                raise ValueError(f'packages: negative {col}')
        if (package_data.package_sessions <= 0).any() or (package_data.package_price < 0).any():
            raise ValueError('packages: invalid quantity or price')
        balances = package_data[['package_sessions', 'sessions_redeemed', 'sessions_remaining']]
        complete_balances = balances.notna().all(axis=1)
        if (balances.loc[complete_balances, 'sessions_redeemed'] +
            balances.loc[complete_balances, 'sessions_remaining'] !=
            balances.loc[complete_balances, 'package_sessions']).any():
            raise ValueError('packages: inconsistent credit balances')
    if not session_data.empty:
        if (session_data.session_fee < 0).any():
            raise ValueError('sessions: negative session fee')
        if (session_data.loc[session_data.pricing_model != 'per_session', 'session_fee'] != 0).any():
            raise ValueError('sessions: membership/package sessions cannot be billed again')
        if (session_data.loc[session_data.pricing_model == 'package', 'package_id'].isna()).any():
            raise ValueError('sessions: missing package_id')
        linked = session_data.loc[session_data.package_id.notna()]
        if not linked.empty:
            if not linked.package_id.isin(package_data.package_id).all():
                raise ValueError('sessions: unknown package_id')
            owners = package_data.set_index('package_id').client_id
            if not (linked.package_id.map(owners).to_numpy() == linked.client_id.to_numpy()).all():
                raise ValueError('sessions: package owned by another client')
        if 'duration_minutes' in session_data and (session_data.duration_minutes.dropna() <= 0).any():
            raise ValueError('sessions: invalid duration')
    if not payment_data.empty:
        if (payment_data.amount <= 0).any():
            raise ValueError('payments: amount must be positive')
        for kind, ref_col, source in [
            ('membership', 'membership_id', membership_data),
            ('package', 'package_id', package_data),
            ('session', 'session_id', session_data),
        ]:
            records = payment_data.loc[payment_data.payment_type == kind]
            if not records.empty:
                if records[ref_col].isna().any() or not records[ref_col].isin(source[ref_col]).all():
                    raise ValueError(f'payments: invalid {ref_col}')
                owners = source.set_index(ref_col).client_id
                if not (records[ref_col].map(owners).to_numpy() == records.client_id.to_numpy()).all():
                    raise ValueError(f'payments: wrong owner for {ref_col}')
        originals = payment_data.loc[payment_data.transaction_type == 'payment'].set_index('payment_id')
        refunds = payment_data.loc[payment_data.transaction_type == 'refund']
        if not refunds.empty:
            if refunds.original_payment_id.isna().any() or not refunds.original_payment_id.isin(originals.index).all():
                raise ValueError('payments: invalid refund reference')
            original_ref = originals.loc[refunds.original_payment_id].reset_index(drop=True)
            if not original_ref.payment_status.eq('completed').all():
                raise ValueError('payments: refund references unpaid charge')
            if not (refunds.client_id.to_numpy() == original_ref.client_id.to_numpy()).all():
                raise ValueError('payments: refund belongs to wrong client')
            refund_total = refunds.groupby('original_payment_id').amount.sum()
            if (refund_total.to_numpy() > originals.loc[refund_total.index, 'amount'].to_numpy()).any():
                raise ValueError('payments: refund exceeds original charge')
        session_charges = payment_data.loc[(payment_data.payment_type == 'session') &
                                           (payment_data.transaction_type == 'payment')]
        if session_charges.session_id.duplicated().any():
            raise ValueError('payments: duplicate session charge')
        package_charges = payment_data.loc[(payment_data.payment_type == 'package') &
                                           (payment_data.transaction_type == 'payment')]
        if package_charges.package_id.duplicated().any():
            raise ValueError('payments: duplicate package charge')
        entitled = payment_data.loc[(payment_data.payment_type.isin(['membership', 'package'])) &
                                    (payment_data.transaction_type == 'payment')]
        if not entitled.payment_status.eq('completed').all():
            raise ValueError('payments: unpaid membership or package entitlement')
        if refunds.payment_type.isin(['membership', 'package']).any():
            raise ValueError('payments: refunded entitlement requires access reconciliation')
    print('All cleaned data validation checks passed.')


def prepare_features(client_data, membership_data, package_data, session_data, payment_data):
    client_data, membership_data, package_data, session_data, payment_data = (
        data.copy() for data in (client_data, membership_data, package_data,
                                 session_data, payment_data))
    client_data['join_month'] = client_data.join_date.dt.to_period('M')
    membership_data['membership_duration_days'] = (
        membership_data.end_date.fillna(REFERENCE_DATE) - membership_data.start_date).dt.days
    package_data['package_duration_days'] = (
        package_data.expiration_date - package_data.purchase_date).dt.days
    package_data['redemption_rate'] = (package_data.sessions_redeemed /
                                       package_data.package_sessions)
    session_data['session_month'] = session_data.session_date.dt.to_period('M')
    session_data['session_completed'] = session_data.session_status.eq('completed').astype(int)
    payment_data['payment_month'] = payment_data.payment_date.dt.to_period('M')
    payment_data['net_amount'] = 0.0
    completed = payment_data.payment_status.eq('completed')
    payment_data.loc[completed & payment_data.transaction_type.eq('payment'), 'net_amount'] = (
        payment_data.loc[completed & payment_data.transaction_type.eq('payment'), 'amount'])
    payment_data.loc[completed & payment_data.transaction_type.eq('refund'), 'net_amount'] = (
        -payment_data.loc[completed & payment_data.transaction_type.eq('refund'), 'amount'])
    #Cash collected is not the same as accrual-basis earned revenue or net profit.
    return client_data, membership_data, package_data, session_data, payment_data


def save_data(client_data, membership_data, package_data, session_data, payment_data):
    datasets = dict(zip(SCHEMA, (client_data, membership_data, package_data,
                                 session_data, payment_data)))
    for name, data in datasets.items():
        data.to_csv(PROCESSED_DIR / f'{name}.csv', index=False)
    print('All cleaned datasets saved successfully.')


if __name__ == '__main__':
    #Change source_dir to a separate folder of CSV/XLSX exports for imports.
    data = load_data()
    inspect_data(*data)
    data = clean_data(*data)
    validate_cleaned_data(*data)
    data = prepare_features(*data)
    save_data(*data)
    print(f'Net collected cash: ${data[4].net_amount.sum():,.2f}')
