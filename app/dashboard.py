#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Phase 7: Dashboard with importing, analytics, and business reports

import sys
from pathlib import Path
from io import BytesIO
import hashlib
import tempfile
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_input.column_mapper import suggest_column_mapping
from data_input.import_pipeline import import_datasets, DATASET_NAMES
from validation import REQUIRED_COLUMNS
from revenue_analysis import (total_revenue, monthly_revenue, revenue_by_type,
                              revenue_growth, revenue_per_client, refund_analysis)
from client_analysis import (client_acquisition, client_status_summary,
                             membership_cancellations, membership_cohort_retention,
                             pay_per_session_inactivity)
from session_analysis import (attendance_summary, monthly_attendance,
                              activity_by_pricing_model, package_redemption)
from trainer_analysis import (trainer_activity, trainer_monthly_activity,
                              trainer_direct_session_cash)
from reporting import build_reports, export_reports
from visualizations import format_chart

st.set_page_config(page_title='Personal Training Analytics', page_icon='📊', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container {max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.035em; font-weight: 750;}
h2, h3 {letter-spacing: -0.02em;}
[data-testid="stMetric"] {padding: 1.05rem 1.15rem; border: 1px solid rgba(128,128,128,.20);
    border-radius: 14px; background: rgba(128,128,128,.045);}
[data-testid="stMetricLabel"] {font-size: .92rem;}
[data-testid="stMetricValue"] {font-weight: 700; letter-spacing: -.03em;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
[data-testid="stVerticalBlock"] > div {min-width: 0;}
</style>''', unsafe_allow_html=True)

LABELS = {'per_session': 'Pay Per Session', 'no_show': 'No-Show',
          'completed': 'Completed', 'canceled': 'Canceled',
          'membership': 'Membership', 'package': 'Package',
          'session': 'Individual Session', 'monthly': 'Monthly',
          'active': 'Active', 'inactive': 'Inactive', 'expired': 'Expired'}

def pretty(value):
    return LABELS.get(str(value), str(value).replace('_', ' ').title())

def money(value):
    return f'${float(value):,.2f}'

def short_money(value):
    sign = '-' if value < 0 else ''
    amount = abs(float(value))
    if amount >= 1_000_000:
        return f'{sign}${amount / 1_000_000:,.1f}M'
    if amount >= 1_000:
        return f'{sign}${amount / 1_000:,.1f}k'
    return f'{sign}${amount:,.0f}'

def chart(fig, currency=False, height=385):
    fig = format_chart(fig, currency=currency, height=height)
    st.plotly_chart(fig, use_container_width=True)

def cash_chart(data, x, y, title=None, kind='line', color=None):
    plot = data.copy()
    plot['Cash Label'] = plot[y].map(short_money)
    plot['Cash Detail'] = plot[y].map(money)
    if kind == 'bar':
        fig = px.bar(plot, x=x, y=y, color=color, text='Cash Label', title=title,
                     custom_data=['Cash Label', 'Cash Detail'])
        fig.update_traces(textposition='outside', cliponaxis=False)
    else:
        fig = px.line(plot, x=x, y=y, color=color, markers=True, title=title,
                      custom_data=['Cash Label', 'Cash Detail'])
    fig.update_traces(hovertemplate='%{x}<br>Cash Collected = %{customdata[0]}<br>Exact: %{customdata[1]}<extra></extra>')
    fig.update_layout(yaxis_title='Cash Collected ($)', xaxis_title=None)
    chart(fig, currency=True)

def table(data, filename=None, labels=None):
    view = data.copy()
    for col in ['payment_type', 'pricing_model', 'session_status', 'status', 'transaction_type']:
        if col in view.columns:
            view[col] = view[col].map(lambda v: pretty(v) if pd.notna(v) else v)
    if 'trainer_id' in view.columns:
        view['trainer_id'] = view['trainer_id'].map(lambda v: labels.get(str(v), str(v)) if labels else v)
    for col in view.columns:
        if pd.api.types.is_period_dtype(view[col].dtype):
            view[col] = view[col].astype(str)
    view.columns = [pretty(c) for c in view.columns]
    st.dataframe(view, use_container_width=True, hide_index=True)
    if filename:
        st.download_button('Download CSV', data=view.to_csv(index=False).encode('utf-8'),
                           file_name=filename, mime='text/csv', key=f'download_{filename}')

@st.cache_data(show_spinner='Loading business data...')
def load_data():
    clients = pd.read_csv(PROCESSED_DIR / 'clients.csv', parse_dates=['join_date'])
    memberships = pd.read_csv(PROCESSED_DIR / 'memberships.csv',
                              parse_dates=['start_date', 'end_date', 'cancellation_date'])
    sessions = pd.read_csv(PROCESSED_DIR / 'sessions.csv', parse_dates=['session_date'])
    packages = pd.read_csv(PROCESSED_DIR / 'packages.csv',
                           parse_dates=['purchase_date', 'expiration_date'])
    payments = pd.read_csv(PROCESSED_DIR / 'payments.csv', parse_dates=['payment_date'])
    required = {'amount', 'payment_status', 'transaction_type', 'payment_type',
                'payment_id', 'client_id', 'payment_date'}
    missing = required - set(payments.columns)
    if missing:
        raise ValueError(f'Missing payment columns: {sorted(missing)}')
    payments['amount'] = pd.to_numeric(payments['amount'], errors='raise')
    if payments['amount'].isna().any() or (payments['amount'] <= 0).any():
        raise ValueError('Payment amounts must be positive and nonmissing.')
    if payments['payment_id'].isna().any() or payments['payment_id'].duplicated().any():
        raise ValueError('Payment IDs must be present and unique.')
    for col, valid in {'payment_status': {'completed', 'pending', 'failed'},
                       'transaction_type': {'payment', 'refund'},
                       'payment_type': {'membership', 'package', 'session'}}.items():
        if not payments[col].isin(valid).all():
            raise ValueError(f'Unexpected values in {col}.')
    if (payments['transaction_type'].eq('refund') & ~payments['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions.')
    completed = payments['payment_status'].eq('completed')
    payments['net_amount'] = 0.0
    payments.loc[completed & payments['transaction_type'].eq('payment'), 'net_amount'] = payments['amount']
    payments.loc[completed & payments['transaction_type'].eq('refund'), 'net_amount'] = -payments['amount']
    payments['payment_month'] = payments['payment_date'].dt.to_period('M')
    return clients, memberships, sessions, payments, packages

def trainer_names(sessions):
    ids = sorted(sessions['trainer_id'].dropna().astype(str).unique().tolist())
    mapping = {trainer_id: f'Trainer {i:02d}' for i, trainer_id in enumerate(ids, 1)}
    path = PROJECT_ROOT / 'data' / 'trainer_names.csv'
    if path.exists():
        names = pd.read_csv(path, dtype='string')
        if {'trainer_id', 'trainer_name'}.issubset(names.columns):
            for _, record in names.dropna(subset=['trainer_id', 'trainer_name']).iterrows():
                name = str(record['trainer_name']).strip()
                if name and record['trainer_id'] in mapping:
                    mapping[str(record['trainer_id'])] = name
    return mapping

def business_overview(clients, sessions, payments, report_date):
    st.subheader('Business Overview')
    payments = payments.loc[payments['payment_date'] <= report_date]
    sessions = sessions.loc[sessions['session_date'] <= report_date]
    a, b, c = st.columns(3)
    a.metric('Net Collected Cash', money(total_revenue(payments)))
    b.metric('Clients Joined', f"{clients.loc[clients['join_date'] <= report_date, 'client_id'].nunique():,}")
    c.metric('Completed Sessions', f"{sessions['session_status'].eq('completed').sum():,}")
    st.divider()
    st.subheader('Monthly Cash Collected')
    monthly = monthly_revenue(payments)
    if monthly.empty:
        st.info('No payment activity for this reporting date.')
    else:
        monthly['Month'] = monthly['payment_month'].dt.to_timestamp()
        cash_chart(monthly, 'Month', 'net_cash_collected')
    with st.expander('View Payment Records'):
        table(payments.sort_values('payment_date', ascending=False), 'payment_records.csv')

def revenue_page(payments, report_date):
    st.subheader('Revenue Analytics')
    st.caption('Cash basis: completed payments less completed refunds. Not profit or earned revenue.')
    filtered = payments.loc[payments['payment_date'] <= report_date].copy()
    refunds = refund_analysis(filtered).iloc[0]
    clients = revenue_per_client(filtered)
    paying = clients.loc[clients['net_cash_collected'] > 0]
    a, b, c, d = st.columns(4)
    a.metric('Net Collected Cash', money(total_revenue(filtered)))
    b.metric('Gross Collected Cash', money(refunds['gross_cash_collected']))
    c.metric('Refunded Cash', money(refunds['refunded_cash']))
    d.metric('Refund % of Gross', f"{refunds['refund_amount_pct_of_gross']:.2f}%")
    st.metric('Average Net Cash per Paying Client', money(total_revenue(filtered) / len(paying) if len(paying) else 0))
    st.divider()
    monthly = revenue_growth(filtered)
    st.subheader('Monthly Cash Collected')
    if monthly.empty:
        st.info('No payments match the selected reporting date.')
    else:
        monthly['Month'] = monthly['payment_month'].dt.to_timestamp()
        cash_chart(monthly, 'Month', 'net_cash_collected')
        with st.expander('View Month-over-Month Growth'):
            st.caption('Growth is undefined when the previous month has zero or negative net cash.')
            table(monthly.drop(columns='Month'), 'revenue_growth.csv')
    st.subheader('Cash Collected by Service Type')
    by_type = revenue_by_type(filtered)
    if not by_type.empty:
        by_type['Service Type'] = by_type['payment_type'].map(pretty)
        cash_chart(by_type, 'Service Type', 'net_cash_collected', kind='bar')
        with st.expander('View Service Breakdown'):
            table(by_type.drop(columns='Service Type'), 'revenue_by_type.csv')
    with st.expander('View Client-Level Cash Collected'):
        table(clients, 'revenue_per_client.csv')

def client_page(clients, memberships, sessions, report_date):
    st.subheader('Client Analytics')
    st.caption('Membership retention describes contracts, not overall client retention. Inactivity is not confirmed churn.')
    joined = clients.loc[clients['join_date'] <= report_date].copy()
    started = memberships.loc[memberships['start_date'] <= report_date].copy()
    past = sessions.loc[sessions['session_date'] <= report_date].copy()
    canceled = started.loc[started['cancellation_date'].notna() &
                           (started['cancellation_date'] <= report_date)].copy()
    canceled['status'] = 'canceled'
    acquisitions = client_acquisition(joined)
    cancellations = membership_cancellations(canceled)
    retention = membership_cohort_retention(started, as_of=report_date)
    inactive = pay_per_session_inactivity(past, as_of=report_date)
    a, b, c, d = st.columns(4)
    a.metric('Clients Joined', f"{joined['client_id'].nunique():,}")
    b.metric('Memberships Started', f'{len(started):,}')
    c.metric('Membership Cancellations', f'{len(canceled):,}')
    d.metric('Inactive Pay-Per-Session Clients (60 Days)',
             f"{int(inactive['inactive_60d_proxy'].sum()) if not inactive.empty else 0:,}")
    st.divider()
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('New Clients by Month')
        if acquisitions.empty:
            st.info('No clients have joined yet.')
        else:
            acquisitions['Month'] = acquisitions['join_month'].dt.to_timestamp()
            fig = px.bar(acquisitions, x='Month', y='new_clients', text='new_clients', labels={'new_clients': 'New Clients'})
            chart(fig)
    with right:
        st.subheader('Membership Cancellations')
        if cancellations.empty:
            st.info('No membership cancellations yet.')
        else:
            cancellations['Month'] = cancellations['cancellation_month'].dt.to_timestamp()
            fig = px.bar(cancellations, x='Month', y='cancellations', text='cancellations',
                         labels={'cancellations': 'Cancellations'})
            chart(fig)
    st.subheader('Membership Cohort Retention')
    st.caption('Percent of eligible contracts in force at each monthly anniversary.')
    if retention.empty:
        st.info('No eligible membership cohorts.')
    else:
        retention['Start Cohort'] = retention['cohort_month'].astype(str)
        fig = px.line(retention, x='months_since_start', y='retention_pct', color='Start Cohort',
                      markers=True, labels={'months_since_start': 'Months Since Start', 'retention_pct': 'Retention (%)'})
        fig.update_yaxes(range=[0, 105], ticksuffix='%')
        chart(fig)
        with st.expander('View Retention Data'):
            table(retention.drop(columns='Start Cohort'), 'membership_retention.csv')
    with st.expander('View Pay-Per-Session Inactivity'):
        st.caption('Only clients with at least one completed pay-per-session visit are included.')
        table(inactive, 'per_session_inactivity.csv')
    if report_date.normalize() == pd.Timestamp('2026-11-20') and st.session_state.get('data_source', 'Sample Data') == 'Sample Data':
        with st.expander('View Client Status (Dataset Snapshot)'):
            table(client_status_summary(clients), 'client_status.csv')

def session_page(sessions, packages, report_date, labels):
    st.subheader('Session Analytics')
    st.caption('Attendance reflects sessions scheduled through the selected reporting date.')
    past = sessions.loc[sessions['session_date'] <= report_date].copy()
    options = sorted(past['pricing_model'].dropna().unique().tolist())
    selected = st.multiselect('Pricing Models', options, default=options, format_func=pretty)
    filtered = past.loc[past['pricing_model'].isin(selected)].copy()
    statuses = filtered['session_status']
    completed, canceled, no_shows = (int(statuses.eq(v).sum()) for v in ['completed', 'canceled', 'no_show'])
    a, b, c, d, e = st.columns(5)
    a.metric('Scheduled', f'{len(filtered):,}')
    b.metric('Completed', f'{completed:,}')
    c.metric('Canceled', f'{canceled:,}')
    d.metric('No-Shows', f'{no_shows:,}')
    e.metric('Completion Rate', f'{100 * completed / len(filtered) if len(filtered) else 0:.1f}%')
    st.divider()
    attendance = attendance_summary(filtered)
    monthly = monthly_attendance(filtered)
    st.subheader('Attendance Breakdown')
    if attendance.empty:
        st.info('No sessions match these filters.')
    else:
        attendance['Status'] = attendance['session_status'].map(pretty)
        fig = px.bar(attendance, x='Status', y='sessions', text='sessions', labels={'sessions': 'Sessions'})
        chart(fig)
        with st.expander('View Attendance Data'):
            table(attendance.drop(columns='Status'), 'attendance_summary.csv')
    st.subheader('Monthly Attendance')
    if monthly.empty:
        st.info('No monthly session data available.')
    else:
        monthly['Month'] = monthly['session_month'].dt.to_timestamp()
        plot = monthly.rename(columns={'completed': 'Completed', 'canceled': 'Canceled', 'no_show': 'No-Show'})
        fig = px.line(plot, x='Month', y=['Completed', 'Canceled', 'No-Show'], markers=True,
                      labels={'value': 'Sessions', 'variable': 'Status'})
        chart(fig)
        with st.expander('View Monthly Attendance Data'):
            table(monthly.drop(columns='Month'), 'monthly_attendance.csv')
    with st.expander('View Activity by Pricing Model'):
        table(activity_by_pricing_model(filtered), 'activity_by_pricing_model.csv')
    st.subheader('Package Redemption (Dataset Snapshot)')
    st.caption('Balances are stored snapshots and do not change when the reporting date changes.')
    redemptions = package_redemption(packages)
    if not redemptions.empty:
        st.metric('Average Package Redemption', f"{redemptions['redemption_rate_pct'].mean():.1f}%")
    with st.expander('View Package Details'):
        table(redemptions, 'package_redemption.csv')

def trainer_page(sessions, payments, report_date, labels):
    st.subheader('Trainer Analytics')
    st.caption('Cash attributed to trainers includes pay-per-session receipts only, not memberships or packages.')
    past = sessions.loc[sessions['session_date'] <= report_date].copy()
    past_payments = payments.loc[payments['payment_date'] <= report_date].copy()
    ids = sorted(past['trainer_id'].dropna().astype(str).unique().tolist())
    selected = st.multiselect('Trainer Names', ids, default=ids,
                              format_func=lambda i: labels.get(i, i))
    filtered = past.loc[past['trainer_id'].astype(str).isin(selected)].copy()
    activity = trainer_activity(filtered)
    monthly = trainer_monthly_activity(filtered)

    try:
        cash = trainer_direct_session_cash(sessions, past_payments)
    except ValueError as error:
        st.warning(f'Trainer cash attribution is unavailable: {error}')
        cash = pd.DataFrame(columns=['trainer_id', 'attributable_net_session_cash'])
    cash = cash.loc[cash['trainer_id'].astype(str).isin(selected)].copy()
    a, b, c, d = st.columns(4)
    a.metric('Selected Trainers', f'{len(selected):,}')
    b.metric('Completed Sessions', f"{int(activity['completed_sessions'].sum()) if not activity.empty else 0:,}")
    c.metric('Completed Training Hours', f"{activity['completed_training_hours'].sum() if not activity.empty else 0:,.1f}")
    d.metric('Attributable Session Cash', money(cash['attributable_net_session_cash'].sum() if not cash.empty else 0))
    st.divider()
    st.subheader('Completed Hours by Trainer')
    if activity.empty:
        st.info('No sessions match the selected trainers.')
    else:
        activity['Trainer'] = activity['trainer_id'].astype(str).map(labels)
        fig = px.bar(activity, x='Trainer', y='completed_training_hours', text_auto='.1f',
                     labels={'completed_training_hours': 'Completed Training Hours'})
        chart(fig)
        with st.expander('View Trainer Activity'):
            table(activity.drop(columns='Trainer'), 'trainer_activity.csv', labels)
    st.subheader('Monthly Completed Sessions by Trainer')
    if monthly.empty:
        st.info('No completed sessions match the selected trainers.')
    else:
        monthly['Month'] = monthly['session_month'].dt.to_timestamp()
        monthly['Trainer'] = monthly['trainer_id'].astype(str).map(labels)
        fig = px.line(monthly, x='Month', y='completed_sessions', color='Trainer', markers=True,
                      labels={'completed_sessions': 'Completed Sessions'})
        chart(fig)
        with st.expander('View Monthly Trainer Activity'):
            table(monthly.drop(columns=['Month', 'Trainer']), 'trainer_monthly_activity.csv', labels)
    st.subheader('Attributable Pay-Per-Session Cash')
    if not cash.empty:
        cash['Trainer'] = cash['trainer_id'].astype(str).map(labels)
        cash_chart(cash, 'Trainer', 'attributable_net_session_cash', kind='bar')
    with st.expander('View Trainer Cash Data'):
        table(cash.drop(columns='Trainer', errors='ignore'), 'trainer_session_cash.csv', labels)


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
    missing_files = set(DATASET_NAMES) - set(datasets)
    if missing_files:
        raise ValueError(f'Missing datasets: {", ".join(sorted(missing_files))}')
    for name, required in ANALYTICS_COLUMNS.items():
        missing = required - set(datasets[name].columns)
        if missing:
            raise ValueError(f'{pretty(name)}: missing dashboard columns: {", ".join(sorted(missing))}')

    result = {name: frame.copy() for name, frame in datasets.items()}
    payments = result['payments']
    payments['amount'] = pd.to_numeric(payments['amount'], errors='raise')
    if payments['amount'].isna().any() or (payments['amount'] <= 0).any():
        raise ValueError('Payment amounts must be positive.')
    if (payments['transaction_type'].eq('refund') &
            ~payments['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions.')
    completed = payments['payment_status'].eq('completed')
    payments['net_amount'] = 0.0
    payments.loc[completed & payments['transaction_type'].eq('payment'), 'net_amount'] = payments['amount']
    payments.loc[completed & payments['transaction_type'].eq('refund'), 'net_amount'] = -payments['amount']
    payments['payment_month'] = payments['payment_date'].dt.to_period('M')
    return result


def data_import_page():
    st.header('Data Import')
    st.caption('Import your business CSVs in memory. Sample and processed files are never overwritten.')
    st.info('Upload all five datasets. Map unfamiliar columns, validate the data, then activate the imported data in the sidebar.')

    uploads = {}
    manual = {}
    for name in DATASET_NAMES:
        with st.expander(f'{pretty(name)} CSV', expanded=name == 'clients'):
            uploaded = st.file_uploader(f'Upload {pretty(name)}', type=['csv'], key=f'upload_{name}')
            if uploaded is None:
                st.caption('No file selected.')
                continue
            uploads[name] = uploaded.getvalue()
            try:
                preview = pd.read_csv(BytesIO(uploads[name]))
                if preview.empty:
                    st.error('This file contains no data rows.')
                    continue
                st.caption(f'{len(preview):,} rows · {len(preview.columns)} columns')
                st.dataframe(preview.head(5), use_container_width=True, hide_index=True)
                try:
                    suggested = suggest_column_mapping(preview)
                except ValueError as error:
                    suggested = {}
                    st.warning(f'Automatic mapping needs attention: {error}')
                st.markdown('**Column mapping**')
                st.caption('Choose a source column for each required platform field. Optional fields can be mapped too.')
                available = list(preview.columns)
                targets = sorted(ANALYTICS_COLUMNS[name] | set(REQUIRED_COLUMNS[name]))
                selected_sources = {}
                for target in targets:
                    guessed = next((source for source, mapped in suggested.items() if mapped == target), None)
                    choices = ['— Not mapped —'] + available
                    selection = st.selectbox(pretty(target), choices,
                                             index=choices.index(guessed) if guessed in choices else 0,
                                             key=f'map_{name}_{target}')
                    if selection != '— Not mapped —':
                        selected_sources[selection] = target
                manual[name] = selected_sources
            except Exception as error:
                st.error(f'Cannot preview this CSV: {error}')

    import_signature = hashlib.sha256()
    for dataset_name in DATASET_NAMES:
        import_signature.update(dataset_name.encode('utf-8'))
        import_signature.update(uploads.get(dataset_name, b''))
        import_signature.update(repr(sorted(manual.get(dataset_name, {}).items())).encode('utf-8'))
    current_signature = import_signature.hexdigest()
    if (st.session_state.get('approved_import_signature') is not None and
            st.session_state['approved_import_signature'] != current_signature):
        st.session_state.pop('imported_data', None)
        st.session_state.pop('last_import_result', None)
        st.session_state.pop('approved_import_signature', None)

    st.divider()
    col1, col2 = st.columns([1, 2], gap='large')
    with col1:
        run = st.button('Validate Uploaded Data', type='primary', use_container_width=True,
                        disabled=len(uploads) != len(DATASET_NAMES))
    with col2:
        st.caption(f'{len(uploads)} of {len(DATASET_NAMES)} required files selected. '
                   'Uploaded data stays in this Streamlit session.')

    if run:
        try:
            result = import_datasets(uploads, manual_mappings=manual)
            if result['valid']:
                try:
                    prepared = prepare_dashboard_data(result['datasets'])
                except (ValueError, KeyError, TypeError) as error:
                    result['valid'] = False
                    result['dashboard_error'] = str(error)
                else:
                    st.session_state['imported_data'] = prepared
                    st.session_state['approved_import_signature'] = current_signature
            if not result['valid']:
                st.session_state.pop('imported_data', None)
                st.session_state.pop('approved_import_signature', None)
            st.session_state['last_import_result'] = result
        except Exception as error:
            st.session_state.pop('imported_data', None)
            st.session_state.pop('approved_import_signature', None)
            st.session_state.pop('last_import_result', None)
            st.error(f'Import failed: {error}')

    result = st.session_state.get('last_import_result')
    if result is not None:
        if result['valid']:
            st.success('Import passed validation. Choose Imported Data in the sidebar, then open an analytics page.')
        else:
            st.error('Import needs corrections before it can be used for analytics.')
            if result.get('dashboard_error'):
                st.warning(result['dashboard_error'])
        report = result['validation_report']
        if not report.empty:
            st.subheader('Validation Findings')
            st.dataframe(report, hide_index=True, use_container_width=True)
            st.download_button('Download Validation Report', report.to_csv(index=False),
                               file_name='validation_report.csv', mime='text/csv')
        else:
            st.caption('No validation issues were found.')
        st.subheader('Column Mapping Results')
        for name, mapping in result['mapping_reports'].items():
            with st.expander(f'{pretty(name)} Mapping'):
                st.dataframe(mapping, hide_index=True, use_container_width=True)


def business_reports_page(datasets, report_date, source):
    st.header('Business Reports')
    st.caption('Generate an HTML business report and downloadable CSV tables.')
    st.info('Reports use the selected data source and reporting date. Client statuses and package balances are snapshots, not historical reconstructions.')
    st.write(f'**Data source:** {source}  |  **Reporting date:** {report_date.date().isoformat()}')

    if st.button('Generate Business Report', type='primary'):
        try:
            result = build_reports(datasets, report_date)
            with tempfile.TemporaryDirectory(prefix='pt_analytics_report_') as temp_dir:
                files = export_reports(result, output_dir=temp_dir)
                html_file = next(path for path in files if path.name == 'business_report.html')
                html_bytes = html_file.read_bytes()
                archive_buffer = BytesIO()
                with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for path in files:
                        archive.write(path, arcname=str(path.relative_to(Path(temp_dir))))
                archive_bytes = archive_buffer.getvalue()
            st.session_state['business_report_downloads'] = {
                'html': html_bytes,
                'zip': archive_bytes,
                'summary': result['summary'],
                'signature': (source, str(report_date), st.session_state.get('approved_import_signature'))
            }
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            st.session_state.pop('business_report_downloads', None)
            st.error(f'Could not generate the report: {error}')

    saved = st.session_state.get('business_report_downloads')
    signature = (source, str(report_date), st.session_state.get('approved_import_signature'))
    if saved and saved['signature'] == signature:
        summary = saved['summary']
        st.success('Business report generated successfully.')
        first, second, third = st.columns(3)
        first.metric('Net Collected Cash', money(summary['net_cash_collected']))
        second.metric('Clients Joined', f"{summary['clients_joined']:,}")
        third.metric('Completed Sessions', f"{summary['completed_sessions']:,}")
        col_html, col_zip = st.columns(2)
        with col_html:
            st.download_button('Download HTML Report', saved['html'],
                               file_name='business_report.html', mime='text/html',
                               use_container_width=True)
        with col_zip:
            st.download_button('Download All Reports (ZIP)', saved['zip'],
                               file_name='personal_training_reports.zip', mime='application/zip',
                               use_container_width=True)
        st.caption('The ZIP contains the HTML report and individual CSV tables.')
    elif saved:
        st.info('The data source or reporting date changed. Generate the report again to refresh the downloads.')


def main():
    st.title('Personal Training Analytics Platform')
    st.caption('Business intelligence for training studios and independent trainers')

    st.sidebar.header('Navigation')
    pages = ['Business Overview', 'Revenue Analytics', 'Client Analytics',
             'Session Analytics', 'Trainer Analytics', 'Data Import', 'Business Reports']
    page = st.sidebar.radio('Dashboard Page', pages)
    st.sidebar.divider()
    st.sidebar.header('Data Source')
    source_options = ['Sample Data']
    if 'imported_data' in st.session_state:
        source_options.append('Imported Data')
    if st.session_state.get('data_source') not in source_options:
        st.session_state['data_source'] = 'Sample Data'
    source = st.sidebar.radio('Use Dataset', source_options, key='data_source')
    st.sidebar.caption('Imported files are held only in the current app session.')

    if page == 'Data Import':
        data_import_page()
        return

    try:
        if source == 'Imported Data':
            data = st.session_state['imported_data']
            clients, memberships, sessions, payments, packages = (
                data[name] for name in ['clients', 'memberships', 'sessions', 'payments', 'packages'])
        else:
            clients, memberships, sessions, payments, packages = load_data()
    except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f'Could not load the selected datasets: {error}')
        if source == 'Sample Data':
            st.info('Run generate_data.py and clean_data.py, then refresh this page.')
        st.stop()

    st.caption(f'Viewing: **{source}**')
    labels = trainer_names(sessions)
    st.sidebar.divider()
    st.sidebar.header('Reporting Filters')
    default_date = pd.Timestamp('2026-11-20').date()
    report_date = pd.Timestamp(st.sidebar.date_input('Reporting Date', value=default_date)) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    st.sidebar.caption('Metrics are calculated through the selected date, except labeled snapshots.')
    with st.sidebar.expander('Customize Trainer Names'):
        st.caption('Names apply during this session. For permanent sample-data names, create data/trainer_names.csv.')
        edits = pd.DataFrame({'Trainer ID': list(labels), 'Trainer Name': list(labels.values())})
        updated = st.data_editor(edits, hide_index=True, disabled=['Trainer ID'],
                                 use_container_width=True, key=f'trainer_name_editor_{source}')
        for _, row in updated.iterrows():
            name = str(row['Trainer Name']).strip()
            if name:
                labels[str(row['Trainer ID'])] = name
        st.download_button('Save Name Mapping CSV', data=updated.rename(columns={
            'Trainer ID': 'trainer_id', 'Trainer Name': 'trainer_name'}).to_csv(index=False),
            file_name='trainer_names.csv', mime='text/csv')

    try:
        if page == 'Business Overview':
            business_overview(clients, sessions, payments, report_date)
        elif page == 'Revenue Analytics':
            revenue_page(payments, report_date)
        elif page == 'Client Analytics':
            client_page(clients, memberships, sessions, report_date)
        elif page == 'Session Analytics':
            session_page(sessions, packages, report_date, labels)
        elif page == 'Trainer Analytics':
            trainer_page(sessions, payments, report_date, labels)
        elif page == 'Business Reports':
            business_reports_page({'clients': clients, 'memberships': memberships,
                                   'packages': packages, 'sessions': sessions,
                                   'payments': payments}, report_date, source)
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        st.error(f'Unable to calculate this page from the selected dataset: {error}')
        st.info('Review the imported columns and values on the Data Import page.')


if __name__ == '__main__':
    main()
