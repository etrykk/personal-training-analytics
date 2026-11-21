#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Standardized Business Reporting

from pathlib import Path
from html import escape
import sys
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from revenue_analysis import (monthly_revenue, revenue_by_type, revenue_growth,
                              revenue_per_client, refund_analysis)
from client_analysis import (client_acquisition, client_status_summary,
                             membership_cancellations, membership_cohort_retention,
                             pay_per_session_inactivity)
from session_analysis import (attendance_summary, monthly_attendance,
                              activity_by_pricing_model, package_redemption)
from trainer_analysis import (trainer_activity, trainer_monthly_activity,
                              trainer_direct_session_cash)
from visualizations import cash_chart, count_chart

PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'reports'
DATASET_NAMES = ('clients', 'memberships', 'packages', 'sessions', 'payments')
DATE_COLUMNS = {
    'clients': ['join_date'],
    'memberships': ['start_date', 'end_date', 'cancellation_date'],
    'packages': ['purchase_date', 'expiration_date'],
    'sessions': ['session_date'],
    'payments': ['payment_date']
}


def load_processed_data(data_dir=PROCESSED_DIR):
    data_dir = Path(data_dir)
    datasets = {}
    for name in DATASET_NAMES:
        path = data_dir / f'{name}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing processed dataset: {path}')
        datasets[name] = pd.read_csv(path)
    return datasets


def prepare_report_data(datasets, as_of):
    missing = set(DATASET_NAMES) - set(datasets)
    if missing:
        raise ValueError(f'Missing datasets: {sorted(missing)}')
    as_of = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of):
        raise ValueError('A valid reporting date is required')
    prepared = {}
    cutoff_columns = {
        'clients': 'join_date', 'memberships': 'start_date',
        'packages': 'purchase_date', 'sessions': 'session_date',
        'payments': 'payment_date'
    }
    for name in DATASET_NAMES:
        data = datasets[name].copy()
        for col in DATE_COLUMNS[name]:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='raise')
        cutoff = cutoff_columns[name]
        if cutoff not in data.columns:
            raise ValueError(f'{name} missing required date column: {cutoff}')
        data = data.loc[data[cutoff] < as_of + pd.Timedelta(days=1)].copy()
        prepared[name] = data

    payments = prepared['payments']
    required = {'amount', 'payment_status', 'transaction_type', 'payment_type'}
    if required - set(payments):
        raise ValueError(f'Payments missing columns: {sorted(required - set(payments))}')
    payments['amount'] = pd.to_numeric(payments['amount'], errors='raise')
    if payments['amount'].isna().any() or (payments['amount'] <= 0).any():
        raise ValueError('Payment amounts must be positive and nonmissing')
    if not payments['payment_status'].isin(['completed', 'pending', 'failed']).all():
        raise ValueError('Unexpected payment status')
    if not payments['transaction_type'].isin(['payment', 'refund']).all():
        raise ValueError('Unexpected transaction type')
    if (payments['transaction_type'].eq('refund') &
            ~payments['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions')
    completed = payments['payment_status'].eq('completed')
    payments['net_amount'] = 0.0
    payments.loc[completed & payments['transaction_type'].eq('payment'), 'net_amount'] = payments['amount']
    payments.loc[completed & payments['transaction_type'].eq('refund'), 'net_amount'] = -payments['amount']
    payments['payment_month'] = payments['payment_date'].dt.to_period('M')
    return prepared


def build_reports(datasets, as_of):
    data = prepare_report_data(datasets, as_of)
    clients = data['clients']
    memberships = data['memberships']
    packages = data['packages']
    sessions = data['sessions']
    payments = data['payments']
    reports = {
        'revenue': {
            'monthly_revenue': monthly_revenue(payments),
            'revenue_by_type': revenue_by_type(payments),
            'revenue_growth': revenue_growth(payments),
            'revenue_per_client': revenue_per_client(payments),
            'refund_analysis': refund_analysis(payments)
        },
        'clients': {
            'client_acquisition': client_acquisition(clients),
            'client_status_snapshot': client_status_summary(clients),
            'membership_cancellations': membership_cancellations(memberships),
            'membership_cohort_retention': membership_cohort_retention(memberships, as_of),
            'pay_per_session_inactivity': pay_per_session_inactivity(sessions, as_of)
        },
        'sessions': {
            'attendance_summary': attendance_summary(sessions),
            'monthly_attendance': monthly_attendance(sessions),
            'activity_by_pricing_model': activity_by_pricing_model(sessions),
            'package_redemption_snapshot': package_redemption(packages)
        },
        'trainers': {
            'trainer_activity': trainer_activity(sessions),
            'trainer_monthly_activity': trainer_monthly_activity(sessions),
            'trainer_direct_session_cash': trainer_direct_session_cash(sessions, payments)
        }
    }
    refund = reports['revenue']['refund_analysis'].iloc[0]
    summary = {
        'reporting_date': pd.Timestamp(as_of).date().isoformat(),
        'net_cash_collected': float(refund['net_cash_collected']),
        'gross_cash_collected': float(refund['gross_cash_collected']),
        'refunded_cash': float(refund['refunded_cash']),
        'clients_joined': int(len(clients)),
        'completed_sessions': int(sessions['session_status'].eq('completed').sum())
    }
    return {'summary': summary, 'reports': reports}


def _html_table(frame):
    view = frame.copy()
    for col in view.columns:
        if isinstance(view[col].dtype, pd.PeriodDtype):
            view[col] = view[col].astype(str)
    return view.to_html(index=False, border=0, classes='report-table', escape=True)


def export_reports(result, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = result['summary']
    for section, reports in result['reports'].items():
        folder = output_dir / section
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            path = folder / f'{name}.csv'
            view = frame.copy()
            for col in view.columns:
                if isinstance(view[col].dtype, pd.PeriodDtype):
                    view[col] = view[col].astype(str)
            view.to_csv(path, index=False)
            written.append(path)

    charts = []
    monthly = result['reports']['revenue']['monthly_revenue']
    if not monthly.empty:
        fig = cash_chart(monthly, 'payment_month', 'net_cash_collected', 'Monthly Cash Collected')
        charts.append(fig.to_html(full_html=False, include_plotlyjs=True))
    acquired = result['reports']['clients']['client_acquisition']
    if not acquired.empty:
        fig = count_chart(acquired, 'join_month', 'new_clients', 'Monthly Client Acquisition')
        charts.append(fig.to_html(full_html=False, include_plotlyjs=False))

    sections = []
    for section, reports in result['reports'].items():
        sections.append(f'<h2>{escape(section.title())}</h2>')
        for name, frame in reports.items():
            sections.append(f'<h3>{escape(name.replace("_", " ").title())}</h3>')
            sections.append(_html_table(frame) if not frame.empty else '<p>No records for this report.</p>')
    cards = ''.join(f'<div class="metric"><span>{escape(key.replace("_", " ").title())}</span>'
                    f'<strong>{escape(f"${value:,.2f}" if "cash" in key else str(value))}</strong></div>'
                    for key, value in summary.items() if key != 'reporting_date')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Personal Training Business Report</title><style>
body {{font-family:Arial,sans-serif;max-width:1150px;margin:40px auto;padding:0 22px;color:#222;}}
h1 {{margin-bottom:4px;}}h2 {{margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:10px;}}
.metrics {{display:flex;flex-wrap:wrap;gap:14px;margin:24px 0;}}
.metric {{border:1px solid #ddd;border-radius:10px;padding:16px;min-width:175px;}}
.metric span {{display:block;color:#555;font-size:13px;}}.metric strong {{font-size:22px;}}
.report-table {{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0 30px;}}
.report-table th,.report-table td {{border-bottom:1px solid #ddd;text-align:left;padding:8px;}}
.report-table th {{background:#f3f5f8;}}.note {{color:#555;font-size:13px;}}
</style></head><body><h1>Personal Training Business Report</h1>
<p>Reporting date: {escape(summary['reporting_date'])}</p>
<div class="metrics">{cards}</div>
<p class="note">Cash collected is completed charges less completed refunds, not profit.
Client status and package balances are snapshots, not historical reconstructions.
Membership cohort retention measures contracts, not unique clients.
Inactivity is a proxy, not confirmed churn. Trainer cash only includes directly attributable sessions.</p>
{''.join(charts)}{''.join(sections)}</body></html>'''
    html_path = output_dir / 'business_report.html'
    html_path.write_text(html, encoding='utf-8')
    written.append(html_path)
    return written


def run_reporting(datasets=None, as_of='2026-11-20', output_dir=OUTPUT_DIR):
    if datasets is None:
        datasets = load_processed_data()
    result = build_reports(datasets, as_of)
    files = export_reports(result, output_dir)
    print(f'Report date: {result["summary"]["reporting_date"]}')
    print(f'Net cash collected: ${result["summary"]["net_cash_collected"]:,.2f}')
    print(f'Saved {len(files)} files to {Path(output_dir)}')
    return result


if __name__ == '__main__':
    run_reporting()
