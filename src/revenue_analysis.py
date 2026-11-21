#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Revenue Analysis (cash basis)

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'revenue'


def load_payments(path=None):
    path = Path(path) if path else PROCESSED_DIR / 'payments.csv'
    data = pd.read_csv(path)
    required = {'payment_id', 'client_id', 'payment_date', 'amount',
                'payment_status', 'transaction_type', 'payment_type'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Payments missing columns: {sorted(missing)}')
    data['payment_date'] = pd.to_datetime(data['payment_date'], errors='raise')
    data['amount'] = pd.to_numeric(data['amount'], errors='raise')
    if data['amount'].isna().any() or (data['amount'] <= 0).any():
        raise ValueError('Payment amounts must be positive and nonmissing')
    if data['payment_id'].isna().any() or data['payment_id'].duplicated().any():
        raise ValueError('Payment IDs must be present and unique')
    allowed = {'completed', 'pending', 'failed'}
    if not data['payment_status'].isin(allowed).all():
        raise ValueError('Unknown payment status')
    if not data['transaction_type'].isin(['payment', 'refund']).all():
        raise ValueError('Unknown transaction type')
    if not data['payment_type'].isin(['membership', 'package', 'session']).all():
        raise ValueError('Unknown payment type')
    if (data['transaction_type'].eq('refund') & ~data['payment_status'].eq('completed')).any():
        raise ValueError('Refunds must be completed transactions')
    completed = data['payment_status'].eq('completed')
    data['net_amount'] = 0.0
    data.loc[completed & data['transaction_type'].eq('payment'), 'net_amount'] = data['amount']
    data.loc[completed & data['transaction_type'].eq('refund'), 'net_amount'] = -data['amount']
    data['payment_month'] = data['payment_date'].dt.to_period('M')
    return data


def total_revenue(payment_data):
    #Net cash collected, not accounting revenue or profit
    return float(payment_data['net_amount'].sum())


def monthly_revenue(payment_data):
    data = payment_data.groupby('payment_month', observed=True)['net_amount'].sum()
    if data.empty:
        return pd.DataFrame(columns=['payment_month', 'net_cash_collected'])
    months = pd.period_range(data.index.min(), data.index.max(), freq='M')
    return data.reindex(months, fill_value=0).rename_axis('payment_month').reset_index(name='net_cash_collected')


def revenue_by_type(payment_data):
    return (payment_data.groupby('payment_type', observed=True)['net_amount']
            .sum().reset_index(name='net_cash_collected'))


def revenue_growth(payment_data):
    monthly = monthly_revenue(payment_data)
    monthly['previous_month'] = monthly['net_cash_collected'].shift(1)
    previous = monthly['previous_month']
    monthly['growth_pct'] = ((monthly['net_cash_collected'] - previous) / previous * 100).where(previous > 0)
    #Undefined when previous month is zero or negative
    return monthly


def revenue_per_client(payment_data):
    return (payment_data.groupby('client_id', observed=True)['net_amount']
            .sum().reset_index(name='net_cash_collected')
            .sort_values('net_cash_collected', ascending=False))


def refund_analysis(payment_data):
    completed = payment_data['payment_status'].eq('completed')
    charges = payment_data.loc[completed & payment_data['transaction_type'].eq('payment')]
    refunds = payment_data.loc[completed & payment_data['transaction_type'].eq('refund')]
    gross = float(charges['amount'].sum())
    refunded = float(refunds['amount'].sum())
    return pd.DataFrame([{'gross_cash_collected': gross, 'refunded_cash': refunded,
                          'net_cash_collected': gross - refunded,
                          'completed_charge_count': len(charges), 'refund_transaction_count': len(refunds),
                          'refund_amount_pct_of_gross': 100 * refunded / gross if gross else 0.0}])


def run_analysis(payment_data=None, output_dir=OUTPUT_DIR):
    payments = load_payments() if payment_data is None else payment_data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'monthly_revenue': monthly_revenue(payments),
               'revenue_by_type': revenue_by_type(payments),
               'revenue_growth': revenue_growth(payments),
               'revenue_per_client': revenue_per_client(payments),
               'refund_analysis': refund_analysis(payments)}
    for name, frame in results.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)
    print(f'Net cash collected: ${total_revenue(payments):,.2f}')
    print(f'Revenue reports saved to {output_dir}')
    return results


if __name__ == '__main__':
    run_analysis()
