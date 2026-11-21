#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Advanced Revenue & Membership Insight Tests

import pandas as pd
import pytest

from src.analytics.financial_membership_insights import (
    monthly_cash_performance,
    payment_status_summary,
    membership_expiry_watchlist,
    membership_coverage_summary,
    membership_period_retention,
    run_financial_membership_insights,
)


@pytest.fixture
def payments():
    return pd.DataFrame([
        ('P1', '2026-01-03', 200, 'completed', 'payment'),
        ('P2', '2026-01-10', 50, 'completed', 'refund'),
        ('P3', '2026-01-11', 100, 'pending', 'payment'),
        ('P4', '2026-01-12', 40, 'failed', 'payment'),
        ('P5', '2026-03-05', 120, 'completed', 'payment'),
        ('P6', '2026-04-01', 900, 'completed', 'payment'),
    ], columns=['payment_id', 'payment_date', 'amount', 'payment_status', 'transaction_type'])


@pytest.fixture
def memberships():
    return pd.DataFrame([
        # C1: a plan change provides continuous coverage at the old expiry
        ('M1', 'C1', 'Basic', '2026-01-01', '2026-03-15', 80, 'expired'),
        ('M2', 'C1', 'Plus', '2026-03-16', None, 140, 'active'),
        # C2: currently active, ends soon
        ('M3', 'C2', 'Basic', '2026-02-01', '2026-03-20', 80, 'canceled'),
        # C3: ended before report date
        ('M4', 'C3', 'Basic', '2026-01-01', '2026-02-01', 80, 'expired'),
        # C4: starts after report date
        ('M5', 'C4', 'Basic', '2026-04-01', None, 80, 'active'),
        # C5: open ended
        ('M6', 'C5', 'Basic', '2026-01-01', None, 80, 'active'),
    ], columns=['membership_id','client_id','plan_name','start_date',
                'end_date','monthly_fee','status'])


def test_cash_revenue_refunds_in_transaction_month(payments):
    result = monthly_cash_performance(payments, '2026-03-15').set_index('payment_month')
    assert result.loc[pd.Period('2026-01'), 'gross_cash_collected'] == pytest.approx(200)
    assert result.loc[pd.Period('2026-01'), 'refunded_cash'] == pytest.approx(50)
    assert result.loc[pd.Period('2026-01'), 'net_cash_collected'] == pytest.approx(150)
    assert result.loc[pd.Period('2026-03'), 'net_cash_collected'] == pytest.approx(120)


def test_monthly_cash_zero_gap_and_no_future(payments):
    result = monthly_cash_performance(payments, '2026-03-15')
    assert result['payment_month'].astype(str).tolist() == ['2026-01', '2026-02', '2026-03']
    assert result['net_cash_collected'].tolist() == pytest.approx([150, 0, 120])
    assert pd.isna(result.loc[2, 'growth_pct'])  # February had zero cash
    assert result.loc[2, 'rolling_3m_net_cash'] == pytest.approx(90)


def test_partial_month_flag(payments):
    partial = monthly_cash_performance(payments, '2026-03-15')
    whole = monthly_cash_performance(payments, '2026-03-31')
    assert bool(partial.iloc[-1]['partial_month']) is True
    assert bool(whole.iloc[-1]['partial_month']) is False


def test_pending_failed_dont_count_as_cash(payments):
    result = monthly_cash_performance(payments, '2026-01-31')
    assert result.iloc[0]['net_cash_collected'] == pytest.approx(150)
    assert result.iloc[0]['completed_charge_count'] == 1
    assert result.iloc[0]['refund_count'] == 1


def test_empty_cash_report():
    empty = pd.DataFrame(columns=['payment_date', 'amount', 'payment_status', 'transaction_type'])
    report = monthly_cash_performance(empty, '2026-03-15')
    assert report.empty
    assert list(report.columns) == [
        'payment_month', 'gross_cash_collected', 'refunded_cash', 'net_cash_collected',
        'completed_charge_count', 'refund_count', 'rolling_3m_net_cash',
        'growth_pct', 'partial_month']


def test_payment_status_excludes_refunds_and_future(payments):
    report = payment_status_summary(payments, '2026-01-31', days=31).set_index('payment_status')
    assert report.loc['completed', 'transaction_count'] == 1
    assert report.loc['pending', 'transaction_count'] == 1
    assert report.loc['failed', 'transaction_count'] == 1
    assert report.loc['completed', 'share_of_attempts_pct'] == pytest.approx(100 / 3)
    assert report.loc['completed', 'attempted_amount'] == pytest.approx(200)


def test_payment_status_zero_attempts_is_unavailable(payments):
    report = payment_status_summary(payments, '2025-12-01', days=30)
    assert report['transaction_count'].tolist() == [0, 0, 0]
    assert report['share_of_attempts_pct'].isna().all()


def test_membership_snapshot_uses_dates_not_status(memberships):
    report = membership_coverage_summary(memberships, '2026-03-10').iloc[0]
    # C1, C2, C5. C2 is still covered even with status canceled.
    assert report['active_clients'] == 3
    assert report['active_membership_records'] == 3
    assert report['expiring_clients_30d'] == 1  # C2; C1 rolls into M2


def test_expiry_watchlist_excludes_continuous_plan_change(memberships):
    report = membership_expiry_watchlist(memberships, '2026-03-10', days_ahead=30)
    assert report['client_id'].tolist() == ['C2']
    assert report.iloc[0]['days_until_expiry'] == 10


def test_expiry_cutoff_inclusive(memberships):
    report = membership_expiry_watchlist(memberships, '2026-03-20', days_ahead=0)
    assert report['client_id'].tolist() == ['C2']
    assert report.iloc[0]['days_until_expiry'] == 0


def test_retention_uses_starting_clients_and_reactivation(memberships):
    report = membership_period_retention(memberships, '2026-03-10', '2026-03-25').iloc[0]
    # C1 and C5 retained, C2 no longer covered. C4 joins later, not counted.
    assert report['starting_clients'] == 3
    assert report['retained_clients'] == 2
    assert report['no_longer_active'] == 1
    assert report['retention_pct'] == pytest.approx(200 / 3)
    assert report['churn_pct'] == pytest.approx(100 / 3)


def test_retention_unavailable_without_starting_clients(memberships):
    report = membership_period_retention(memberships, '2025-01-01', '2025-02-01').iloc[0]
    assert report['starting_clients'] == 0
    assert pd.isna(report['retention_pct'])
    assert pd.isna(report['churn_pct'])


def test_invalid_date_order_raises(memberships):
    with pytest.raises(ValueError, match='on or after'):
        membership_period_retention(memberships, '2026-03-20', '2026-03-10')


def test_full_report_writes_expected_files(payments, memberships, tmp_path):
    reports = run_financial_membership_insights(payments, memberships, '2026-03-10', tmp_path)
    assert len(reports) == 5
    for name in reports:
        assert (tmp_path / f'{name}.csv').exists()


def test_input_frames_not_modified(payments, memberships):
    before_p = payments.copy(deep=True)
    before_m = memberships.copy(deep=True)
    run_financial_membership_insights(payments, memberships, '2026-03-10')
    pd.testing.assert_frame_equal(payments, before_p)
    pd.testing.assert_frame_equal(memberships, before_m)
