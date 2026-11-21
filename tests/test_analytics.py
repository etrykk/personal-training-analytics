#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Automated Business Analytics Tests

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from revenue_analysis import (total_revenue, monthly_revenue, revenue_by_type,
                              revenue_growth, revenue_per_client, refund_analysis)
from client_analysis import (client_acquisition, client_status_summary,
                             membership_cancellations, membership_cohort_retention,
                             pay_per_session_inactivity)
from session_analysis import (attendance_summary, monthly_attendance,
                              activity_by_pricing_model, package_redemption)
from trainer_analysis import (trainer_activity, trainer_monthly_activity,
                              trainer_direct_session_cash)


@pytest.fixture
def payments():
    data = pd.DataFrame({
        'payment_id': ['P1', 'P2', 'P3', 'P4'],
        'client_id': ['C1', 'C1', 'C2', 'C2'],
        'payment_date': pd.to_datetime(['2026-01-05', '2026-01-10',
                                        '2026-02-05', '2026-02-06']),
        'amount': [1000.0, 150.0, 400.0, 100.0],
        'payment_status': ['completed', 'completed', 'completed', 'pending'],
        'transaction_type': ['payment', 'refund', 'payment', 'payment'],
        'payment_type': ['membership', 'membership', 'session', 'session']
    })
    data['net_amount'] = [1000.0, -150.0, 400.0, 0.0]
    data['payment_month'] = data['payment_date'].dt.to_period('M')
    return data


@pytest.fixture
def sessions():
    return pd.DataFrame({
        'session_id': ['S1', 'S2', 'S3', 'S4'],
        'client_id': ['C1', 'C1', 'C2', 'C2'],
        'trainer_id': ['T1', 'T1', 'T1', 'T2'],
        'session_date': pd.to_datetime(['2026-01-05', '2026-01-06',
                                        '2026-02-05', '2026-02-06']),
        'session_status': ['completed', 'canceled', 'no_show', 'completed'],
        'duration_minutes': [60, 30, 45, 90],
        'pricing_model': ['per_session', 'per_session', 'monthly', 'per_session'],
        'session_fee': [50, 50, 0, 75]
    })


def test_total_net_cash(payments):
    assert total_revenue(payments) == pytest.approx(1250.0)


def test_monthly_cash(payments):
    report = monthly_revenue(payments)
    assert report['net_cash_collected'].tolist() == pytest.approx([850.0, 400.0])


def test_revenue_by_type(payments):
    report = revenue_by_type(payments).set_index('payment_type')
    assert report.loc['membership', 'net_cash_collected'] == pytest.approx(850.0)
    assert report.loc['session', 'net_cash_collected'] == pytest.approx(400.0)


def test_revenue_growth(payments):
    report = revenue_growth(payments)
    assert pd.isna(report.loc[0, 'growth_pct'])
    assert report.loc[1, 'growth_pct'] == pytest.approx((400 - 850) / 850 * 100)


def test_revenue_per_client(payments):
    report = revenue_per_client(payments).set_index('client_id')
    assert report.loc['C1', 'net_cash_collected'] == pytest.approx(850.0)
    assert report.loc['C2', 'net_cash_collected'] == pytest.approx(400.0)


def test_refund_analysis(payments):
    report = refund_analysis(payments).iloc[0]
    assert report['gross_cash_collected'] == pytest.approx(1400.0)
    assert report['refunded_cash'] == pytest.approx(150.0)
    assert report['net_cash_collected'] == pytest.approx(1250.0)
    assert report['refund_amount_pct_of_gross'] == pytest.approx(150 / 1400 * 100)


def test_client_acquisition():
    clients = pd.DataFrame({
        'join_date': pd.to_datetime(['2026-01-01', '2026-01-10', '2026-02-01'])
    })
    report = client_acquisition(clients)
    assert report['new_clients'].tolist() == [2, 1]


def test_client_status_snapshot():
    clients = pd.DataFrame({'status': ['active', 'active', 'inactive']})
    report = client_status_summary(clients).set_index('status')
    assert report.loc['active', 'clients'] == 2
    assert report.loc['inactive', 'clients'] == 1


def test_membership_cancellations():
    memberships = pd.DataFrame({
        'status': ['canceled', 'active', 'canceled'],
        'cancellation_date': pd.to_datetime(['2026-01-12', None, '2026-02-12'])
    })
    report = membership_cancellations(memberships)
    assert report['cancellations'].tolist() == [1, 1]


def test_membership_cohort_retention():
    memberships = pd.DataFrame({
        'membership_id': ['M1', 'M2'],
        'start_date': pd.to_datetime(['2026-01-01', '2026-01-01']),
        'end_date': pd.to_datetime(['2026-12-31', '2026-01-20']),
        'cancellation_date': pd.to_datetime([None, '2026-01-20'])
    })
    report = membership_cohort_retention(memberships, as_of='2026-03-01')
    month_one = report.loc[report['months_since_start'].eq(1)].iloc[0]
    assert month_one['eligible_memberships'] == 2
    assert month_one['retained_memberships'] == 1
    assert month_one['retention_pct'] == pytest.approx(50.0)


def test_pay_per_session_inactivity(sessions):
    report = pay_per_session_inactivity(sessions, as_of='2026-04-10', inactivity_days=60)
    result = report.set_index('client_id')
    assert bool(result.loc['C1', 'inactive_60d_proxy']) is True
    assert bool(result.loc['C2', 'inactive_60d_proxy']) is True


def test_attendance_summary(sessions):
    report = attendance_summary(sessions).set_index('session_status')
    assert report.loc['completed', 'sessions'] == 2
    assert report.loc['completed', 'pct_of_scheduled'] == pytest.approx(50.0)


def test_monthly_attendance(sessions):
    report = monthly_attendance(sessions)
    assert report['scheduled'].tolist() == [2, 2]
    assert report['completed'].tolist() == [1, 1]
    assert report['completion_rate_pct'].tolist() == pytest.approx([50.0, 50.0])


def test_activity_by_pricing_model(sessions):
    report = activity_by_pricing_model(sessions).set_index('pricing_model')
    assert report.loc['per_session', 'completed'] == 2
    assert report.loc['monthly', 'no_show'] == 1


def test_package_redemption():
    packages = pd.DataFrame({
        'package_id': ['PK1'], 'client_id': ['C1'], 'package_sessions': [10],
        'sessions_redeemed': [4], 'sessions_remaining': [6], 'status': ['active']
    })
    report = package_redemption(packages)
    assert report.loc[0, 'redemption_rate_pct'] == pytest.approx(40.0)


def test_trainer_activity(sessions):
    report = trainer_activity(sessions).set_index('trainer_id')
    assert report.loc['T1', 'scheduled_sessions'] == 3
    assert report.loc['T1', 'completed_sessions'] == 1
    assert report.loc['T1', 'completed_training_hours'] == pytest.approx(1.0)
    assert report.loc['T2', 'completed_training_hours'] == pytest.approx(1.5)


def test_trainer_monthly_activity(sessions):
    report = trainer_monthly_activity(sessions).set_index('trainer_id')
    assert report.loc['T1', 'completed_sessions'] == 1
    assert report.loc['T2', 'completed_training_hours'] == pytest.approx(1.5)


def test_trainer_direct_session_cash(sessions):
    payments = pd.DataFrame({
        'session_id': ['S1', 'S4', 'S4'],
        'amount': [50.0, 75.0, 10.0],
        'payment_status': ['completed', 'completed', 'completed'],
        'payment_type': ['session', 'session', 'session'],
        'transaction_type': ['payment', 'payment', 'refund']
    })
    report = trainer_direct_session_cash(sessions, payments).set_index('trainer_id')
    assert report.loc['T1', 'attributable_net_session_cash'] == pytest.approx(50.0)
    assert report.loc['T2', 'attributable_net_session_cash'] == pytest.approx(65.0)


def test_unmatched_trainer_payment_raises_error(sessions):
    payments = pd.DataFrame({
        'session_id': ['missing'], 'amount': [50.0],
        'payment_status': ['completed'], 'payment_type': ['session'],
        'transaction_type': ['payment']
    })
    with pytest.raises(ValueError, match='cannot be matched'):
        trainer_direct_session_cash(sessions, payments)
