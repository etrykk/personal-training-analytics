#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5A - Revenue Forecast Readiness Tests

import pandas as pd
import pytest

from src.ml.revenue_forecast_readiness import (
    assess_revenue_forecast_readiness, run_revenue_readiness,
)


def example_payments(months=30, first_day=1):
    """Tiny fixture with reproducible calendar months; one charge per month."""
    data = []
    for i, month in enumerate(pd.period_range('2024-01', periods=months, freq='M')):
        data.append({'payment_id': f'P{i+1}',
                     'payment_date': f'{month}-{first_day:02}',
                     'amount': 100 + i,
                     'payment_status': 'completed',
                     'transaction_type': 'payment'})
    return pd.DataFrame(data)


def audit(data, as_of, **kw):
    return assess_revenue_forecast_readiness(data, as_of, **kw)


def test_complete_history_is_eligible_for_backtest():
    data = example_payments()
    results = audit(data, '2026-06-01')
    decision = results['forecast_readiness_decision'].iloc[0]
    assert bool(decision['ready_for_experimental_backtesting'])
    assert int(decision['eligible_complete_months']) == 29
    assert int(decision['training_months_available']) == 23
    assert not bool(decision['ready_for_live_forecasting'])


def test_partial_reporting_month_is_excluded():
    data = example_payments(30)
    history = audit(data, '2026-06-01')['forecast_history']
    june = history.loc[history.payment_month.eq('2026-06')].iloc[0]
    assert bool(june['partial_month'])
    assert not bool(june['eligible_for_backtest'])
    assert june['net_cash_collected'] == 129


def test_first_observed_month_can_be_coverage_uncertain():
    data = example_payments(30, first_day=16)
    history = audit(data, '2026-06-16')['forecast_history']
    assert bool(history.iloc[0]['first_month_coverage_uncertain'])
    assert not bool(history.iloc[0]['eligible_for_backtest'])
    assert history.iloc[0]['exclusion_reason']


def test_zero_months_are_not_automatically_missing():
    data = example_payments(30)
    data = data.loc[~data.payment_date.str.startswith('2025-01')].copy()
    results = audit(data, '2026-06-01')
    history = results['forecast_history']
    assert history.loc[history.payment_month.eq('2025-01'), 'net_cash_collected'].iloc[0] == 0
    assert bool(results['forecast_readiness_checks'].loc[
        lambda x: x.check.eq('Zero-cash months'), 'passed'].iloc[0])


def test_completed_refunds_reduce_cash():
    data = example_payments(30)
    data = pd.concat([data, pd.DataFrame([{'payment_id': 'REF1',
        'payment_date': '2025-05-08', 'amount': 25,
        'payment_status': 'completed', 'transaction_type': 'refund'}])], ignore_index=True)
    hist = audit(data, '2026-06-01')['forecast_history']
    may = hist.loc[hist.payment_month.eq('2025-05')].iloc[0]
    assert may['gross_cash_collected'] == 116
    assert may['refunded_cash'] == 25
    assert may['net_cash_collected'] == 91


def test_pending_and_failed_not_in_cash():
    data = example_payments(30)
    extra = pd.DataFrame([
        {'payment_id': 'pending', 'payment_date': '2025-05-10', 'amount': 500,
         'payment_status': 'pending', 'transaction_type': 'payment'},
        {'payment_id': 'failed', 'payment_date': '2025-05-12', 'amount': 600,
         'payment_status': 'failed', 'transaction_type': 'payment'}])
    data = pd.concat([data, extra], ignore_index=True)
    hist = audit(data, '2026-06-01')['forecast_history']
    assert hist.loc[hist.payment_month.eq('2025-05'), 'net_cash_collected'].iloc[0] == 116


def test_stale_export_does_not_add_fake_zero_months():
    data = example_payments(25)
    results = audit(data, '2027-11-30')
    hist = results['forecast_history']
    assert hist.payment_month.iloc[-1] == '2026-01'
    assert not bool(results['forecast_readiness_decision'].iloc[0]['ready_for_experimental_backtesting'])
    assert not bool(results['forecast_readiness_checks'].loc[
        lambda x: x.check.eq('Source freshness'), 'passed'].iloc[0])


def test_short_history_is_blocked():
    data = example_payments(12)
    result = audit(data, '2024-12-01')['forecast_readiness_decision'].iloc[0]
    assert not bool(result['ready_for_experimental_backtesting'])


def test_month_end_can_be_complete():
    data = example_payments(30)
    hist = audit(data, '2026-06-30')['forecast_history']
    assert not bool(hist.iloc[-1]['partial_month'])


def test_dataframes_are_not_mutated():
    data = example_payments(25)
    original = data.copy(deep=True)
    audit(data, '2026-01-01')
    pd.testing.assert_frame_equal(data, original)


def test_missing_columns_fail_explicitly():
    data = example_payments().drop(columns='transaction_type')
    with pytest.raises(ValueError, match='Missing payment columns'):
        audit(data, '2026-06-01')


def test_invalid_payment_amount_is_rejected():
    data = example_payments()
    data.loc[0, 'amount'] = -10
    with pytest.raises(ValueError):
        audit(data, '2026-06-01')


def test_duplicate_ids_are_rejected():
    data = example_payments()
    data.loc[1, 'payment_id'] = data.loc[0, 'payment_id']
    with pytest.raises(ValueError, match='IDs'):
        audit(data, '2026-06-01')


def test_empty_input_is_graceful():
    data = example_payments(1).iloc[:0]
    results = audit(data, '2026-06-01')
    assert results['forecast_history'].empty
    assert not bool(results['forecast_readiness_decision'].iloc[0]['ready_for_experimental_backtesting'])


def test_future_payments_are_not_included():
    data = example_payments(30)
    results = audit(data, '2024-10-01')
    assert results['forecast_history'].payment_month.max() == '2024-10'
    assert results['forecast_readiness_decision'].iloc[0]['calendar_months'] == 10


def test_invalid_settings_raise():
    data = example_payments()
    with pytest.raises(ValueError, match='min_train_months'):
        audit(data, '2026-06-01', min_train_months=0)
    with pytest.raises(ValueError, match='as_of'):
        audit(data, pd.NaT)


def test_csv_report_generation(tmp_path):
    reports = run_revenue_readiness(example_payments(), '2026-06-01', output_dir=tmp_path)
    assert set(reports) == {'forecast_history', 'forecast_readiness_checks', 'forecast_readiness_decision'}
    for name, frame in reports.items():
        path = tmp_path / f'{name}.csv'
        assert path.exists()
        assert len(pd.read_csv(path)) == len(frame)


def test_no_forecasts_or_client_scores_are_generated():
    reports = audit(example_payments(), '2026-06-01')
    assert 'predicted_cash' not in reports['forecast_history'].columns
    assert 'client_id' not in reports['forecast_history'].columns
    assert not bool(reports['forecast_readiness_decision'].iloc[0]['ready_for_live_forecasting'])
