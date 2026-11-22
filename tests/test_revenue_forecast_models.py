#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5B - Revenue Forecast Candidate Tests

import numpy as np
import pandas as pd
import pytest

from src.ml.revenue_forecast_models import (
    create_revenue_forecast_candidates, run_forecast_candidates,
)


def fixture_history(n=33, start='2024-02', holdout=6, values=None):
    months = pd.period_range(start, periods=n, freq='M').astype(str)
    if values is None:
        values = [100 + 10*i for i in range(n)]
    history = pd.DataFrame({
        'payment_month': months, 'net_cash_collected': values,
        'eligible_for_backtest': [True] * n,
    })
    decision = pd.DataFrame([{
        'ready_for_experimental_backtesting': True,
        'reserved_backtest_months': holdout,
        'training_months_available': n-holdout,
    }])
    return history, decision


def result(history=None, decision=None, **kwargs):
    if history is None:
        history, decision = fixture_history()
    return create_revenue_forecast_candidates(history, decision, **kwargs)


def test_output_has_all_three_models_and_three_horizons():
    candidates = result()['forecast_model_candidates']
    assert len(candidates) == 9
    assert candidates.model.nunique() == 3
    assert set(candidates.horizon_months) == {1, 2, 3}


def test_training_and_holdout_are_chronological():
    decision = result()['forecast_model_decision'].iloc[0]
    assert decision.training_start_month == '2024-02'
    assert decision.training_end_month == '2026-04'
    assert decision.holdout_start_month == '2026-05'
    assert decision.holdout_end_month == '2026-10'
    assert decision.training_months == 27
    assert decision.reserved_holdout_months == 6


def test_last_month_naive_values():
    rows = result()['forecast_model_candidates'].query("model == 'last_month'")
    assert rows.forecast_net_cash.tolist() == [360.0] * 3


def test_three_month_mean_values():
    rows = result()['forecast_model_candidates'].query("model == 'trailing_3_month_mean'")
    assert rows.forecast_net_cash.tolist() == [350.0] * 3


def test_linear_trend_correct_for_straight_line_data():
    rows = result()['forecast_model_candidates'].query("model == 'linear_trend_last_6_months'")
    assert rows.forecast_net_cash.tolist() == [370.0, 380.0, 390.0]


def test_forecast_months_are_after_training_end():
    rows = result()['forecast_model_candidates']
    assert set(rows.forecast_month) == {'2026-05', '2026-06', '2026-07'}


def test_holdout_values_do_not_affect_any_forecast():
    history, decision = fixture_history()
    original = result(history, decision)['forecast_model_candidates']
    modified = history.copy()
    modified.loc[27:, 'net_cash_collected'] = [1000000] * 6
    comparison = result(modified, decision)['forecast_model_candidates']
    pd.testing.assert_frame_equal(original, comparison)


def test_mutation_does_not_touch_source_frames():
    history, decision = fixture_history()
    hcopy, dcopy = history.copy(deep=True), decision.copy(deep=True)
    result(history, decision)
    pd.testing.assert_frame_equal(history, hcopy)
    pd.testing.assert_frame_equal(decision, dcopy)


def test_can_read_csv_serialized_boolean_flags(tmp_path):
    history, decision = fixture_history()
    history.to_csv(tmp_path / 'h.csv', index=False)
    decision.to_csv(tmp_path / 'd.csv', index=False)
    output = result(pd.read_csv(tmp_path / 'h.csv'), pd.read_csv(tmp_path / 'd.csv'))
    assert len(output['forecast_model_candidates']) == 9


def test_rejects_false_readiness():
    history, decision = fixture_history()
    decision.loc[0, 'ready_for_experimental_backtesting'] = False
    with pytest.raises(ValueError, match='did not approve'):
        result(history, decision)


def test_rejects_invalid_boolean_flag_strings():
    history, decision = fixture_history()
    history['eligible_for_backtest'] = history['eligible_for_backtest'].astype(object)
    history.loc[0, 'eligible_for_backtest'] = 'maybe'
    with pytest.raises(ValueError, match='True or False'):
        result(history, decision)


def test_rejects_gapped_history():
    history, decision = fixture_history()
    history.loc[7, 'eligible_for_backtest'] = False
    decision.loc[0, 'training_months_available'] = 26
    with pytest.raises(ValueError, match='consecutive'):
        result(history, decision)


def test_rejects_duplicate_month():
    history, decision = fixture_history()
    history.loc[3, 'payment_month'] = history.loc[2, 'payment_month']
    with pytest.raises(ValueError, match='strictly increasing'):
        result(history, decision)


def test_rejects_missing_columns():
    history, decision = fixture_history()
    with pytest.raises(ValueError, match='Missing history fields'):
        result(history.drop(columns='net_cash_collected'), decision)


def test_rejects_insufficient_training_months():
    history, decision = fixture_history(n=22)
    with pytest.raises(ValueError, match='Not enough'):
        result(history, decision)


def test_rejects_incorrect_holdout_count():
    history, decision = fixture_history()
    decision.loc[0, 'reserved_backtest_months'] = 5
    with pytest.raises(ValueError, match='Holdout size'):
        result(history, decision)


def test_rejects_incorrect_training_count():
    history, decision = fixture_history()
    decision.loc[0, 'training_months_available'] = 22
    with pytest.raises(ValueError, match='training-month count'):
        result(history, decision)


def test_rejects_nonfinite_cash():
    history, decision = fixture_history()
    history.loc[0, 'net_cash_collected'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        result(history, decision)


def test_reports_do_not_include_any_holdout_actuals():
    outputs = result()
    assert 'actual_net_cash_collected' not in outputs['forecast_model_candidates']
    assert 'holdout_cash' not in outputs['forecast_model_candidates']
    assert not bool(outputs['forecast_model_decision'].iloc[0].approved_for_live_forecasting)


def test_write_three_reports(tmp_path):
    history, decision = fixture_history()
    outputs = run_forecast_candidates(history, decision, output_dir=tmp_path)
    assert len(list(tmp_path.glob('*.csv'))) == 3
    for name, original in outputs.items():
        loaded = pd.read_csv(tmp_path / f'{name}.csv')
        assert len(loaded) == len(original)


def test_variable_horizon_within_holdout():
    outputs = result(max_horizon=2)
    assert len(outputs['forecast_model_candidates']) == 6
    assert outputs['forecast_model_candidates'].horizon_months.max() == 2


def test_rejects_horizon_larger_than_holdout():
    with pytest.raises(ValueError, match='cannot exceed'):
        result(max_horizon=7)


def test_rejects_bad_month_format():
    history, decision = fixture_history()
    history.loc[3, 'payment_month'] = '2024/05'
    with pytest.raises(ValueError, match='YYYY-MM'):
        result(history, decision)
