#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5C - Revenue Forecast Evaluation Tests

import numpy as np
import pandas as pd
import pytest

from src.ml.revenue_forecast_models import create_revenue_forecast_candidates
from src.ml.revenue_forecast_evaluation import (
    evaluate_revenue_forecasts, run_forecast_evaluation,
)


def inputs(values=None):
    months = pd.period_range('2024-02', periods=33, freq='M').astype(str)
    if values is None:
        values = [100 + 10*i for i in range(33)]
    history = pd.DataFrame({
        'payment_month': months,
        'net_cash_collected': values,
        'eligible_for_backtest': [True] * 33,
    })
    readiness = pd.DataFrame([{
        'ready_for_experimental_backtesting': True,
        'reserved_backtest_months': 6,
        'training_months_available': 27,
    }])
    prior = create_revenue_forecast_candidates(history, readiness)
    return history, readiness, prior['forecast_model_candidates'], prior['forecast_model_decision']


def evaluate(data=None, **kwargs):
    return evaluate_revenue_forecasts(*(data or inputs()), **kwargs)


def test_five_expected_reports():
    reports = evaluate()
    assert set(reports) == {
        'forecast_backtest_predictions', 'forecast_validation_scores',
        'forecast_holdout_scores', 'forecast_backtest_checks',
        'forecast_backtest_decision',
    }


def test_development_selection_never_depends_on_holdout_values():
    data = list(inputs())
    selected = evaluate(data)['forecast_backtest_decision'].iloc[0].selected_model
    dev_score = evaluate(data)['forecast_validation_scores']
    modified = data[0].copy()
    modified.loc[27:, 'net_cash_collected'] = np.arange(6) * 90000
    data[0] = modified
    new = evaluate(data)
    assert new['forecast_backtest_decision'].iloc[0].selected_model == selected
    pd.testing.assert_frame_equal(new['forecast_validation_scores'], dev_score)


def test_holdout_uses_six_actual_months():
    rows = evaluate()['forecast_backtest_predictions']
    holdout = rows.loc[rows['split'].eq('reserved_holdout')]
    assert holdout.target_month.nunique() == 6
    assert set(holdout.target_month) == set(pd.period_range('2026-05', periods=6, freq='M').astype(str))
    assert len(holdout) == 54


def test_internal_validation_uses_only_development_months():
    rows = evaluate()['forecast_backtest_predictions']
    dev = rows.query("split == 'development_validation'")
    assert len(dev) == 54
    assert dev.target_month.max() == '2026-04'
    assert dev.origin_month.max() < dev.target_month.max()


def test_predictions_never_read_current_or_future_months():
    rows = evaluate()['forecast_backtest_predictions']
    origins = pd.PeriodIndex(rows.origin_month, freq='M')
    targets = pd.PeriodIndex(rows.target_month, freq='M')
    assert (origins < targets).all()
    assert ((targets.astype(int) - origins.astype(int)) == rows.horizon_months.to_numpy()).all()


def test_linear_trend_predicts_linear_series_perfectly():
    rows = evaluate()['forecast_backtest_predictions']
    trend = rows.loc[rows.model.eq('linear_trend_last_6_months')]
    assert trend.absolute_error.max() == 0
    scores = evaluate()['forecast_validation_scores']
    trend_mae = scores.loc[scores.model.eq('linear_trend_last_6_months'), 'mae'].iloc[0]
    assert trend_mae == 0


def test_linear_trend_selected_for_linear_series():
    d = evaluate()['forecast_backtest_decision'].iloc[0]
    assert d.selected_model == 'linear_trend_last_6_months'
    assert d.holdout_mae == 0
    assert d.beats_last_month_on_holdout


def test_no_retroactive_retraining_on_target_month():
    data = list(inputs())
    rows = evaluate(data)['forecast_backtest_predictions']
    chosen = rows.query("split == 'reserved_holdout' and model == 'last_month' and target_month == '2026-05' and horizon_months == 1")
    assert chosen.predicted_net_cash.iloc[0] == data[0].net_cash_collected.iloc[26]
    assert chosen.actual_net_cash.iloc[0] == data[0].net_cash_collected.iloc[27]


def test_later_origins_may_use_previously_observed_holdout_values():
    data = list(inputs())
    baseline = evaluate(data)['forecast_backtest_predictions']
    modified = data[0].copy()
    modified.loc[27, 'net_cash_collected'] += 10000
    data[0] = modified
    revised = evaluate(data)['forecast_backtest_predictions']
    def value(frame, month):
        q = frame.query("split == 'reserved_holdout' and model == 'last_month' and target_month == @month and horizon_months == 1")
        return q.predicted_net_cash.iloc[0]
    assert value(baseline, '2026-05') == value(revised, '2026-05')
    assert value(revised, '2026-06') - value(baseline, '2026-06') == 10000


def test_accuracy_metrics_have_correct_formulas():
    reports = evaluate()
    rows = reports['forecast_backtest_predictions']
    choice = rows.query("split == 'reserved_holdout' and model == 'last_month'")
    actual_mae = np.abs(choice['predicted_net_cash'] - choice['actual_net_cash']).mean()
    actual_rmse = np.sqrt(((choice['predicted_net_cash'] - choice['actual_net_cash']) ** 2).mean())
    scores = reports['forecast_holdout_scores']
    row = scores.loc[scores.model.eq('last_month') & scores.horizon_months.eq('all')].iloc[0]
    assert row.mae == pytest.approx(actual_mae, abs=0.01)
    assert row.rmse == pytest.approx(actual_rmse, abs=0.01)


def test_all_forecast_horizons_reported():
    scores = evaluate()['forecast_holdout_scores']
    assert len(scores) == 12
    assert set(scores.horizon_months.astype(str)) == {'all', '1', '2', '3'}
    assert all(scores.loc[scores.horizon_months.eq('all'), 'target_months'].eq(6))


def test_live_forecasting_always_disabled():
    decision = evaluate()['forecast_backtest_decision'].iloc[0]
    assert not decision.approved_for_live_forecasting
    assert decision.research_results_available
    assert decision.status == 'retrospective_backtest_complete'


def test_csv_serialized_inputs_work(tmp_path):
    data = inputs()
    copies = []
    for i, frame in enumerate(data):
        path = tmp_path / f'file{i}.csv'
        frame.to_csv(path, index=False)
        copies.append(pd.read_csv(path))
    assert len(evaluate(copies)['forecast_holdout_scores']) == 12


def test_rejects_incorrect_candidates():
    data = list(inputs())
    data[2] = data[2].copy()
    data[2].loc[0, 'forecast_net_cash'] += 999
    with pytest.raises(ValueError, match='differ from training-only'):
        evaluate(data)


def test_rejects_duplicate_candidate_pairs():
    data = list(inputs())
    data[2] = data[2].copy()
    data[2].loc[1, 'horizon_months'] = 1
    with pytest.raises(ValueError, match='Duplicate candidate'):
        evaluate(data)


def test_rejects_wrong_model_split():
    data = list(inputs())
    data[3] = data[3].copy()
    data[3].loc[0, 'training_end_month'] = '2026-03'
    with pytest.raises(ValueError, match='disagrees'):
        evaluate(data)


def test_rejects_wrong_status():
    data = list(inputs())
    data[3] = data[3].copy()
    data[3].loc[0, 'status'] = 'not ready'
    with pytest.raises(ValueError, match='not approved'):
        evaluate(data)


def test_rejects_not_ready():
    data = list(inputs())
    data[1] = data[1].copy()
    data[1].loc[0, 'ready_for_experimental_backtesting'] = False
    with pytest.raises(ValueError, match='did not approve'):
        evaluate(data)


def test_rejects_bad_horizon():
    with pytest.raises(ValueError, match='Forecast horizon'):
        evaluate(max_horizon=7)


def test_rejects_too_long_internal_validation():
    with pytest.raises(ValueError, match='too short'):
        evaluate(validation_months=20)


def test_does_not_modify_source_frames():
    data = inputs()
    before = [frame.copy(deep=True) for frame in data]
    evaluate(data)
    for actual, original in zip(data, before):
        pd.testing.assert_frame_equal(actual, original)


def test_write_reports(tmp_path):
    reports = run_forecast_evaluation(*inputs(), output_dir=tmp_path)
    assert len(list(tmp_path.glob('*.csv'))) == 5
    for name, frame in reports.items():
        assert len(pd.read_csv(tmp_path / f'{name}.csv')) == len(frame)


def test_rejects_invalid_eligibility():
    data = list(inputs())
    modified = data[0].copy()
    modified['eligible_for_backtest'] = modified['eligible_for_backtest'].astype(object)
    modified.loc[4, 'eligible_for_backtest'] = 'unknown'
    data[0] = modified
    with pytest.raises(ValueError, match='True or False'):
        evaluate(data)


def test_rejects_missing_candidate_rows():
    data = list(inputs())
    data[2] = data[2].iloc[:-1].copy()
    with pytest.raises(ValueError, match='incorrect row count'):
        evaluate(data)
