#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5D - Revenue research dashboard tests

import ast
from pathlib import Path
import pandas as pd
import pytest

from src.ml.revenue_lab_reports import (
    MODELS, LABELS, load_revenue_backtest_summary, revenue_comparison_table,
)


def make_reports(tmp_path):
    #Small self-contained historical fixture: outputs are research-only.
    pd.DataFrame([{
        'status': 'retrospective_backtest_complete',
        'selected_model': 'last_month',
        'selection_method': 'Lowest development-validation MAE; RMSE tie-break',
        'training_end_month': '2026-04',
        'holdout_start_month': '2026-05',
        'holdout_end_month': '2026-10',
        'holdout_months': 6, 'forecast_horizons': 3,
        'development_validation_mae': 100,
        'holdout_mae': 220, 'holdout_rmse': 300,
        'holdout_last_month_mae': 220,
        'research_results_available': True,
        'approved_for_live_forecasting': False,
    }]).to_csv(tmp_path / 'forecast_backtest_decision.csv', index=False)
    pd.DataFrame([{'check': 'historical integrity', 'passed': True}]).to_csv(
        tmp_path / 'forecast_backtest_checks.csv', index=False)
    pd.DataFrame([
        {'model': m, 'horizon_months': 'all', 'target_months': 6,
         'predictions': 18, 'mae': mae, 'rmse': rmse,
         'selected_on_development': m == 'last_month', 'selection_rank': i}
        for i,(m,mae,rmse) in enumerate(zip(MODELS,[100,120,130],[150,160,170]),1)
    ]).to_csv(tmp_path / 'forecast_validation_scores.csv', index=False)
    holdout = []
    for m,mae,rmse in zip(MODELS,[220,265,202],[300,350,260]):
        for h in ['all','1','2','3']:
            holdout.append({'model': m, 'horizon_months': h,
                            'target_months': 6, 'predictions': 18 if h=='all' else 6,
                            'mae': mae, 'rmse': rmse,
                            'selected_on_development': m=='last_month'})
    pd.DataFrame(holdout).to_csv(tmp_path / 'forecast_holdout_scores.csv', index=False)
    records = []
    for month in pd.period_range('2026-05','2026-10',freq='M'):
        records.append({'split': 'reserved_holdout', 'model': 'last_month',
                        'origin_month': str(month-1), 'target_month': str(month),
                        'horizon_months': 1, 'actual_net_cash': 1200,
                        'predicted_net_cash': 1000, 'experimental_only': True})
    pd.DataFrame(records).to_csv(tmp_path / 'forecast_backtest_predictions.csv', index=False)
    return tmp_path


def rewrite(path, filename, updater):
    file = path / filename
    data = pd.read_csv(file)
    updater(data)
    data.to_csv(file, index=False)


def test_missing_reports_are_not_displayed(tmp_path):
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'missing'


def test_valid_historical_report_is_loaded(tmp_path):
    data = load_revenue_backtest_summary(make_reports(tmp_path))
    assert data['status'] == 'historical_research'
    assert data['selected_model'] == 'last_month'
    assert data['holdout_mae'] == 220
    assert data['live_forecasting_enabled'] is False
    assert len(data['historical_months']) == 6


def test_live_forecasts_stay_disabled(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_decision.csv', lambda f: f.__setitem__('approved_for_live_forecasting', True))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_wrong_status_rejected(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_decision.csv',lambda f: f.__setitem__('status','production'))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_failed_quality_check_rejected(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_checks.csv',lambda f: f.__setitem__('passed',False))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_missing_column_rejected(tmp_path):
    make_reports(tmp_path)
    file = tmp_path/'forecast_holdout_scores.csv'
    pd.read_csv(file).drop(columns=['mae']).to_csv(file,index=False)
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_duplicate_validation_model_rejected(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_validation_scores.csv',lambda f: f.__setitem__('model',[MODELS[0]]*3))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_selection_must_follow_development_not_holdout(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_decision.csv',lambda f: f.__setitem__('selected_model','linear_trend_last_6_months'))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_development_rank_must_match_mae(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_validation_scores.csv',lambda f: f.__setitem__('mae',[180,120,130]))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_holdout_grid_must_be_complete(tmp_path):
    make_reports(tmp_path)
    path=tmp_path/'forecast_holdout_scores.csv'
    pd.read_csv(path).iloc[:-1].to_csv(path,index=False)
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_decision_must_match_scores(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_decision.csv',lambda f:f.__setitem__('holdout_mae',50))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_negative_error_rejected(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_holdout_scores.csv',lambda f: f.__setitem__('mae',[-1]+[220]*11))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_future_prediction_row_is_rejected(tmp_path):
    make_reports(tmp_path)
    rewrite(tmp_path,'forecast_backtest_predictions.csv',lambda f:f.__setitem__('origin_month',['2026-11']+list(f['origin_month'].iloc[1:])))
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_missing_holdout_month_is_rejected(tmp_path):
    make_reports(tmp_path)
    path=tmp_path/'forecast_backtest_predictions.csv'
    pd.read_csv(path).iloc[:-1].to_csv(path,index=False)
    assert load_revenue_backtest_summary(tmp_path)['status'] == 'incomplete'


def test_no_unverified_future_projection_is_exposed(tmp_path):
    data=load_revenue_backtest_summary(make_reports(tmp_path))
    assert data['live_forecasting_enabled'] is False
    assert set(data['historical_months'].columns)=={'Month','Actual net cash','Historical forecast'}
    assert not any('client' in x.lower() for x in data['historical_months'].columns)
    assert data['historical_months']['Month'].max()=='2026-10'


def test_comparison_table_is_aggregate_only(tmp_path):
    d=load_revenue_backtest_summary(make_reports(tmp_path))
    table = revenue_comparison_table(d,'1')
    assert len(table)==3
    assert set(table.columns)=={'Model','MAE','RMSE','Selected on development'}
    assert table.loc[table['Model'].eq(LABELS['last_month']),'Selected on development'].all()


def test_reject_invalid_horizon(tmp_path):
    d=load_revenue_backtest_summary(make_reports(tmp_path))
    with pytest.raises(ValueError):
        revenue_comparison_table(d,'14')


def test_incomplete_summary_produces_empty_comparison():
    assert revenue_comparison_table({'status':'incomplete'}).empty


def test_churn_and_revenue_both_visible_independently():
    path=Path(__file__).resolve().parents[1]/'app'/'dashboard_v2.py'
    content=path.read_text(encoding='utf-8')
    tree=ast.parse(content)
    page=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='experimental_page')
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(page)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert {'load_experimental_summary','research_comparison_table',
            'load_revenue_backtest_summary','revenue_comparison_table'}.issubset(calls)
    assert not calls.intersection({'fit','predict','predict_proba','read_csv','load_processed_data'})


def test_indicators_remain_above_predictive_lab():
    path=Path(__file__).resolve().parents[1]/'app'/'dashboard_v2.py'
    source=path.read_text(encoding='utf-8')
    assert source.index("('Business Indicators', ':material/insights:')") < source.index("('Predictive Lab', ':material/science:')")
