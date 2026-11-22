#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 Phase 5D - Historical revenue forecast reports for Predictive Lab

"""Read validated, aggregate historical forecast research; never create live forecasts.

The five inputs are Phase 5C generated reports, not uploaded client records.
The selected model is fixed by development validation, never by the holdout.
"""

from pathlib import Path
import math
import pandas as pd

MODELS = ('last_month', 'trailing_3_month_mean', 'linear_trend_last_6_months')
LABELS = {
    'last_month': 'Last-month baseline',
    'trailing_3_month_mean': 'Three-month average',
    'linear_trend_last_6_months': 'Six-month linear trend',
}
REQUIRED = {
    'decision': {'status', 'selected_model', 'selection_method', 'training_end_month',
                 'holdout_start_month', 'holdout_end_month', 'holdout_months',
                 'forecast_horizons', 'development_validation_mae', 'holdout_mae',
                 'holdout_rmse', 'holdout_last_month_mae', 'research_results_available',
                 'approved_for_live_forecasting'},
    'checks': {'check', 'passed'},
    'validation_scores': {'model', 'horizon_months', 'target_months', 'predictions',
                          'mae', 'rmse', 'selected_on_development', 'selection_rank'},
    'holdout_scores': {'model', 'horizon_months', 'target_months', 'predictions',
                       'mae', 'rmse', 'selected_on_development'},
    'predictions': {'split', 'model', 'origin_month', 'target_month', 'horizon_months',
                    'actual_net_cash', 'predicted_net_cash', 'experimental_only'},
}
FILES = {
    'decision': 'forecast_backtest_decision.csv',
    'checks': 'forecast_backtest_checks.csv',
    'validation_scores': 'forecast_validation_scores.csv',
    'holdout_scores': 'forecast_holdout_scores.csv',
    'predictions': 'forecast_backtest_predictions.csv',
}


def _true(value):
    return str(value).strip().lower() == 'true'


def _number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid nonnegative numeric report value')
    return value


def _month(value):
    text = str(value)
    if len(text) != 7 or text[4] != '-':
        raise ValueError('Expected YYYY-MM month')
    period = pd.Period(text, freq='M')
    if str(period) != text:
        raise ValueError('Noncanonical month')
    return period


def _single_model_rows(frame):
    if (len(frame) != len(MODELS) or frame['model'].duplicated().any()
            or set(frame['model']) != set(MODELS)):
        raise ValueError('Expected exactly one aggregate score per model')
    for field in ('mae', 'rmse'):
        for value in frame[field]:
            _number(value)
    return frame


def load_revenue_backtest_summary(report_dir):
    """Return research-only summary, or status missing/incomplete.

    No prediction values from future months or individual clients are exposed.
    Historical monthly points are restricted to one-month-ahead holdout rows.
    """
    folder = Path(report_dir)
    paths = {name: folder / filename for name, filename in FILES.items()}
    if not all(path.is_file() for path in paths.values()):
        return {'status': 'missing', 'live_forecasting_enabled': False}
    try:
        reports = {key: pd.read_csv(path) for key, path in paths.items()}
        if any(not REQUIRED[k].issubset(frame.columns) for k, frame in reports.items()):
            raise ValueError('Missing required report columns')
        dec, chk = reports['decision'], reports['checks']
        if len(dec) != 1 or chk.empty or not chk['passed'].map(_true).all():
            raise ValueError('Decision or checks invalid')
        d = dec.iloc[0]
        if (d['status'] != 'retrospective_backtest_complete'
                or not _true(d['research_results_available'])
                or _true(d['approved_for_live_forecasting'])
                or d['selected_model'] not in MODELS):
            raise ValueError('Research-only decision required')
        selected = str(d['selected_model'])
        end = _month(d['training_end_month'])
        first, last = _month(d['holdout_start_month']), _month(d['holdout_end_month'])
        months, horizons = int(d['holdout_months']), int(d['forecast_horizons'])
        if months < 1 or horizons != 3 or first != end + 1 or last != first + months - 1:
            raise ValueError('Invalid historical holdout boundaries')
        validation = _single_model_rows(reports['validation_scores'].copy())
        if (not validation['horizon_months'].astype(str).eq('all').all()
                or sorted(validation['selection_rank'].astype(int)) != list(range(1, len(MODELS)+1))):
            raise ValueError('Invalid development selection ranks')
        val_winner = validation.loc[validation['selection_rank'].astype(int).eq(1)]
        if (len(val_winner) != 1 or val_winner.iloc[0]['model'] != selected
                or validation['selected_on_development'].map(_true).sum() != 1
                or not _true(val_winner.iloc[0]['selected_on_development'])):
            raise ValueError('Selection does not match development validation')
        ranked = validation.sort_values(['mae', 'rmse', 'model']).iloc[0]
        if ranked['model'] != selected:
            raise ValueError('Selected model was not best on development MAE')
        holdout = reports['holdout_scores'].copy()
        holdout['horizon'] = holdout['horizon_months'].astype(str)
        wanted = {(m, h) for m in MODELS for h in ('all', '1', '2', '3')}
        if (len(holdout) != len(wanted)
                or holdout[['model','horizon']].duplicated().any()
                or set(zip(holdout['model'], holdout['horizon'])) != wanted):
            raise ValueError('Incomplete historical holdout score grid')
        for row in holdout.itertuples(index=False):
            _number(row.mae)
            _number(row.rmse)
            if (_true(row.selected_on_development) != (row.model == selected)
                    or int(row.target_months) != months
                    or int(row.predictions) != months * (horizons if row.horizon == 'all' else 1)):
                raise ValueError('Inconsistent historical score rows')
        all_scores = _single_model_rows(holdout.loc[holdout['horizon'].eq('all')].copy())
        chosen = all_scores.loc[all_scores['model'].eq(selected)].iloc[0]
        baseline = all_scores.loc[all_scores['model'].eq('last_month')].iloc[0]
        checks = (
            (d['holdout_mae'], chosen['mae']),
            (d['holdout_rmse'], chosen['rmse']),
            (d['holdout_last_month_mae'], baseline['mae']),
            (d['development_validation_mae'], val_winner.iloc[0]['mae']),
        )
        if any(abs(_number(a) - _number(b)) > .011 for a,b in checks):
            raise ValueError('Decision and scores disagree')
        predictions = reports['predictions']
        month_view = predictions.loc[
            (predictions['split'].eq('reserved_holdout'))
            & (predictions['model'].eq(selected))
            & (pd.to_numeric(predictions['horizon_months'], errors='coerce') == 1)
        ].copy()
        if len(month_view) != months or month_view['target_month'].duplicated().any():
            raise ValueError('Invalid historical display window')
        for row in month_view.itertuples(index=False):
            if (not _true(row.experimental_only)
                    or _month(row.target_month) != _month(row.origin_month) + 1
                    or not first <= _month(row.target_month) <= last):
                raise ValueError('Invalid or nonhistorical forecast point')
            _number(row.actual_net_cash)
            _number(row.predicted_net_cash)
        if set(month_view['target_month']) != {str(first + offset) for offset in range(months)}:
            raise ValueError('Holdout monthly series incomplete')
        month_view = (month_view[['target_month','actual_net_cash','predicted_net_cash']]
                      .sort_values('target_month').reset_index(drop=True))
        month_view.columns = ['Month','Actual net cash','Historical forecast']
        return {
            'status': 'historical_research',
            'live_forecasting_enabled': False,
            'selected_model': selected,
            'selected_label': LABELS[selected],
            'training_end_month': str(end),
            'holdout_start_month': str(first),
            'holdout_end_month': str(last),
            'holdout_months': months,
            'forecast_horizons': horizons,
            'development_mae': float(val_winner.iloc[0]['mae']),
            'holdout_mae': float(chosen['mae']),
            'holdout_rmse': float(chosen['rmse']),
            'baseline_mae': float(baseline['mae']),
            'validation_scores': validation,
            'holdout_scores': holdout,
            'historical_months': month_view,
        }
    except (OSError, ValueError, TypeError, KeyError, OverflowError, IndexError,
            pd.errors.ParserError):
        return {'status': 'incomplete', 'live_forecasting_enabled': False}


def revenue_comparison_table(summary, horizon='all'):
    """Display only historical aggregate scores, without future projections."""
    columns = ['Model', 'MAE', 'RMSE', 'Selected on development']
    if summary.get('status') != 'historical_research':
        return pd.DataFrame(columns=columns)
    horizon = str(horizon)
    if horizon not in {'all','1','2','3'}:
        raise ValueError('Horizon must be all, 1, 2, or 3')
    rows = summary['holdout_scores'].loc[summary['holdout_scores']['horizon'].eq(horizon)]
    result = pd.DataFrame([
        {'Model': LABELS[row['model']], 'MAE': float(row['mae']),
         'RMSE': float(row['rmse']), 'Selected on development': _true(row['selected_on_development'])}
        for _, row in rows.iterrows()
    ])
    return result.sort_values('Model').reset_index(drop=True)
