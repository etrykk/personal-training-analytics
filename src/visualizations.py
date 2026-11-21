#Author: etrykk
#Date: 11/20/2026
#Personal Training Analytics Platform
#Reusable Plotly Visualizations

import pandas as pd
import plotly.express as px

LABELS = {
    'per_session': 'Pay Per Session', 'no_show': 'No-Show',
    'completed': 'Completed', 'canceled': 'Canceled',
    'membership': 'Membership', 'package': 'Package',
    'session': 'Individual Session', 'active': 'Active',
    'inactive': 'Inactive', 'expired': 'Expired'
}


def pretty(value):
    return LABELS.get(str(value), str(value).replace('_', ' ').title())


def money(value):
    return f'${float(value):,.2f}'


def short_money(value):
    amount = abs(float(value))
    sign = '-' if float(value) < 0 else ''
    if amount >= 1_000_000:
        return f'{sign}${amount / 1_000_000:,.1f}M'
    if amount >= 1_000:
        return f'{sign}${amount / 1_000:,.1f}k'
    return f'{sign}${amount:,.0f}'


def format_chart(fig, currency=False, height=385):
    fig.update_layout(
        height=height,
        margin=dict(l=35, r=35, t=55, b=45),
        hovermode='closest', legend_title_text='',
        font=dict(size=13),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(font_size=13)
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.16)',
                     zeroline=False, automargin=True)
    if currency:
        fig.update_yaxes(tickprefix='$', tickformat='~s')
    return fig


def _plot_ready(data, x, y):
    missing = {x, y} - set(data.columns)
    if missing:
        raise ValueError(f'Chart missing columns: {sorted(missing)}')
    plot = data.copy()
    if isinstance(plot[x].dtype, pd.PeriodDtype):
        plot[x] = plot[x].dt.to_timestamp()
    return plot


def cash_chart(data, x, y, title='Cash Collected', kind='line', color=None):
    plot = _plot_ready(data, x, y)
    plot['Cash Label'] = plot[y].map(short_money)
    plot['Cash Detail'] = plot[y].map(money)
    if kind == 'bar':
        fig = px.bar(plot, x=x, y=y, color=color, title=title,
                     text='Cash Label', custom_data=['Cash Label', 'Cash Detail'])
        fig.update_traces(textposition='outside', cliponaxis=False)
    elif kind == 'line':
        fig = px.line(plot, x=x, y=y, color=color, title=title,
                      markers=True, custom_data=['Cash Label', 'Cash Detail'])
    else:
        raise ValueError("kind must be 'line' or 'bar'")
    fig.update_traces(hovertemplate='%{x}<br>Cash Collected = %{customdata[0]}'
                      '<br>Exact: %{customdata[1]}<extra></extra>')
    fig.update_layout(xaxis_title=None, yaxis_title='Cash Collected ($)')
    return format_chart(fig, currency=True)


def count_chart(data, x, y, title, kind='bar', color=None):
    plot = _plot_ready(data, x, y)
    if kind == 'bar':
        fig = px.bar(plot, x=x, y=y, color=color, title=title)
    elif kind == 'line':
        fig = px.line(plot, x=x, y=y, color=color, title=title, markers=True)
    else:
        raise ValueError("kind must be 'line' or 'bar'")
    fig.update_layout(xaxis_title=None, yaxis_title=pretty(y))
    fig.update_traces(hovertemplate='%{x}<br>' + pretty(y) + ': %{y:,.0f}<extra></extra>')
    return format_chart(fig)


def rate_chart(data, x, y, title, kind='line', color=None):
    plot = _plot_ready(data, x, y)
    if kind == 'line':
        fig = px.line(plot, x=x, y=y, color=color, title=title, markers=True)
    elif kind == 'bar':
        fig = px.bar(plot, x=x, y=y, color=color, title=title)
    else:
        raise ValueError("kind must be 'line' or 'bar'")
    fig.update_layout(xaxis_title=None, yaxis_title=pretty(y) + ' (%)')
    fig.update_traces(hovertemplate='%{x}<br>' + pretty(y) + ': %{y:.1f}%<extra></extra>')
    return format_chart(fig)


def save_chart(fig, file_path):
    from pathlib import Path
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path
