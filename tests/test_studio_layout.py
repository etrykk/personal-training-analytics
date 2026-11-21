#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 UI navigation and chart-label regression checks

import ast
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / 'app' / 'dashboard_v2.py'


def dashboard_source():
    return DASHBOARD.read_text(encoding='utf-8')


def test_business_indicators_navigation_is_before_predictive_lab():
    source = dashboard_source()
    assert "('Business Indicators', ':material/insights:')" in source
    assert source.index("('Business Indicators', ':material/insights:')") < source.index("('Predictive Lab', ':material/science:')")
    assert "elif section == 'Business Indicators':" in source
    assert "business_indicators_page(reports)" in source


def test_business_indicators_have_a_standalone_page():
    tree = ast.parse(dashboard_source())
    functions = {item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)}
    assert 'business_indicators_page' in functions
    overview = ast.get_source_segment(dashboard_source(), functions['overview_page'])
    indicators = ast.get_source_segment(dashboard_source(), functions['business_indicators_page'])
    assert 'View all business indicators' in overview
    assert 'Filter by priority' in indicators
    assert 'for _, indicator in indicators.iterrows()' in indicators
    assert 'for _, indicator in indicators.iterrows()' not in overview


def test_all_plotly_figures_are_formatted_by_shared_chart_helper():
    tree = ast.parse(dashboard_source())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_chart')
    body = ast.get_source_segment(dashboard_source(), function)
    assert 'readable_chart_label(trace.name)' in body
    assert "trace.hovertemplate.replace('_', ' ')" in body
    assert "title_text=''" in body


def test_sidebar_and_main_content_are_left_aligned():
    source = dashboard_source()
    assert 'justify-content: flex-start !important;' in source
    assert 'margin-left: 0 !important;' in source
    assert '[data-testid="stSidebarUserContent"]' in source
