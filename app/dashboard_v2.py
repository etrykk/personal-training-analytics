#Author: etrykk
#Date: 11/21/2026
#Personal Training Analytics Platform
#V2 - Studio-style dashboard (separate from the working V1 dashboard)

import sys
from html import escape
from pathlib import Path
from io import BytesIO
import tempfile
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))  #V1 CSV importer uses top-level src imports

from src.analytics.business_health import run_business_health, dashboard_monthly_trends
from src.analytics.client_insights import run_client_insights
from src.analytics.financial_membership_insights import run_financial_membership_insights
from src.analytics.session_trainer_insights import run_session_trainer_insights
from src.dashboard_views import (filter_monthly_trends, overview_kpis,
                                 sorted_business_insights, search_workspace,
                                 readable_chart_label)
from src.dashboard_data import (ANALYTICS_COLUMNS, prepare_dashboard_data,
                                import_signature, trainer_name_mapping, labeled_trainers)
from src.data_input.column_mapper import suggest_column_mapping
from src.data_input.import_pipeline import import_datasets, DATASET_NAMES
from src.data_schema import REQUIRED_COLUMNS
from src.reporting import build_reports, export_reports
from src.trainer_analysis import (trainer_activity, trainer_monthly_activity,
                                  trainer_direct_session_cash)

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

st.set_page_config(page_title='PT Analytics Studio | V2', page_icon='📊',
                   layout='wide', initial_sidebar_state='expanded')


@st.cache_data(show_spinner=False)
def load_processed_data():
    #Reuse V1's processed CSVs; never change or overwrite them.
    names = DATASET_NAMES
    return {name: pd.read_csv(DATA_DIR / f'{name}.csv') for name in names}


def generate_reports(data, as_of):
    #All analytics receive the selected source. Imported data is never globally cached.
    health = run_business_health(data['clients'], data['memberships'],
                                 data['sessions'], data['payments'], as_of)
    client = run_client_insights(data['clients'], data['sessions'], as_of, months=12)
    finance = run_financial_membership_insights(data['payments'], data['memberships'], as_of)
    #The overview defaults to 6 months; prepare up to 12 for the date-range control.
    health['dashboard_monthly_trends'] = dashboard_monthly_trends(
        client['monthly_client_activity'], finance['monthly_cash_performance'])
    return {'health': health, 'clients': client, 'finance': finance,
            'sessions': run_session_trainer_insights(data['sessions'], as_of)}


@st.cache_data(show_spinner='Calculating sample business insights...')
def load_reports(as_of):
    #Only the public/sample data is cached across Streamlit sessions.
    return generate_reports(load_processed_data(), as_of)


def money(value):
    return f'${value:,.0f}' if pd.notna(value) else 'N/A'


def percent(value):
    return f'{value:.1f}%' if pd.notna(value) else 'N/A'


def make_chart(fig, dark, height=330):
    #Only change chart-facing labels: underlying DataFrames and CSV exports stay intact.
    #Plotly Express produces names like completed_sessions and trainer_id by default.
    for trace in fig.data:
        if getattr(trace, 'name', None):
            trace.name = readable_chart_label(trace.name)
        if isinstance(getattr(trace, 'hovertemplate', None), str):
            trace.hovertemplate = trace.hovertemplate.replace('_', ' ')
    fig.update_layout(
        template='plotly_dark' if dark else 'plotly_white',
        height=height, margin=dict(l=12, r=12, t=15, b=22),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12), showlegend=True,
        legend=dict(orientation='h', y=1.12, x=0, title_text=''),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor='rgba(150,150,150,.15)', title=None, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


def csv_download(frame, filename):
    if not frame.empty:
        st.download_button('Download CSV', frame.to_csv(index=False).encode('utf-8'),
                           file_name=filename, mime='text/csv')


def cash_trend(trends, dark, months):
    view = filter_monthly_trends(trends, months=months)
    if view.empty:
        st.info('No monthly cash data is available.')
        return
    view['period_label'] = view['partial_month'].map({True: 'Partial month', False: 'Completed month'})
    fig = px.bar(view, x='month', y='net_cash_collected', color='period_label',
                 color_discrete_map={'Completed month': '#22A18C', 'Partial month': '#93A5C2'},
                 hover_data={'net_cash_collected': ':,.2f', 'month': '|%b %Y',
                             'period_label': True})
    fig.update_yaxes(tickprefix='$')
    make_chart(fig, dark)
    st.caption('Net cash collected = completed payments minus completed refunds. Partial months are shown separately; this is not profit.')


def attendance_trend(trends, dark, months):
    view = filter_monthly_trends(trends, months=months)
    if view.empty:
        st.info('No monthly activity data is available.')
        return
    fig = px.line(view, x='month', y='completed_sessions', markers=True)
    fig.update_traces(line=dict(color='#22A18C', width=3), marker=dict(size=8))
    make_chart(fig, dark)
    if view['partial_month'].any():
        st.caption('The latest month may contain only activity through the selected report date.')


def inject_styles(dark):
    #Studio-inspired neutral panels, compact navigation, and a safe top inset.
    background = '#0F0F0F' if dark else '#F7F7F7'
    surface = '#181818' if dark else '#FFFFFF'
    border = '#343434' if dark else '#E3E3E3'
    text = '#F1F1F1' if dark else '#171717'
    muted = '#AAAAAA' if dark else '#666666'
    sidebar = '#101010' if dark else '#FFFFFF'
    selected = '#2B2B2B' if dark else '#E9E9E9'
    hover = '#242424' if dark else '#EFEFEF'
    st.markdown(f'''<style>
        .stApp {{ background: {background}; color: {text}; }}
        header[data-testid="stHeader"] {{ background: {background}; border-bottom: 1px solid {border}; }}
        [data-testid="stMainBlockContainer"],
        .main .block-container {{
            max-width: none !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            padding-top: 5.1rem !important;
            padding-bottom: 3rem !important;
            padding-left: clamp(1.25rem, 2.1vw, 2.5rem) !important;
            padding-right: clamp(1.25rem, 2.1vw, 2.5rem) !important;
            overflow: visible !important;
        }}
        [data-testid="stSidebar"] {{ background: {sidebar}; border-right: 1px solid {border}; }}
        [data-testid="stSidebarContent"] {{ padding-top: 1.6rem; }}
        /* Keep navigation against the left side, like YouTube Studio. */
        [data-testid="stSidebarUserContent"] {{
            padding-left: .65rem !important;
            padding-right: .65rem !important;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button,
        [data-testid="stSidebar"] button[data-testid^="stBaseButton"] {{
            width: 100%; display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            text-align: left !important;
            gap: .65rem;
            border-radius: 9px; padding-left: .78rem !important;
            min-height: 2.35rem; font-weight: 550; box-shadow: none;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button p {{
            text-align: left !important;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] {{
            justify-content: flex-start !important;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {{
            color: {text}; background: transparent; border: 1px solid transparent;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {{
            background: {hover}; border-color: transparent;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
            color: {text}; background: {selected}; border: 1px solid {selected};
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {{
            background: {selected}; color: {text}; border-color: {selected};
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {border}; border-radius: 16px; }}
        [data-testid="stMetric"] {{padding: .28rem .25rem;}}
        [data-testid="stMetricLabel"] {{ color: {muted}; }}
        [data-testid="stMetricValue"] {{ font-weight: 720; letter-spacing: -.035em; }}
        .pt-brand {{ color: {text}; font-size: 1.24rem; font-weight: 750;
            letter-spacing: -.04em; margin: .6rem .78rem .28rem;
            text-align: left; white-space: nowrap; }}
        .pt-brand small {{ display:block; margin-top:.26rem; font-size:.65rem;
            letter-spacing:.13em; font-weight:600; color:{muted}; }}
        .studio-topbar {{ font-size:.88rem; font-weight: 700; letter-spacing: -.02em;
            color: {text}; padding: .55rem 0 .15rem; }}
        .studio-date {{ display:inline-block; border:1px solid {border};
            color: {muted}; border-radius: 25px; padding:.45rem .8rem;
            font-size:.78rem; margin-top:.15rem; white-space: nowrap; }}
        .studio-card {{ background:{surface}; border:1px solid {border}; border-radius:16px;
            padding: 1.3rem 1.45rem; min-height: 302px;
            color:{text}; line-height:1.4; }}
        .studio-card-title {{ font-weight:720; letter-spacing:-.025em;
            font-size:1.04rem; margin:0 0 .75rem; }}
        .studio-card-label {{ font-size:.79rem; color:{muted}; margin:.25rem 0 .2rem; }}
        .studio-card-number {{font-size:2.25rem; letter-spacing:-.055em;
            font-weight:760; margin:.05rem 0 .2rem; line-height:1.22; }}
        .studio-card-accent {{ color: #4BC5A2; font-size:.79rem; font-weight:620; }}
        .studio-card-down {{ color: #E19A9A; font-size:.79rem; font-weight:620; }}
        .studio-card-info {{color:{muted}; font-size:.79rem;}}
        .studio-card-rule {{border-top:1px solid {border}; margin:1rem 0;}}
        .studio-card-row {{ display:flex; align-items:baseline;
            justify-content:space-between; gap: .5rem; margin:.52rem 0; font-size:.82rem; }}
        .studio-card-row span:first-child {{color:{muted};}}
        .studio-card-row strong {{font-size:.94rem; text-align:right;}}
        .studio-small-note {{ color:{muted}; font-size:.75rem; margin-top:1rem; }}
        h1,h2,h3 {{ letter-spacing:-.04em; color:{text}; }}
        h1 {{padding-top: .25rem !important;}}
        [data-testid="stPlotlyChart"] {{margin-bottom: .4rem;}}
        @media(max-width:900px) {{
            .studio-card {{min-height: unset;}}
            [data-testid="stMainBlockContainer"], .main .block-container {{padding-top:4rem!important;}}
        }}
    </style>''', unsafe_allow_html=True)

def overview_page(reports, dark, months):
    #A three-panel summary, inspired by Studio's channel dashboard cards.
    health = reports['health']
    kpis = overview_kpis(health['business_overview'])
    insights = sorted_business_insights(health['business_health_insights'])

    st.title('Business dashboard')
    st.caption('Performance, client activity, and business indicators at a glance')
    left, middle, right = st.columns([1, 1, 1], gap='medium')

    growth = kpis['last_completed_month_growth_pct']
    growth_display = (f'<span class="studio-card-accent">↑ {percent(growth)} vs. previous month</span>'
                      if pd.notna(growth) and growth >= 0
                      else f'<span class="studio-card-down">↓ {percent(abs(growth))} vs. previous month</span>'
                      if pd.notna(growth) else '<span class="studio-card-info">No previous month comparison</span>')
    last_month = escape(str(kpis['last_completed_month']))
    with left:
        st.markdown(f'''<div class="studio-card">
            <div class="studio-card-title">Cash performance</div>
            <div class="studio-card-label">Last completed month · {last_month}</div>
            <div class="studio-card-number">{money(kpis['last_completed_month_net_cash'])}</div>
            {growth_display}
            <div class="studio-card-rule"></div>
            <div class="studio-card-row"><span>Current month to date</span>
              <strong>{money(kpis['current_month_net_cash'])}</strong></div>
            <div class="studio-small-note">Current month is {"partial" if kpis['current_month_partial'] else "complete"}. Cash is not profit.</div>
        </div>''', unsafe_allow_html=True)

    with middle:
        st.markdown(f'''<div class="studio-card">
            <div class="studio-card-title">Training analytics</div>
            <div class="studio-card-label">Completed sessions · last 30 days</div>
            <div class="studio-card-number">{int(kpis['completed_sessions_30d']):,}</div>
            <div class="studio-card-info">Attendance and delivery summary</div>
            <div class="studio-card-rule"></div>
            <div class="studio-card-row"><span>Clients attending</span><strong>{int(kpis['clients_attending_30d']):,}</strong></div>
            <div class="studio-card-row"><span>Completion rate</span><strong>{percent(kpis['session_completion_rate_30d_pct'])}</strong></div>
        </div>''', unsafe_allow_html=True)

    with right:
        flag_count = int(kpis['clients_needing_followup'])
        top_insights = insights.head(2)
        lines = ''.join(
            f'<div class="studio-card-row"><span>{escape(str(r["category"]).title())}</span>'
            f'<strong>{escape(str(r["headline"]))}</strong></div>'
            for _, r in top_insights.iterrows())
        st.markdown(f'''<div class="studio-card">
            <div class="studio-card-title">Business insights</div>
            <div class="studio-card-label">Clients flagged for engagement review</div>
            <div class="studio-card-number">{flag_count:,}</div>
            <div class="studio-card-info">Descriptive indicators, not churn predictions</div>
            <div class="studio-card-rule"></div>
            {lines or '<div class="studio-card-info">No business indicators currently available.</div>'}
        </div>''', unsafe_allow_html=True)

    st.write('')
    charts_left, charts_right = st.columns(2, gap='medium')
    with charts_left:
        with st.container(border=True):
            st.subheader('Cash collection')
            cash_trend(health['dashboard_monthly_trends'], dark, months)
    with charts_right:
        with st.container(border=True):
            st.subheader('Completed training sessions')
            attendance_trend(health['dashboard_monthly_trends'], dark, months)

    st.write('')
    if st.button('View all business indicators', icon=':material/insights:',
                 key='dashboard_to_indicators'):
        st.session_state['studio_section'] = 'Business Indicators'
        st.rerun()


def business_indicators_page(reports):
    #Detailed indicators live on their own page; the dashboard retains a short preview.
    st.title('Business indicators')
    st.caption('Observed client engagement, payment and operational trends · Not churn predictions')
    indicators = sorted_business_insights(reports['health']['business_health_insights'])
    health = overview_kpis(reports['health']['business_overview'])
    first, second, third = st.columns(3)
    first.metric('Clients for engagement review', f"{int(health['clients_needing_followup']):,}")
    second.metric('Business indicators', len(indicators))
    third.metric('Priority reviews', int(indicators['priority'].eq('review').sum()))
    st.divider()
    if indicators.empty:
        st.info('No business indicators are available for this reporting period.')
        return
    selected_priority = st.selectbox('Filter by priority',
                                     ['All priorities', 'Review', 'Monitor', 'Info'])
    if selected_priority != 'All priorities':
        indicators = indicators.loc[indicators['priority'].str.casefold()
                                    == selected_priority.casefold()]
    if indicators.empty:
        st.info('No indicators match the selected priority.')
        return
    for _, indicator in indicators.iterrows():
        with st.container(border=True):
            st.markdown(f"**{indicator['headline']}**")
            st.caption(f"{readable_chart_label(str(indicator['category']).title())} · "
                       f"{str(indicator['priority']).title()} priority")
            st.write(str(indicator['detail']))
    st.caption('These indicators are based on recorded data and are not predictive model outputs.')
    csv_download(indicators, 'v2_business_indicators.csv')


def revenue_page(reports, dark, months):
    st.title('Revenue analytics')
    st.caption('Cash collected is not profit or earned revenue. Incomplete months are labeled.')
    health = reports['health']['business_overview'].iloc[0]
    finance = reports['finance']
    a, b, c = st.columns(3)
    a.metric('Last completed month cash', money(health['last_completed_month_net_cash']))
    b.metric('Current month cash to date', money(health['current_month_net_cash']))
    c.metric('Failed attempts · 90d', f"{int(health['failed_payment_attempts_90d']):,}")
    st.subheader('Monthly net cash')
    cash_trend(reports['health']['dashboard_monthly_trends'], dark, months)
    st.subheader('Payment attempts · last 90 days')
    st.dataframe(finance['payment_status_90d'], use_container_width=True, hide_index=True)
    csv_download(finance['monthly_cash_performance'], 'v2_monthly_cash.csv')


def clients_page(reports, dark, months):
    st.title('Client analytics')
    st.caption('Attendance and engagement indicators; not cancellation predictions.')
    kpis = overview_kpis(reports['health']['business_overview'])
    a, b, c = st.columns(3)
    a.metric('Clients on record', f"{int(kpis['total_clients']):,}")
    b.metric('Attended · last 30d', f"{int(kpis['clients_attending_30d']):,}")
    c.metric('Flagged for engagement review', f"{int(kpis['clients_needing_followup']):,}")
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Engagement categories')
        engagement = reports['clients']['client_engagement_status']
        counts = engagement['engagement_status'].value_counts().rename_axis('category').reset_index(name='clients')
        counts['category_display'] = counts['category'].map(readable_chart_label)
        fig = px.bar(counts, x='clients', y='category_display', orientation='h',
                     labels={'category_display': 'Engagement category', 'clients': 'Clients'},
                     color_discrete_sequence=['#22A18C'])
        make_chart(fig, dark)
    with right:
        st.subheader('Monthly client attendance')
        monthly = reports['clients']['monthly_client_activity'].copy()
        if not monthly.empty:
            monthly['month'] = pd.to_datetime(monthly['month'].astype(str), format='%Y-%m')
            fig = px.line(monthly.tail(months), x='month', y='clients_with_completed_sessions', markers=True)
            fig.update_traces(line=dict(color='#22A18C', width=3))
            make_chart(fig, dark)
        else:
            st.info('No client attendance records in this window.')
    st.subheader('Client activity details')
    st.dataframe(reports['clients']['client_engagement_status'],
                 use_container_width=True, hide_index=True)
    csv_download(reports['clients']['client_engagement_status'], 'v2_client_engagement.csv')


def sessions_page(reports, dark):
    st.title('Session analytics')
    st.caption('Completed sessions, cancellations, no-shows and weekday patterns.')
    health = reports['health']['business_overview'].iloc[0]
    a, b, c = st.columns(3)
    a.metric('Completed sessions · 30d', f"{int(health['completed_sessions_30d']):,}")
    b.metric('Session completion · 30d', percent(health['session_completion_rate_30d_pct']))
    c.metric('No-show rate · 30d', percent(health['no_show_rate_30d_pct']))
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Weekly session volume')
        weekly = reports['sessions']['weekly_session_trends'].copy()
        if not weekly.empty:
            weekly['week_start'] = pd.to_datetime(weekly['week_start'])
            fig = px.bar(weekly, x='week_start', y=['completed_sessions', 'canceled_sessions', 'no_shows'],
                         color_discrete_sequence=['#22A18C', '#93A5C2', '#D8A465'])
            make_chart(fig, dark)
    with right:
        st.subheader('Weekday completion rate')
        weekday = reports['sessions']['weekday_session_patterns']
        fig = px.bar(weekday, x='weekday', y='completion_rate_pct',
                     color_discrete_sequence=['#22A18C'])
        make_chart(fig, dark)
    st.subheader('Recent vs. previous 30 days')
    st.dataframe(reports['sessions']['session_period_comparison'], use_container_width=True, hide_index=True)
    csv_download(reports['sessions']['weekly_session_trends'], 'v2_weekly_sessions.csv')


def membership_page(reports):
    st.title('Membership analytics')
    st.caption('Membership coverage and contract retention are not overall client retention.')
    coverage = reports['finance']['membership_coverage_snapshot'].iloc[0]
    retention = reports['finance']['membership_retention_30d'].iloc[0]
    a, b, c = st.columns(3)
    a.metric('Active membership clients', int(coverage['active_clients']))
    b.metric('Membership retention · 30d', percent(retention['retention_pct']))
    c.metric('Expiring in next 30d', int(coverage['expiring_clients_30d']))
    st.subheader('Upcoming expirations')
    watchlist = reports['finance']['membership_expiry_watchlist_30d']
    if watchlist.empty:
        st.success('No matching memberships due to expire in the next 30 days.')
    else:
        st.dataframe(watchlist, use_container_width=True, hide_index=True)
        csv_download(watchlist, 'v2_membership_expirations.csv')
    st.subheader('30-day membership retention details')
    st.dataframe(reports['finance']['membership_retention_30d'], use_container_width=True, hide_index=True)


def trainers_page(reports, dark, labels, datasets, report_date):
    st.title('Trainer analytics')
    st.caption('Select trainers by display name. Trainer IDs remain unchanged in the data.')
    trainer_ids = sorted(datasets['sessions']['trainer_id'].dropna().astype(str).unique())
    selected = st.multiselect('Trainer names', trainer_ids, default=trainer_ids,
                              format_func=lambda identifier: labels.get(identifier, identifier))
    if not selected:
        st.info('Select at least one trainer to display performance metrics.')
        return

    recent, all_time = st.tabs(['Recent 30-day performance', 'All-time performance (V1)'])
    with recent:
        trainers = reports['sessions']['trainer_period_comparison'].copy()
        trainers = trainers.loc[trainers['trainer_id'].astype(str).isin(selected)]
        trainers = labeled_trainers(trainers, labels)
        if trainers.empty:
            st.info('No trainer activity is available for this reporting date.')
        else:
            a, b, c = st.columns(3)
            a.metric('Trainers selected', len(selected))
            b.metric('Completed sessions · 30d', int(trainers['recent_completed'].sum()))
            c.metric('Training hours · 30d', f"{trainers['recent_training_hours'].sum():,.1f}")
            st.subheader('Completed sessions by trainer · 30 days')
            fig = px.bar(trainers.sort_values('recent_completed', ascending=False),
                         x='trainer_name', y='recent_completed', hover_data=['trainer_id'],
                         color_discrete_sequence=['#22A18C'])
            make_chart(fig, dark)
            st.dataframe(trainers, use_container_width=True, hide_index=True)
            csv_download(trainers, 'v2_trainer_performance.csv')

    with all_time:
        st.caption('V1-style trainer analytics for all sessions through the reporting date. '
                   'Cash includes only attributable pay-per-session payments, not packages or memberships.')
        cutoff = pd.Timestamp(report_date) + pd.Timedelta(days=1)
        past_sessions = datasets['sessions'].copy()
        past_sessions['session_date'] = pd.to_datetime(past_sessions['session_date'], errors='raise')
        past_sessions['duration_minutes'] = pd.to_numeric(past_sessions['duration_minutes'], errors='raise')
        past_sessions = past_sessions.loc[past_sessions['session_date'] < cutoff]
        selection = past_sessions.loc[past_sessions['trainer_id'].astype(str).isin(selected)]
        activity = trainer_activity(selection)
        monthly = trainer_monthly_activity(selection)
        past_payments = datasets['payments'].copy()
        past_payments['payment_date'] = pd.to_datetime(past_payments['payment_date'], errors='raise')
        past_payments = past_payments.loc[past_payments['payment_date'] < cutoff]
        try:
            cash = trainer_direct_session_cash(datasets['sessions'], past_payments)
            cash = cash.loc[cash['trainer_id'].astype(str).isin(selected)]
        except ValueError as error:
            st.warning(f'Direct session cash is unavailable: {error}')
            cash = pd.DataFrame(columns=['trainer_id', 'attributable_net_session_cash'])

        a, b, c, d = st.columns(4)
        a.metric('Selected trainers', len(selected))
        b.metric('Completed sessions', int(activity['completed_sessions'].sum()) if not activity.empty else 0)
        c.metric('Completed training hours', f"{activity['completed_training_hours'].sum():,.1f}" if not activity.empty else '0.0')
        d.metric('Pay-per-session cash', money(cash['attributable_net_session_cash'].sum()) if not cash.empty else '$0')
        if not activity.empty:
            activity = labeled_trainers(activity, labels)
            st.subheader('Completed hours by trainer')
            fig = px.bar(activity, x='trainer_name', y='completed_training_hours',
                         hover_data=['trainer_id'], color_discrete_sequence=['#22A18C'])
            make_chart(fig, dark)
            with st.expander('Trainer activity details'):
                st.dataframe(activity, use_container_width=True, hide_index=True)
                csv_download(activity, 'v2_trainer_all_time.csv')
        if not monthly.empty:
            monthly = labeled_trainers(monthly, labels)
            monthly['month'] = monthly['session_month'].dt.to_timestamp()
            st.subheader('Monthly completed sessions by trainer')
            fig = px.line(monthly, x='month', y='completed_sessions', color='trainer_name',
                          markers=True, hover_data=['trainer_id'])
            make_chart(fig, dark)
        if not cash.empty:
            cash = labeled_trainers(cash, labels)
            st.subheader('Attributable pay-per-session cash')
            fig = px.bar(cash, x='trainer_name', y='attributable_net_session_cash',
                         hover_data=['trainer_id'], color_discrete_sequence=['#22A18C'])
            fig.update_yaxes(tickprefix='$')
            make_chart(fig, dark)
            csv_download(cash, 'v2_trainer_cash.csv')


def experimental_page():
    st.title('Predictive lab')
    st.caption('Experimental · Not part of the primary business overview')
    st.info('Churn prediction and revenue forecasting have not been implemented in V2 yet.')
    st.write('Models will only be presented after adequate historical data, baseline comparisons, '
             'and appropriate validation. No client is assigned a machine-learning churn risk here.')


def search_page(reports, query, labels):
    st.title('Search results')
    st.caption(f'Matches for “{query.strip()}” in client IDs, trainer IDs, and business insights')
    matches = search_workspace(reports, query, trainer_labels=labels)
    found = sum(len(frame) for frame in matches.values())
    if found == 0:
        st.info('No matches were found. Try a different client ID, trainer ID, or insight keyword.')
        return
    for title, key in [('Clients', 'clients'), ('Trainers', 'trainers'), ('Business insights', 'insights')]:
        frame = matches[key]
        if key == 'trainers' and not frame.empty:
            frame = labeled_trainers(frame, labels)
        if not frame.empty:
            with st.container(border=True):
                st.subheader(f'{title} · {len(frame)} match(es)')
                st.dataframe(frame, use_container_width=True, hide_index=True)


def data_import_page():
    #Bring V1's memory-only five-file import, mapping and validation to V2.
    st.title('Data import')
    st.caption('Upload your five business CSVs, review the column mapping, then validate the data.')
    st.info('Imported files stay in this Streamlit session. The project sample CSVs are never overwritten.')

    uploads, manual = {}, {}
    for name in DATASET_NAMES:
        with st.expander(f'{name.title()} CSV', expanded=(name == 'clients')):
            uploaded = st.file_uploader(f'Upload {name.title()}', type=['csv'],
                                        key=f'v2_upload_{name}')
            if uploaded is None:
                st.caption('No file selected.')
                continue
            uploads[name] = uploaded.getvalue()
            try:
                preview = pd.read_csv(BytesIO(uploads[name]))
                if preview.empty:
                    st.error('This CSV has no data rows.')
                    continue
                st.caption(f'{len(preview):,} rows · {len(preview.columns)} columns')
                st.dataframe(preview.head(5), use_container_width=True, hide_index=True)
                try:
                    suggested = suggest_column_mapping(preview)
                except ValueError as error:
                    suggested = {}
                    st.warning(f'Automatic mapping needs attention: {error}')
                st.markdown('**Column mapping**')
                st.caption('Match each required platform field to an uploaded CSV column.')
                targets = sorted(ANALYTICS_COLUMNS[name] | set(REQUIRED_COLUMNS[name]))
                options = ['— Not mapped —'] + list(preview.columns)
                selected_sources = {}
                for target in targets:
                    guessed = next((source for source, mapped in suggested.items()
                                    if mapped == target), None)
                    selected = st.selectbox(target.replace('_', ' ').title(), options,
                        index=options.index(guessed) if guessed in options else 0,
                        key=f'v2_map_{name}_{target}')
                    if selected != '— Not mapped —':
                        selected_sources[selected] = target
                manual[name] = selected_sources
            except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as error:
                st.error(f'Cannot preview this CSV: {error}')

    current_signature = import_signature(uploads, manual)
    st.divider()
    left, right = st.columns([1, 2], vertical_alignment='center')
    with left:
        validate_clicked = st.button('Validate uploaded data', type='primary',
            use_container_width=True, disabled=len(uploads) != len(DATASET_NAMES))
    with right:
        st.caption(f'{len(uploads)} of {len(DATASET_NAMES)} required datasets supplied. '
                   'Current approved imports remain available until replaced or cleared.')

    if validate_clicked:
        try:
            result = import_datasets(uploads, manual_mappings=manual)
            if result['valid']:
                try:
                    prepared = prepare_dashboard_data(result['datasets'])
                except (ValueError, TypeError, KeyError) as error:
                    result['valid'] = False
                    result['dashboard_error'] = str(error)
                else:
                    st.session_state['v2_imported_data'] = prepared
                    st.session_state['v2_approved_signature'] = current_signature
            st.session_state['v2_last_import_result'] = result
        except (ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
            st.error(f'Import failed: {error}')

    result = st.session_state.get('v2_last_import_result')
    if result is not None:
        if result['valid']:
            st.success('Validation passed. Choose Imported Data in the sidebar to view its analytics.')
        else:
            st.error('This import needs corrections before it can replace the current data.')
            if result.get('dashboard_error'):
                st.warning(result['dashboard_error'])
        report = result['validation_report']
        if not report.empty:
            st.subheader('Validation findings')
            st.dataframe(report, use_container_width=True, hide_index=True)
            st.download_button('Download validation report', report.to_csv(index=False),
                               file_name='validation_report.csv', mime='text/csv')
        else:
            st.caption('No validation issues were found.')
        st.subheader('Column mapping results')
        for name, frame in result['mapping_reports'].items():
            with st.expander(f'{name.title()} mapping'):
                st.dataframe(frame, use_container_width=True, hide_index=True)
    if 'v2_imported_data' in st.session_state:
        st.divider()
        st.caption('An imported dataset is available for this session.')
        if st.button('Clear imported data', type='secondary'):
            for key in ('v2_imported_data', 'v2_approved_signature',
                        'v2_last_import_result', 'v2_report_downloads'):
                st.session_state.pop(key, None)
            #The source radio was already rendered; its value is reset on the next run.
            st.rerun()


def business_reports_page(datasets, report_date, source):
    #Reuse V1's report generation with the selected V2 data source.
    st.title('Business reports')
    st.caption('Create an HTML business report and downloadable CSV archive.')
    st.info('Client statuses and package balances are snapshots, not historical reconstructions.')
    st.write(f'**Data source:** {source} · **Report date:** {report_date}')
    signature = (source, str(report_date), st.session_state.get('v2_approved_signature'))
    if st.button('Generate business report', type='primary'):
        try:
            result = build_reports(datasets, report_date)
            with tempfile.TemporaryDirectory(prefix='pt_analytics_v2_') as folder:
                paths = export_reports(result, output_dir=folder)
                html_path = next(p for p in paths if p.name == 'business_report.html')
                html_bytes = html_path.read_bytes()
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for path in paths:
                        archive.write(path, arcname=str(path.relative_to(Path(folder))))
            st.session_state['v2_report_downloads'] = {
                'html': html_bytes, 'zip': buffer.getvalue(),
                'summary': result['summary'], 'signature': signature,
            }
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            st.session_state.pop('v2_report_downloads', None)
            st.error(f'Cannot generate business report: {error}')

    saved = st.session_state.get('v2_report_downloads')
    if saved is not None and saved['signature'] == signature:
        st.success('Business report generated.')
        first, second, third = st.columns(3)
        summary = saved['summary']
        first.metric('Net collected cash', money(summary['net_cash_collected']))
        second.metric('Clients joined', f"{summary['clients_joined']:,}")
        third.metric('Completed sessions', f"{summary['completed_sessions']:,}")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button('Download HTML report', saved['html'],
                file_name='business_report.html', mime='text/html', use_container_width=True)
        with col2:
            st.download_button('Download all reports (ZIP)', saved['zip'],
                file_name='personal_training_reports.zip', mime='application/zip',
                use_container_width=True)
    elif saved is not None:
        st.info('Data source or reporting date changed. Regenerate the report to refresh downloads.')


def main():
    if 'studio_section' not in st.session_state:
        st.session_state['studio_section'] = 'Dashboard'

    navigation = [
        ('Dashboard', ':material/dashboard:'),
        ('Clients', ':material/groups:'),
        ('Revenue', ':material/payments:'),
        ('Sessions', ':material/calendar_month:'),
        ('Memberships', ':material/card_membership:'),
        ('Trainers', ':material/fitness_center:'),
        ('Data Import', ':material/upload_file:'),
        ('Business Reports', ':material/description:'),
        ('Business Indicators', ':material/insights:'),
        ('Predictive Lab', ':material/science:'),
    ]
    with st.sidebar:
        st.markdown('<div class="pt-brand">PT Analytics Studio'
                    '<small>BUSINESS INTELLIGENCE · V2</small></div>',
                    unsafe_allow_html=True)
        st.write('')
        for name, icon in navigation:
            selected = st.session_state['studio_section'] == name
            if st.button(name, key=f'nav_{name.replace(" ", "_").lower()}',
                         icon=icon, type='primary' if selected else 'secondary',
                         use_container_width=True):
                st.session_state['studio_section'] = name
                st.rerun()
        st.divider()
        st.markdown('**Data source**')
        sources = ['Sample Data']
        if 'v2_imported_data' in st.session_state:
            sources.append('Imported Data')
        if st.session_state.get('v2_source') not in sources:
            st.session_state['v2_source'] = 'Sample Data'
        source = st.radio('Use dataset', sources, key='v2_source')
        st.caption('Imported CSVs are kept only in this Streamlit session.')
        st.divider()
        st.markdown('**Workspace settings**')
        theme = st.radio('Appearance', ['Dark', 'Light'], horizontal=True, key='studio_theme')
        selected_date = st.date_input('Report date', value=pd.Timestamp('2026-11-20').date())
        months = st.selectbox('Trend period', [3, 6, 12], index=1,
                              format_func=lambda x: f'{x} months')
        if st.button('Refresh processed data', use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    dark = theme == 'Dark'
    inject_styles(dark)
    section = st.session_state['studio_section']
    if section == 'Data Import':
        data_import_page()
        return

    try:
        data = (st.session_state['v2_imported_data'] if source == 'Imported Data'
                else load_processed_data())
    except (ValueError, TypeError, FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        st.error(f'Cannot load {source}: {error}')
        st.stop()

    #Saved labels are optional. Changes in the editor affect only this session.
    labels = trainer_name_mapping(data['sessions'],
                  names_file=PROJECT_ROOT / 'data' / 'trainer_names.csv',
                  edits=st.session_state.get(f'v2_trainer_edits_{source}', {}))
    with st.sidebar:
        with st.expander('Customize trainer names'):
            st.caption('Edit display names for this session. To save names permanently '
                       'with project sample data, add the exported CSV as data/trainer_names.csv.')
            edits_frame = pd.DataFrame({'Trainer ID': list(labels),
                                        'Trainer Name': list(labels.values())})
            updated = st.data_editor(edits_frame, hide_index=True,
                 disabled=['Trainer ID'], use_container_width=True,
                 key=f'v2_trainer_name_editor_{source}')
            updated_names = {}
            for _, row in updated.iterrows():
                label = str(row['Trainer Name']).strip()
                if label:
                    updated_names[str(row['Trainer ID'])] = label
            st.session_state[f'v2_trainer_edits_{source}'] = updated_names
            labels = trainer_name_mapping(data['sessions'],
                 names_file=PROJECT_ROOT / 'data' / 'trainer_names.csv',
                 edits=updated_names)
            export_names = pd.DataFrame({'trainer_id': list(labels),
                                         'trainer_name': list(labels.values())})
            st.download_button('Save trainer names CSV', export_names.to_csv(index=False),
                file_name='trainer_names.csv', mime='text/csv')

    #Reuse V1's HTML + CSV export without recomputing the V2 charts.
    if section == 'Business Reports':
        business_reports_page(data, selected_date, source)
        return

    top_left, top_center, top_right = st.columns([1.3, 2.4, 1.0], vertical_alignment='center')
    with top_left:
        st.markdown('<div class="studio-topbar">PT Analytics Studio</div>',
                    unsafe_allow_html=True)
    with top_center:
        query = st.text_input('Search workspace', placeholder='Search clients, trainers, insights',
                              label_visibility='collapsed', key='studio_search')
    with top_right:
        st.markdown(f'<div class="studio-date">As of {selected_date.isoformat()}</div>',
                    unsafe_allow_html=True)
    st.caption(f'Viewing: **{source}**')
    st.divider()

    try:
        #Do not share uploaded business data through Streamlit's global cache.
        reports = (generate_reports(data, selected_date.isoformat())
                   if source == 'Imported Data' else load_reports(selected_date.isoformat()))
    except (ValueError, KeyError, TypeError, IndexError, FileNotFoundError) as exc:
        st.error(f'Cannot generate reports for the selected date and data: {exc}')
        st.info('Review the uploaded CSVs in Data Import if you are using Imported Data.')
        st.stop()

    if query.strip():
        search_page(reports, query, labels)
    elif section == 'Dashboard':
        overview_page(reports, dark, months)
    elif section == 'Clients':
        clients_page(reports, dark, months)
    elif section == 'Revenue':
        revenue_page(reports, dark, months)
    elif section == 'Sessions':
        sessions_page(reports, dark)
    elif section == 'Memberships':
        membership_page(reports)
    elif section == 'Trainers':
        trainers_page(reports, dark, labels, data, selected_date)
    elif section == 'Business Indicators':
        business_indicators_page(reports)
    elif section == 'Predictive Lab':
        experimental_page()


if __name__ == '__main__':
    main()
