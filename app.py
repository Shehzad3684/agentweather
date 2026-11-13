"""
AgentWeather  |  Autonomous Weather Intelligence Platform
Islamabad, Pakistan  |  Powered by LangGraph, Hopsworks, and Groq
"""

import os
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    DATA_DIR,
    EDA_ARTIFACTS_DIR,
    HAZARD_THRESHOLDS,
    TARGET_METRICS,
    FEATURE_GROUP_FORECAST_LEDGER,
    FEATURE_GROUP_AGENT_CALL_LOG,
    FEATURE_GROUP_DRIFT_LOG,
)
from src.features import ingest_live_row, build_engineered_features
from src.agents import run_agent_pipeline

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"AgentWeather | {CITY_NAME}",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Stylesheet ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root palette ── */
:root {
    --navy:        #0A1628;
    --navy-mid:    #1B3A6B;
    --navy-light:  #2C5F8A;
    --slate:       #3D5A80;
    --steel:       #98C1D9;
    --white:       #FFFFFF;
    --offwhite:    #F4F6FA;
    --muted:       #8A9BB0;
    --amber:       #C8922A;
    --amber-pale:  #FFF3CD;
    --danger:      #9B2335;
    --danger-pale: #FDEAEA;
    --safe:        #1A5C38;
    --safe-pale:   #EAF5EE;
    --border:      #DDE3EE;
    --shadow:      0 2px 12px rgba(10,22,40,0.08);
    --radius:      6px;
}

/* ── Global background & font ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--offwhite);
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    color: var(--navy);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
    border-right: 1px solid #1a2c48;
}
[data-testid="stSidebar"] * {
    color: #CBD5E8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--white) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: var(--navy-light) !important;
    color: var(--white) !important;
    border: 1px solid var(--slate) !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--slate) !important;
}

/* ── Masthead ── */
.masthead {
    background: linear-gradient(120deg, var(--navy) 0%, var(--navy-mid) 60%, var(--navy-light) 100%);
    padding: 28px 32px 24px;
    border-radius: var(--radius);
    margin-bottom: 28px;
    box-shadow: var(--shadow);
}
.masthead-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--steel);
    margin-bottom: 6px;
}
.masthead-title {
    font-size: 28px;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin-bottom: 10px;
}
.masthead-meta {
    font-size: 13px;
    color: var(--steel);
    font-weight: 400;
}
.masthead-meta b {
    color: var(--white);
    font-weight: 600;
}

/* ── Section headers ── */
.section-header {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 28px 0 16px;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 16px 14px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, var(--navy-light));
}
.metric-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.metric-value-large {
    font-size: 30px;
    font-weight: 800;
    color: var(--navy);
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-unit {
    font-size: 14px;
    font-weight: 500;
    color: var(--slate);
    margin-left: 2px;
}
.metric-note {
    font-size: 11px;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-safe    { background: var(--safe-pale);   color: var(--safe);   }
.badge-warning { background: var(--amber-pale);  color: var(--amber);  }
.badge-danger  { background: var(--danger-pale); color: var(--danger); }

/* ── Status block ── */
.status-block {
    padding: 16px 20px;
    border-radius: var(--radius);
    border: 1px solid;
    margin-bottom: 12px;
    font-size: 14px;
    line-height: 1.6;
}
.status-block.safe {
    background: var(--safe-pale);
    border-color: #b7dfc8;
    color: var(--safe);
}
.status-block.warning {
    background: var(--amber-pale);
    border-color: #f0d58c;
    color: #7a5a0a;
}
.status-block.danger {
    background: var(--danger-pale);
    border-color: #f1b8b8;
    color: var(--danger);
}
.status-block .status-title {
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* ── Report card ── */
.report-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    box-shadow: var(--shadow);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 15.5px;
    line-height: 1.8;
    color: #1a2440;
}
.report-card p {
    margin-bottom: 14px;
}
.report-card p:last-child {
    margin-bottom: 0;
}

/* ── Advisory tabs ── */
.advisory-panel {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #2a3a52;
    min-height: 100px;
}
.advisory-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 32px 0;
}

/* ── Chart container ── */
.chart-container {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 4px;
    box-shadow: var(--shadow);
}

/* ── Audit expander text ── */
.audit-entry-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--navy-mid);
    letter-spacing: 0.03em;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Suppress default Streamlit color strip on subheaders ── */
h2, h3 { color: var(--navy) !important; }

/* ── Tabs styling ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 2px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 8px 16px !important;
    border-radius: 4px 4px 0 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--navy-mid) !important;
    border-bottom: 2px solid var(--navy-mid) !important;
    background: var(--white) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_cached_agent_state():
    state_file = DATA_DIR / "latest_agent_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=60)
def load_historical_data():
    parquet_file = DATA_DIR / "historical_features.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None


@st.cache_data(ttl=60)
def load_training_metadata():
    tr_file = DATA_DIR / "training_results.json"
    if tr_file.exists():
        with open(tr_file, "r") as f:
            return json.load(f)
    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:11px;font-weight:700;letter-spacing:0.18em;"
        "text-transform:uppercase;color:#98C1D9;margin-bottom:4px;'>AgentWeather</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:20px;font-weight:800;color:#FFFFFF;"
        "line-height:1.2;margin-bottom:16px;'>Autonomous Weather<br>Intelligence Platform</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:12px;color:#98C1D9;margin-bottom:4px;'>"
        f"<b style='color:#FFFFFF;'>Target City</b><br>{CITY_NAME}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:12px;color:#98C1D9;margin-bottom:20px;'>"
        f"<b style='color:#FFFFFF;'>Coordinates</b><br>{CITY_LAT} N, {CITY_LON} E</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#1a2c48;margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#98C1D9;margin-bottom:10px;'>Live Operations</div>",
        unsafe_allow_html=True,
    )
    if st.button("Run Ingestion and Agent Cycle", use_container_width=True):
        with st.spinner("Fetching live observations and running LangGraph agent pipeline..."):
            live_df = ingest_live_row(city=CITY_NAME)
            eng_df = build_engineered_features(live_df)
            new_state = run_agent_pipeline(eng_df)
            st.cache_data.clear()
            st.success("Pipeline executed successfully.")
            st.rerun()

    st.markdown("<hr style='border-color:#1a2c48;margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#98C1D9;margin-bottom:10px;'>Agent Architecture</div>",
        unsafe_allow_html=True,
    )
    for idx, label in enumerate([
        "Forecast Agent  |  8 Champion Models",
        "Alert Agent  |  Groq LLM with Fallback",
        "Drift Agent  |  KS Test and Model MAE",
        "Report Agent  |  Broadcast Synthesis",
    ], 1):
        st.markdown(
            f"<div style='font-size:12px;color:#CBD5E8;margin-bottom:6px;'>"
            f"<span style='color:#98C1D9;font-weight:700;'>Node {idx}</span>  {label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1a2c48;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:10px;color:#5a7a9a;line-height:1.6;'>"
        "LangGraph  |  Hopsworks  |  Groq<br>OpenWeatherMap  |  AQICN  |  OpenMeteo</div>",
        unsafe_allow_html=True,
    )


# ─── Bootstrap ────────────────────────────────────────────────────────────────
agent_state = load_cached_agent_state()
hist_df = load_historical_data()
training_meta = load_training_metadata()

if agent_state is None:
    if hist_df is not None:
        with st.spinner("Initializing agent pipeline on first run..."):
            agent_state = run_agent_pipeline()
    else:
        st.warning("No historical data found. Run backfill and model training first.")
        st.stop()

forecasts   = agent_state.get("forecasts", {})
active_alerts = agent_state.get("active_alerts", [])
advisories  = agent_state.get("alert_advisories", {})
cycle_ts    = agent_state.get("timestamp", "N/A")
city_label  = agent_state.get("city", CITY_NAME)


# ─── Masthead ─────────────────────────────────────────────────────────────────
n_alerts = len(active_alerts)
alert_color = "#C8922A" if n_alerts == 0 else "#9B2335"
alert_label = "All Systems Nominal" if n_alerts == 0 else f"{n_alerts} Hazard Alert Active" if n_alerts == 1 else f"{n_alerts} Hazard Alerts Active"

st.markdown(f"""
<div class="masthead">
    <div class="masthead-eyebrow">Autonomous Weather Intelligence Platform</div>
    <div class="masthead-title">AgentWeather  |  {city_label}</div>
    <div class="masthead-meta">
        24 Hour Predictive Forecast and Air Quality Intelligence &nbsp;&nbsp;
        <b>Last Cycle:</b> {cycle_ts} UTC &nbsp;&nbsp;
        <span style="color:{alert_color};font-weight:700;">{alert_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Section 1: Metric Overview ───────────────────────────────────────────────
st.markdown('<div class="section-header">24 Hour Predictive Forecast  |  t + 24h Outlook</div>', unsafe_allow_html=True)

CARD_SPECS = [
    ("Air Quality Index",   "aqi",         "AQI",   "#E05C5C",  "Hazard threshold above 150"),
    ("Temperature",         "temp_c",      "deg C", "#E07A2F",  "Hazard below 0 or above 40"),
    ("Precipitation",       "precip_mm",   "mm",    "#3A7EC8",  f"Rain probability {round(forecasts.get('rain_prob', 0), 1)} pct"),
    ("Relative Humidity",   "humidity_pct","%",     "#2E9E6B",  "Hazard below 20 or above 85"),
    ("UV Index",            "uv_index",    "index", "#8A5EC8",  "Hazard at or above 8"),
    ("Wind Speed",          "wind_speed",  "km/h",  "#3D6B8A",  f"Direction {round(forecasts.get('wind_dir', 0), 0)} deg"),
]

cols = st.columns(6)
for col, (title, key, unit, accent, note) in zip(cols, CARD_SPECS):
    raw = forecasts.get(key, 0)
    display_val = round(float(raw), 1) if raw is not None else "N/A"
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--accent:{accent};">
            <div class="metric-eyebrow">{title}</div>
            <div class="metric-value-large">{display_val}
                <span class="metric-unit">{unit}</span>
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Section 2: Broadcast Report and Hazard Center ────────────────────────────
st.markdown('<div class="section-header">Official Broadcast Report and Hazard Advisory</div>', unsafe_allow_html=True)

col_report, col_hazard = st.columns([3, 2], gap="large")

with col_report:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:10px;'>"
        "Official 24 Hour Weather Report</div>",
        unsafe_allow_html=True,
    )
    report_text = agent_state.get("weather_report", "Weather report is being compiled.")
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
    para_html = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(f'<div class="report-card">{para_html}</div>', unsafe_allow_html=True)

with col_hazard:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:10px;'>"
        "Hazard Status and Public Advisories</div>",
        unsafe_allow_html=True,
    )

    if active_alerts:
        metrics_flagged = ", ".join(a["metric"].upper() for a in active_alerts)
        st.markdown(f"""
        <div class="status-block danger">
            <div class="status-title">Active Hazard Alert</div>
            {len(active_alerts)} parameter(s) exceed safe thresholds: {metrics_flagged}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-block safe">
            <div class="status-title">Status Normal</div>
            All 8 predicted parameters are within established safe thresholds.
        </div>
        """, unsafe_allow_html=True)

    tab_gen, tab_sens, tab_work = st.tabs(["General Public", "Sensitive Groups", "Outdoor Workers"])

    with tab_gen:
        st.markdown(
            f'<div class="advisory-panel">'
            f'<div class="advisory-label">Public Safety Advisory</div>'
            f'{advisories.get("general", "Normal precautions apply.")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with tab_sens:
        st.markdown(
            f'<div class="advisory-panel">'
            f'<div class="advisory-label">Health and Respiratory Protocol</div>'
            f'{advisories.get("sensitive", "No restrictions for sensitive individuals.")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with tab_work:
        st.markdown(
            f'<div class="advisory-panel">'
            f'<div class="advisory-label">Occupational Safety Rules</div>'
            f'{advisories.get("outdoor", "Standard outdoor workplace routines apply.")}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Section 3: Trend Chart ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Historical Trend and 24 Hour Forecast Horizon</div>', unsafe_allow_html=True)

METRIC_LABELS = {
    "temp_c":       "Temperature (deg C)",
    "aqi":          "Air Quality Index",
    "humidity_pct": "Relative Humidity (%)",
    "wind_speed":   "Wind Speed (km/h)",
    "precip_mm":    "Precipitation (mm)",
    "uv_index":     "UV Index",
}

if hist_df is not None and not hist_df.empty:
    chart_col, ctrl_col = st.columns([5, 1])
    with ctrl_col:
        chart_metric = st.selectbox(
            "Metric",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda x: METRIC_LABELS[x],
        )
        window_hrs = st.selectbox("Window", options=[48, 72, 168], format_func=lambda x: f"{x} hours", index=2)

    with chart_col:
        recent_history = hist_df.iloc[-window_hrs:].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_history["timestamp"],
            y=recent_history[chart_metric],
            mode="lines",
            name="Observed",
            line=dict(color="#2C5F8A", width=1.8),
            hovertemplate="%{y:.2f}<extra>Observed</extra>",
        ))

        last_ts  = recent_history["timestamp"].iloc[-1]
        pred_ts  = last_ts + pd.Timedelta(hours=24)
        pred_val = forecasts.get(chart_metric, recent_history[chart_metric].iloc[-1])

        fig.add_trace(go.Scatter(
            x=[last_ts, pred_ts],
            y=[recent_history[chart_metric].iloc[-1], pred_val],
            mode="lines+markers",
            name="24h Forecast",
            line=dict(color="#C8922A", width=2, dash="dot"),
            marker=dict(size=[5, 10], color=["#2C5F8A", "#C8922A"]),
            hovertemplate="%{y:.2f}<extra>Forecast</extra>",
        ))

        thresh = HAZARD_THRESHOLDS.get(chart_metric, {})
        if thresh.get("max") is not None:
            fig.add_hline(
                y=thresh["max"],
                line_dash="dot", line_color="#9B2335", line_width=1,
                annotation_text=f"Hazard Max  {thresh['max']}",
                annotation_font_size=10,
                annotation_font_color="#9B2335",
            )
        if thresh.get("min") is not None:
            fig.add_hline(
                y=thresh["min"],
                line_dash="dot", line_color="#C8922A", line_width=1,
                annotation_text=f"Hazard Min  {thresh['min']}",
                annotation_font_size=10,
                annotation_font_color="#C8922A",
            )

        fig.update_layout(
            margin=dict(l=12, r=12, t=28, b=12),
            height=360,
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="#FAFBFD",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1B3A6B", size=12),
            title=dict(
                text=f"{METRIC_LABELS[chart_metric]}  |  {window_hrs}h Observed with 24h Forecast",
                font=dict(size=13, color="#0A1628", weight=700),
                x=0.0, xanchor="left",
            ),
            xaxis=dict(
                title="Timestamp (UTC)",
                showgrid=True, gridcolor="#EDF0F5", linecolor="#DDE3EE",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=METRIC_LABELS[chart_metric],
                showgrid=True, gridcolor="#EDF0F5", linecolor="#DDE3EE",
                tickfont=dict(size=11),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─── Section 4: Agent Intelligence Panel ──────────────────────────────────────
st.markdown('<div class="section-header">LangGraph Agent Intelligence Panel</div>', unsafe_allow_html=True)

tab_ledger, tab_shap, tab_drift, tab_audit = st.tabs([
    "Forecast Ledger",
    "EDA and SHAP Interpretability",
    "Drift Surveillance",
    "LLM Audit Log",
])


# ── Tab 1: Forecast Ledger ────────────────────────────────────────────────────
with tab_ledger:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:12px;'>"
        "Hazard Rule Evaluation and Prediction Ledger</div>",
        unsafe_allow_html=True,
    )
    ledger_entries = agent_state.get("forecast_ledger_entries", [])
    if ledger_entries:
        df_ledger = pd.DataFrame(ledger_entries)
        st.dataframe(df_ledger, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction ledger records found in the current session.")

    # Training metrics table
    if training_meta:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#8A9BB0;margin-top:24px;margin-bottom:12px;'>"
            "Champion Model Validation Metrics</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for metric, info in training_meta.items():
            rows.append({
                "Target Variable": metric,
                "Champion Family": info.get("family", "N/A"),
                "RMSE": round(info.get("rmse", 0), 4),
                "MAE":  round(info.get("mae", 0), 4),
                "R2":   round(info.get("r2", 0), 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab 2: EDA and SHAP ───────────────────────────────────────────────────────
with tab_shap:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:12px;'>"
        "Machine Learning Interpretability and Exploratory Data Analysis</div>",
        unsafe_allow_html=True,
    )

    shap_col, eda_col = st.columns([1, 1], gap="large")

    with shap_col:
        selected_metric = st.selectbox(
            "Target Variable for SHAP Global Importance",
            options=TARGET_METRICS,
        )
        shap_img_path = EDA_ARTIFACTS_DIR / f"shap_importance_{selected_metric}.png"
        if shap_img_path.exists():
            st.image(
                str(shap_img_path),
                caption=f"SHAP Global Feature Importance  |  {selected_metric}  |  24h Forecast",
                use_container_width=True,
            )
        else:
            st.info(f"SHAP artifact for {selected_metric} is generated during model training.")

    with eda_col:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#8A9BB0;margin-bottom:10px;'>"
            "Metric Distributions</div>",
            unsafe_allow_html=True,
        )
        dist_path = EDA_ARTIFACTS_DIR / "eda_metric_distributions.png"
        if dist_path.exists():
            st.image(str(dist_path), caption="Observed Distributions Across 6 Target Categories", use_container_width=True)

        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#8A9BB0;margin-top:16px;margin-bottom:10px;'>"
            "Diurnal Seasonality</div>",
            unsafe_allow_html=True,
        )
        diurnal_path = EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png"
        if diurnal_path.exists():
            st.image(str(diurnal_path), caption="24 Hour Diurnal Cycles Across All Metrics", use_container_width=True)


# ── Tab 3: Drift Surveillance ─────────────────────────────────────────────────
with tab_drift:
    drift_data     = agent_state.get("drift_results", {})
    overall_drift  = drift_data.get("overall_drift_detected", False)

    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:12px;'>"
        "Data and Model Drift Surveillance</div>",
        unsafe_allow_html=True,
    )

    if overall_drift:
        st.markdown("""
        <div class="status-block danger">
            <div class="status-title">Drift Detected</div>
            Automated retraining workflow has been dispatched via GitHub Actions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-block safe">
            <div class="status-title">Nominal Conditions</div>
            No data drift detected (KS p-values above 0.05). Model forecast accuracy within acceptable bounds (MAE degradation below 25 percent).
        </div>
        """, unsafe_allow_html=True)

    ks_col, md_col = st.columns(2, gap="large")

    with ks_col:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#8A9BB0;margin-bottom:10px;'>"
            "Kolmogorov Smirnov Data Drift Test</div>",
            unsafe_allow_html=True,
        )
        ks_details = drift_data.get("data_drift", {}).get("details", {})
        if ks_details:
            ks_rows = []
            for feat, res in ks_details.items():
                ks_rows.append({
                    "Feature":      feat,
                    "KS Statistic": round(res.get("ks_statistic", 0), 4),
                    "P Value":      round(res.get("p_value", 1), 4),
                    "Status":       "DRIFT" if res.get("drift_detected") else "Stable",
                })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No KS test results available yet.")

    with md_col:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
            "text-transform:uppercase;color:#8A9BB0;margin-bottom:10px;'>"
            "Model Forecast Error Degradation</div>",
            unsafe_allow_html=True,
        )
        md_metrics = drift_data.get("model_drift", {}).get("metrics", {})
        if md_metrics:
            md_rows = []
            for m, res in md_metrics.items():
                md_rows.append({
                    "Target":        m,
                    "Recent MAE":    round(res.get("recent_mae", 0), 4),
                    "Baseline MAE":  round(res.get("baseline_mae", 0), 4),
                    "Degradation":   f"{round(res.get('pct_degradation', 0), 1)} pct",
                    "Status":        "DRIFT" if res.get("drift_detected") else "Accurate",
                })
            st.dataframe(pd.DataFrame(md_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating 24 hour prediction and actual observation pairs for model drift analysis.")


# ── Tab 4: LLM Audit Log ──────────────────────────────────────────────────────
with tab_audit:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#8A9BB0;margin-bottom:12px;'>"
        "Groq LLM Call Audit and Failure Log</div>",
        unsafe_allow_html=True,
    )

    call_logs = agent_state.get("audit_call_logs", [])

    filter_col, _ = st.columns([2, 4])
    with filter_col:
        filter_option = st.selectbox(
            "Filter",
            options=["All Calls", "Successful Calls Only", "Failed Calls Only"],
        )

    if call_logs:
        df_calls = pd.DataFrame(call_logs)
        if filter_option == "Failed Calls Only":
            df_calls = df_calls[df_calls["status"] == "failed"]
        elif filter_option == "Successful Calls Only":
            df_calls = df_calls[df_calls["status"] == "success"]

        if df_calls.empty:
            st.info(f"No log entries match the selected filter.")
        else:
            for idx, row in df_calls.iterrows():
                status_val   = row.get("status", "").upper()
                agent_name   = row.get("agent_name", "N/A")
                ts_val       = row.get("timestamp", "N/A")
                latency_val  = row.get("latency_ms", "N/A")
                label = f"{status_val}  |  {agent_name}  |  {ts_val}  |  {latency_val} ms"

                with st.expander(label):
                    info_col, _ = st.columns([3, 1])
                    with info_col:
                        st.markdown(
                            f"<div style='font-size:12px;'>"
                            f"<b>Agent:</b> {agent_name} &nbsp; "
                            f"<b>Status:</b> {status_val}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    if row.get("error_message"):
                        st.error(f"Error: {row.get('error_message')}")
                    st.text_area("Prompt Submitted", value=row.get("prompt", ""), height=90, key=f"p_{idx}", disabled=True)
                    st.text_area("Response Output",  value=str(row.get("response", "")), height=130, key=f"r_{idx}", disabled=True)
    else:
        st.info("No LLM call records logged yet in this session.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding:20px 0 8px;border-top:1px solid #DDE3EE;
            text-align:center;font-size:11px;color:#8A9BB0;letter-spacing:0.04em;">
    AgentWeather  |  Islamabad, Pakistan  |  LangGraph  Hopsworks  Groq  OpenWeatherMap  AQICN
</div>
""", unsafe_allow_html=True)
