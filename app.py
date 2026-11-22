"""
AgentWeather | Autonomous Weather Intelligence Platform
Islamabad, Pakistan | Powered by LangGraph, Hopsworks, and Groq
"""

import os
import sys
import json
import re
import datetime
from pathlib import Path

# Ensure project root in sys.path
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


# ─── Sanitization Utility (Guarantees ZERO dashes, hyphens, or em dashes) ─────
def clean_text(val) -> str:
    """
    Strips any hyphen, en dash, or em dash from UI visible strings,
    replacing them with natural spaces or connective wording.
    """
    if val is None:
        return ""
    text = str(val)
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    # Collapse multiple consecutive spaces
    return re.sub(r" +", " ", text).strip()


# ─── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"AgentWeather: {CITY_NAME} Forecast and Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Google Antigravity Design System Stylesheet ──────────────────────────────
st.markdown("""
<style>
/* ── Google Antigravity Palette & Design Tokens ── */
:root {
    --google-blue:        #1A73E8;
    --google-blue-hover:  #1557B0;
    --google-blue-soft:   #E8F0FE;
    --google-green:       #1E8E3E;
    --google-green-soft:  #E6F4EA;
    --google-red:         #D93025;
    --google-red-soft:    #FCE8E6;
    --google-amber:       #F9AB00;
    --google-amber-soft:  #FEF7E0;
    --google-dark:        #1F1F1F;
    --google-gray-700:    #444746;
    --google-gray-500:    #5F6368;
    --google-gray-300:    #DADCE0;
    --google-gray-100:    #F1F3F4;
    --google-gray-50:     #F8F9FA;
    --google-surface:     #FFFFFF;
    --card-shadow:        0 1px 3px 0 rgba(60, 64, 67, 0.08), 0 1px 2px 0 rgba(60, 64, 67, 0.04);
    --card-hover-shadow:  0 4px 12px 0 rgba(60, 64, 67, 0.12);
    --pill-radius:        9999px;
    --card-radius:        16px;
    --inner-radius:       12px;
}

/* ── Typography and Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--google-gray-50) !important;
    font-family: "Inter", "Google Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--google-dark) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Clean Google Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--google-gray-300) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] * {
    color: var(--google-gray-700) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--google-dark) !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: var(--google-blue) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--pill-radius) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    padding: 10px 20px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: var(--google-blue-hover) !important;
    box-shadow: 0 2px 6px rgba(26,115,232,0.3) !important;
}

/* ── Google Antigravity Masthead Card ── */
.antigravity-hero {
    background: var(--google-surface);
    border: 1px solid var(--google-gray-300);
    border-radius: var(--card-radius);
    padding: 32px 36px 28px;
    margin-bottom: 24px;
    box-shadow: var(--card-shadow);
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--google-blue-soft);
    color: var(--google-blue);
    padding: 4px 14px;
    border-radius: var(--pill-radius);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: var(--google-dark);
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 15px;
    color: var(--google-gray-700);
    line-height: 1.5;
    margin-bottom: 16px;
    font-weight: 400;
}
.hero-meta-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--google-gray-100);
    font-size: 13px;
    color: var(--google-gray-500);
}
.hero-meta-item b {
    color: var(--google-dark);
    font-weight: 600;
}

/* ── Section Titles ── */
.antigravity-section-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--google-gray-500);
    margin: 28px 0 14px 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.antigravity-section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--google-gray-300);
}

/* ── Metric Cards in Grid ── */
.google-metric-card {
    background: var(--google-surface);
    border: 1px solid var(--google-gray-300);
    border-radius: var(--card-radius);
    padding: 22px 20px 18px;
    box-shadow: var(--card-shadow);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.google-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--card-hover-shadow);
}
.metric-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.metric-category-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--google-gray-500);
}
.metric-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.metric-numeric-large {
    font-size: 34px;
    font-weight: 800;
    color: var(--google-dark);
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 6px 0;
}
.metric-unit-text {
    font-size: 14px;
    font-weight: 500;
    color: var(--google-gray-500);
    margin-left: 3px;
}
.metric-footer-note {
    font-size: 12px;
    color: var(--google-gray-500);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--google-gray-100);
    line-height: 1.4;
}

/* ── Content Containers ── */
.antigravity-card {
    background: var(--google-surface);
    border: 1px solid var(--google-gray-300);
    border-radius: var(--card-radius);
    padding: 26px 28px;
    box-shadow: var(--card-shadow);
    height: 100%;
}
.card-header-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--google-blue);
    margin-bottom: 8px;
}
.card-heading {
    font-size: 18px;
    font-weight: 700;
    color: var(--google-dark);
    margin-bottom: 16px;
    line-height: 1.3;
}

/* ── Status Pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: var(--pill-radius);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.status-pill.safe {
    background: var(--google-green-soft);
    color: var(--google-green);
    border: 1px solid #C8E6C9;
}
.status-pill.danger {
    background: var(--google-red-soft);
    color: var(--google-red);
    border: 1px solid #FADBD8;
}
.status-pill.warning {
    background: var(--google-amber-soft);
    color: #B06000;
    border: 1px solid #FFE082;
}

/* ── Notice Banner Blocks ── */
.notice-box {
    border-radius: var(--inner-radius);
    padding: 16px 20px;
    margin-bottom: 16px;
    font-size: 13.5px;
    line-height: 1.6;
    border: 1px solid;
}
.notice-box.safe {
    background: var(--google-green-soft);
    border-color: #C8E6C9;
    color: #137333;
}
.notice-box.danger {
    background: var(--google-red-soft);
    border-color: #FADBD8;
    color: #A50E0E;
}

/* ── Editorial Broadcast Prose ── */
.broadcast-editorial-body {
    font-size: 15px;
    line-height: 1.8;
    color: #2D3748;
}
.broadcast-editorial-body p {
    margin-bottom: 14px;
}
.broadcast-editorial-body p:last-child {
    margin-bottom: 0;
}

/* ── Advisory Panel Box ── */
.advisory-body-box {
    background: var(--google-gray-50);
    border: 1px solid var(--google-gray-300);
    border-radius: var(--inner-radius);
    padding: 18px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--google-gray-700);
    min-height: 120px;
}

/* ── Tabs Customization ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--google-gray-300) !important;
    padding-bottom: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: var(--pill-radius) !important;
    color: var(--google-gray-500) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: var(--google-gray-100) !important;
    color: var(--google-dark) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--google-blue-soft) !important;
    color: var(--google-blue) !important;
}

/* ── Clean DataFrames and Expanders ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--google-gray-300) !important;
    border-radius: var(--inner-radius) !important;
    overflow: hidden !important;
}
.streamlit-expanderHeader {
    background: var(--google-surface) !important;
    border: 1px solid var(--google-gray-300) !important;
    border-radius: var(--inner-radius) !important;
    font-weight: 600 !important;
    color: var(--google-dark) !important;
}

/* ── Footer ── */
.antigravity-footer {
    margin-top: 48px;
    padding: 24px 0 12px;
    border-top: 1px solid var(--google-gray-300);
    text-align: center;
    font-size: 12px;
    color: var(--google-gray-500);
    letter-spacing: 0.04em;
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


# ─── Sidebar: Controls and Architecture ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px;">
        <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #1A73E8; margin-bottom: 4px;">Google Antigravity Architecture</div>
        <div style="font-size: 22px; font-weight: 800; color: #1F1F1F; line-height: 1.2;">AgentWeather</div>
        <div style="font-size: 13px; color: #5F6368; margin-top: 4px;">Serverless Weather Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #F8F9FA; border: 1px solid #DADCE0; border-radius: 12px; padding: 14px 16px; margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #5F6368; margin-bottom: 4px;">Target Location</div>
        <div style="font-size: 15px; font-weight: 700; color: #1F1F1F;">{clean_text(CITY_NAME)}</div>
        <div style="font-size: 12px; color: #5F6368; margin-top: 2px;">Latitude {CITY_LAT} North, Longitude {CITY_LON} East</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #5F6368; margin-bottom: 8px;'>On Demand Execution</div>", unsafe_allow_html=True)
    if st.button("Run Ingestion and Agent Cycle", use_container_width=True):
        with st.spinner("Executing live multi source ingestion and LangGraph agent cycle..."):
            live_df = ingest_live_row(city=CITY_NAME)
            eng_df = build_engineered_features(live_df)
            new_state = run_agent_pipeline(eng_df)
            st.cache_data.clear()
            st.success("Hourly ingestion and agent cycle completed successfully.")
            st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #DADCE0; margin: 24px 0;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size: 11px; font-weight: 700; text-transform: uppercase; color: #5F6368; margin-bottom: 12px;'>Orchestration Pipeline</div>", unsafe_allow_html=True)
    nodes = [
        ("Node 1", "Forecast Agent", "8 Champion Models for 24h Ahead Targets"),
        ("Node 2", "Alert Agent", "Hazard Rules and Groq Llama Qwen Inference"),
        ("Node 3", "Drift Agent", "Kolmogorov Smirnov and MAE Drift Tests"),
        ("Node 4", "Report Agent", "Broadcast Journalistic Weather Synthesis"),
    ]
    for node_id, name, desc in nodes:
        st.markdown(f"""
        <div style="margin-bottom: 12px; padding: 10px 12px; background: #F8F9FA; border: 1px solid #E5E7EB; border-radius: 10px;">
            <div style="font-size: 10px; font-weight: 700; color: #1A73E8; text-transform: uppercase;">{node_id}</div>
            <div style="font-size: 13px; font-weight: 700; color: #1F1F1F;">{name}</div>
            <div style="font-size: 11px; color: #5F6368; margin-top: 2px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #DADCE0; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 11px; color: #70757A; line-height: 1.6;">
        <b>Technology Stack:</b><br>
        LangGraph Orchestrator<br>
        Hopsworks Feature Store<br>
        Groq Inference Engine<br>
        GitHub Actions Continuous Retraining
    </div>
    """, unsafe_allow_html=True)


# ─── Data Initialization ──────────────────────────────────────────────────────
agent_state = load_cached_agent_state()
hist_df = load_historical_data()
training_meta = load_training_metadata()

if agent_state is None:
    if hist_df is not None:
        with st.spinner("Initializing first agent intelligence run..."):
            agent_state = run_agent_pipeline()
    else:
        st.warning("No historical observations found. Please run backfill and model training first.")
        st.stop()

forecasts = agent_state.get("forecasts", {})
active_alerts = agent_state.get("active_alerts", [])
advisories = agent_state.get("alert_advisories", {})
cycle_raw = agent_state.get("timestamp", "N A")
# Format timestamp without dashes
cycle_display = clean_text(cycle_raw)
city_label = clean_text(agent_state.get("city", CITY_NAME))

n_alerts = len(active_alerts)
if n_alerts == 0:
    status_class = "safe"
    status_label = "● All Meteorological Parameters Nominal"
else:
    status_class = "danger"
    status_label = f"● {n_alerts} Active Hazard Warning Triggered"


# ─── Antigravity Hero Masthead ────────────────────────────────────────────────
st.markdown(f"""
<div class="antigravity-hero">
    <div class="hero-chip">Autonomous Weather Intelligence System</div>
    <div class="hero-title">AgentWeather Intelligence Portal: {city_label}</div>
    <div class="hero-subtitle">Continuous 24 hour predictive forecasting, municipal hazard detection, and machine learning surveillance</div>
    <div class="hero-meta-bar">
        <div class="hero-meta-item">Location: <b>{city_label}, Pakistan</b></div>
        <div class="hero-meta-item">Operational Cycle: <b>{cycle_display} UTC</b></div>
        <div><span class="status-pill {status_class}">{status_label}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Section 1: 24 Hour Predictive Metric Cards ───────────────────────────────
st.markdown('<div class="antigravity-section-title">24 Hour Predictive Forecast: Target Horizon at t plus 24 Hours</div>', unsafe_allow_html=True)

CARD_DEFINITIONS = [
    ("Air Quality Index",   "aqi",         "AQI",          "#D93025", "Hazard threshold above 150"),
    ("Temperature",         "temp_c",      "Celsius",      "#E37400", "Hazard below 0 or above 40"),
    ("Precipitation",       "precip_mm",   "mm",           "#1A73E8", f"Rain probability {round(forecasts.get('rain_prob', 0), 1)} percent"),
    ("Relative Humidity",   "humidity_pct","percent",      "#1E8E3E", "Hazard below 20 or above 85"),
    ("UV Index",            "uv_index",    "index",        "#9334E6", "Hazard threshold at or above 8"),
    ("Wind Velocity",       "wind_speed",  "km per hour",  "#475569", f"Direction {round(forecasts.get('wind_dir', 0), 0)} degrees"),
]

metric_cols = st.columns(6)
for col, (title, key, unit, dot_color, note) in zip(metric_cols, CARD_DEFINITIONS):
    raw_val = forecasts.get(key, 0)
    formatted_val = round(float(raw_val), 1) if raw_val is not None else "N A"
    with col:
        st.markdown(f"""
        <div class="google-metric-card">
            <div>
                <div class="metric-header-row">
                    <span class="metric-category-label">{title}</span>
                    <span class="metric-dot" style="background: {dot_color};"></span>
                </div>
                <div class="metric-numeric-large">{formatted_val}<span class="metric-unit-text">{unit}</span></div>
            </div>
            <div class="metric-footer-note">{clean_text(note)}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Section 2: Broadcast Weather Report and Hazard Advisories ────────────────
st.markdown('<div class="antigravity-section-title">Official Broadcast Summary and Municipal Hazard Advisories</div>', unsafe_allow_html=True)

col_report, col_advisories = st.columns([3, 2], gap="large")

with col_report:
    st.markdown("""
    <div class="antigravity-card">
        <div class="card-header-label">Journalistic Meteorological Briefing</div>
        <div class="card-heading">Official 24 Hour Weather Broadcast</div>
    """, unsafe_allow_html=True)

    report_raw = agent_state.get("weather_report", "Weather report is currently compiling.")
    # Sanitize and ensure no dashes in the editorial body
    paragraphs = [clean_text(p) for p in report_raw.split("\n\n") if p.strip()]
    paragraphs_html = "".join([f"<p>{p}</p>" for p in paragraphs])

    st.markdown(f"""
        <div class="broadcast-editorial-body">
            {paragraphs_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_advisories:
    st.markdown("""
    <div class="antigravity-card">
        <div class="card-header-label">Safety Evaluation Engine</div>
        <div class="card-heading">Hazard Advisories by Audience</div>
    """, unsafe_allow_html=True)

    if active_alerts:
        flagged_summary = ", ".join([clean_text(a["metric"]).upper() for a in active_alerts])
        st.markdown(f"""
        <div class="notice-box danger">
            <b>Active Hazard Notice:</b> {len(active_alerts)} target metric(s) exceed normal limits ({flagged_summary}). Protective measures are advised for municipal groups.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="notice-box safe">
            <b>Status Green:</b> All 8 predicted atmospheric and environmental parameters remain well within established safety boundaries.
        </div>
        """, unsafe_allow_html=True)

    tab_public, tab_sens, tab_workers = st.tabs(["General Public", "Sensitive Groups", "Outdoor Workers"])

    with tab_public:
        gen_text = clean_text(advisories.get("general", "Normal atmospheric precautions apply throughout the municipal area."))
        st.markdown(f'<div class="advisory-body-box">{gen_text}</div>', unsafe_allow_html=True)

    with tab_sens:
        sens_text = clean_text(advisories.get("sensitive", "Air quality parameters are normal. No specialized health protocols are currently required."))
        st.markdown(f'<div class="advisory-body-box">{sens_text}</div>', unsafe_allow_html=True)

    with tab_workers:
        work_text = clean_text(advisories.get("outdoor", "Standard workplace occupational guidelines remain in effect for field activities."))
        st.markdown(f'<div class="advisory-body-box">{work_text}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Section 3: Interactive Horizon Chart ─────────────────────────────────────
st.markdown('<div class="antigravity-section-title">Historical Observations and 24 Hour Forecast Trajectory</div>', unsafe_allow_html=True)

CHART_OPTIONS = {
    "temp_c":       "Temperature (Celsius)",
    "aqi":          "Air Quality Index (AQI)",
    "humidity_pct": "Relative Humidity (Percent)",
    "wind_speed":   "Wind Speed (km per hour)",
    "precip_mm":    "Precipitation (mm)",
    "uv_index":     "UV Index",
}

if hist_df is not None and not hist_df.empty:
    chart_view_col, controls_col = st.columns([5, 1])
    with controls_col:
        chart_metric = st.selectbox(
            "Target Metric",
            options=list(CHART_OPTIONS.keys()),
            format_func=lambda x: clean_text(CHART_OPTIONS[x]),
        )
        window_options = [48, 72, 168]
        selected_window = st.selectbox("Lookback Window", options=window_options, format_func=lambda x: f"{x} Hours", index=2)

    with chart_view_col:
        recent_window_df = hist_df.iloc[-selected_window:].copy()

        fig = go.Figure()
        # Observed trend
        fig.add_trace(go.Scatter(
            x=recent_window_df["timestamp"],
            y=recent_window_df[chart_metric],
            mode="lines",
            name="Observed Values",
            line=dict(color="#1A73E8", width=2.2),
            hovertemplate="%{y:.2f}<extra>Observed</extra>",
        ))

        # Prediction horizon point
        last_obs_time = recent_window_df["timestamp"].iloc[-1]
        pred_target_time = last_obs_time + pd.Timedelta(hours=24)
        pred_horizon_val = forecasts.get(chart_metric, recent_window_df[chart_metric].iloc[-1])

        fig.add_trace(go.Scatter(
            x=[last_obs_time, pred_target_time],
            y=[recent_window_df[chart_metric].iloc[-1], pred_horizon_val],
            mode="lines+markers",
            name="24 Hour Forecast Horizon",
            line=dict(color="#F9AB00", width=2.4, dash="dot"),
            marker=dict(size=[6, 11], color=["#1A73E8", "#F9AB00"]),
            hovertemplate="%{y:.2f}<extra>24h Ahead</extra>",
        ))

        # Hazard threshold guides
        rule = HAZARD_THRESHOLDS.get(chart_metric, {})
        if rule.get("max") is not None:
            fig.add_hline(
                y=rule["max"],
                line_dash="dot",
                line_color="#D93025",
                line_width=1.2,
                annotation_text=f"Hazard Upper Limit: {rule['max']}",
                annotation_font_size=10,
                annotation_font_color="#D93025",
            )
        if rule.get("min") is not None:
            fig.add_hline(
                y=rule["min"],
                line_dash="dot",
                line_color="#E37400",
                line_width=1.2,
                annotation_text=f"Hazard Lower Limit: {rule['min']}",
                annotation_font_size=10,
                annotation_font_color="#E37400",
            )

        metric_clean_label = clean_text(CHART_OPTIONS[chart_metric])
        fig.update_layout(
            margin=dict(l=14, r=14, t=32, b=14),
            height=370,
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FAFAFA",
            font=dict(family="Inter, Google Sans, Segoe UI, sans-serif", color="#1F1F1F", size=12),
            title=dict(
                text=f"{metric_clean_label}: {selected_window} Hours Observed with 24 Hour Prediction",
                font=dict(size=14, color="#1F1F1F", weight=700),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Timestamp UTC",
                showgrid=True,
                gridcolor="#E5E7EB",
                linecolor="#DADCE0",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=metric_clean_label,
                showgrid=True,
                gridcolor="#E5E7EB",
                linecolor="#DADCE0",
                tickfont=dict(size=11),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )
        st.markdown('<div class="antigravity-card" style="padding: 18px 22px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Section 4: LangGraph Agent Activity Panel ────────────────────────────────
st.markdown('<div class="antigravity-section-title">LangGraph Agent Intelligence and Observability Panel</div>', unsafe_allow_html=True)

tab_ledger, tab_shap, tab_drift, tab_audit = st.tabs([
    "Forecast Ledger",
    "EDA and SHAP Interpretability",
    "Drift Surveillance",
    "LLM Audit Log",
])

# ── Tab 1: Forecast Ledger ──
with tab_ledger:
    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #5F6368; margin-bottom: 12px;">
        Prediction Ledger Entries for Recent Cycle
    </div>
    """, unsafe_allow_html=True)

    ledger_records = agent_state.get("forecast_ledger_entries", [])
    if ledger_records:
        clean_ledger = []
        for r in ledger_records:
            clean_ledger.append({
                "Forecast Timestamp": clean_text(r.get("forecast_created_at")),
                "Target Timestamp":   clean_text(r.get("target_timestamp")),
                "Metric":             clean_text(r.get("metric")),
                "Predicted Value":    r.get("predicted_value"),
                "Model Family":       clean_text(r.get("model_family")),
                "City":               clean_text(r.get("city")),
            })
        st.dataframe(pd.DataFrame(clean_ledger), use_container_width=True, hide_index=True)
    else:
        st.info("No prediction ledger records found in the current operational cycle.")

    if training_meta:
        st.markdown("""
        <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #5F6368; margin-top: 24px; margin-bottom: 12px;">
            Champion Models and Validation Performance
        </div>
        """, unsafe_allow_html=True)
        benchmarks = []
        for target_name, info in training_meta.items():
            benchmarks.append({
                "Target Metric":   clean_text(target_name),
                "Champion Family": clean_text(info.get("family", "N A")),
                "RMSE Score":      round(info.get("rmse", 0), 4),
                "MAE Score":       round(info.get("mae", 0), 4),
                "R2 Score":        round(info.get("r2", 0), 4),
            })
        st.dataframe(pd.DataFrame(benchmarks), use_container_width=True, hide_index=True)

# ── Tab 2: EDA and SHAP ──
with tab_shap:
    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #5F6368; margin-bottom: 14px;">
        Machine Learning Interpretability via SHAP and Exploratory Analysis
    </div>
    """, unsafe_allow_html=True)

    shap_col, eda_col = st.columns([1, 1], gap="large")

    with shap_col:
        selected_shap_metric = st.selectbox(
            "Select Target Variable for SHAP Importance Analysis",
            options=TARGET_METRICS,
            format_func=lambda x: clean_text(x),
        )
        shap_file = EDA_ARTIFACTS_DIR / f"shap_importance_{selected_shap_metric}.png"
        if shap_file.exists():
            st.image(
                str(shap_file),
                caption=f"SHAP Global Feature Importance for {clean_text(selected_shap_metric)} Forecast",
                use_container_width=True,
            )
        else:
            st.info(f"SHAP artifact for {clean_text(selected_shap_metric)} was generated during model training.")

    with eda_col:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>Target Variable Distributions</div>", unsafe_allow_html=True)
        dist_img = EDA_ARTIFACTS_DIR / "eda_metric_distributions.png"
        if dist_img.exists():
            st.image(str(dist_img), caption="Historical Distributions Across 6 Target Categories", use_container_width=True)

        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1F1F1F; margin-top: 16px; margin-bottom: 8px;'>Diurnal Seasonality Profiles</div>", unsafe_allow_html=True)
        diurnal_img = EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png"
        if diurnal_img.exists():
            st.image(str(diurnal_img), caption="24 Hour Diurnal Day and Night Cycles", use_container_width=True)

# ── Tab 3: Drift Surveillance ──
with tab_drift:
    drift_info = agent_state.get("drift_results", {})
    drift_flag = drift_info.get("overall_drift_detected", False)

    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #5F6368; margin-bottom: 14px;">
        Data and Model Drift Surveillance
    </div>
    """, unsafe_allow_html=True)

    if drift_flag:
        st.markdown("""
        <div class="notice-box danger">
            <b>Statistical Drift Detected:</b> Automated continuous retraining workflow has been dispatched via GitHub Actions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="notice-box safe">
            <b>Nominal Conditions:</b> Kolmogorov Smirnov p values exceed 0.05. Model forecast error degradation remains under 25 percent.
        </div>
        """, unsafe_allow_html=True)

    ks_col, md_col = st.columns(2, gap="large")

    with ks_col:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>Two Sample Kolmogorov Smirnov Test</div>", unsafe_allow_html=True)
        ks_details = drift_info.get("data_drift", {}).get("details", {})
        if ks_details:
            ks_rows = []
            for feat_name, item in ks_details.items():
                ks_rows.append({
                    "Feature":      clean_text(feat_name),
                    "KS Statistic": round(item.get("ks_statistic", 0), 4),
                    "P Value":      round(item.get("p_value", 1), 4),
                    "Status":       "Drift Detected" if item.get("drift_detected") else "Stable",
                })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating feature rows for Kolmogorov Smirnov verification.")

    with md_col:
        st.markdown("<div style='font-size: 13px; font-weight: 700; color: #1F1F1F; margin-bottom: 8px;'>Model Forecast Error Degradation</div>", unsafe_allow_html=True)
        md_details = drift_info.get("model_drift", {}).get("metrics", {})
        if md_details:
            md_rows = []
            for metric_k, item in md_details.items():
                md_rows.append({
                    "Target Metric": clean_text(metric_k),
                    "Recent MAE":    round(item.get("recent_mae", 0), 4),
                    "Baseline MAE":  round(item.get("baseline_mae", 0), 4),
                    "Degradation":   f"{round(item.get('pct_degradation', 0), 1)} percent",
                    "Status":        "Drift Detected" if item.get("drift_detected") else "Accurate",
                })
            st.dataframe(pd.DataFrame(md_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating 24 hour prediction and actual observation pairs for model error evaluation.")

# ── Tab 4: LLM Audit Log ──
with tab_audit:
    st.markdown("""
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #5F6368; margin-bottom: 14px;">
        Groq Language Model Inference and Failure Audit Log
    </div>
    """, unsafe_allow_html=True)

    calls_list = agent_state.get("audit_call_logs", [])
    filter_choice = st.selectbox("Log Filter", options=["All Logged Inferences", "Successful Inferences Only", "Failed Inferences Only"])

    if calls_list:
        df_llm = pd.DataFrame(calls_list)
        if filter_choice == "Failed Inferences Only":
            df_llm = df_llm[df_llm["status"] == "failed"]
        elif filter_choice == "Successful Inferences Only":
            df_llm = df_llm[df_llm["status"] == "success"]

        if df_llm.empty:
            st.info("No inference logs match the selected filter.")
        else:
            for idx, r in df_llm.iterrows():
                status_raw = clean_text(r.get("status", "")).upper()
                agent_lbl  = clean_text(r.get("agent_name", "UnknownAgent"))
                ts_lbl     = clean_text(r.get("timestamp", ""))
                lat_lbl    = r.get("latency_ms", 0)

                expander_label = f"Status: {status_raw} | Agent: {agent_lbl} | Timestamp: {ts_lbl} | Latency: {lat_lbl} ms"
                with st.expander(expander_label):
                    st.markdown(f"<b>Agent Node:</b> {agent_lbl} &nbsp;&nbsp; <b>Execution Status:</b> {status_raw}", unsafe_allow_html=True)
                    if r.get("error_message"):
                        st.error(f"Error Message: {clean_text(r.get('error_message'))}")
                    st.text_area("Submitted System and User Prompt", value=clean_text(r.get("prompt", "")), height=90, key=f"p_{idx}", disabled=True)
                    st.text_area("Generated Model Output", value=clean_text(r.get("response", "")), height=130, key=f"r_{idx}", disabled=True)
    else:
        st.info("No Language Model inference events logged in the active session.")


# ─── Clean Google Antigravity Footer ──────────────────────────────────────────
st.markdown(f"""
<div class="antigravity-footer">
    AgentWeather Platform &bull; {clean_text(CITY_NAME)}, Pakistan &bull; LangGraph &bull; Hopsworks &bull; Groq &bull; OpenWeatherMap &bull; AQICN
</div>
""", unsafe_allow_html=True)
