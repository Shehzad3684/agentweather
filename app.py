"""
AgentWeather Streamlit Broadcast Dashboard
A clean, TV news-style weather forecasting and agentic intelligence portal:
1. Top broadcast news weather report & color-coded hazard alerts (General, Sensitive, Outdoor).
2. 24-hour predictive metric cards & interactive forecast charts.
3. Agent Activity Panel with 4 tabs:
   - Active and Past Alerts
   - EDA and SHAP Explanations
   - Drift Monitor
   - LLM Call and Failure Logs
"""

import os
import sys
import json
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

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"AgentWeather — {CITY_NAME} Forecast & Agent Intelligence",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern broadcast styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .broadcast-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .broadcast-headline {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #2a5298;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .alert-box-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 16px;
        border-radius: 8px;
        border-left: 6px solid #dc3545;
        margin-bottom: 12px;
    }
    .alert-box-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 16px;
        border-radius: 8px;
        border-left: 6px solid #28a745;
        margin-bottom: 12px;
    }
    .report-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        line-height: 1.7;
        font-size: 16px;
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading Helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Sidebar: Controls & System Telemetry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/partly-cloudy-day.png", width=64)
    st.title("AgentWeather")
    st.markdown(f"**Target City:** `{CITY_NAME}`")
    st.markdown(f"**Coordinates:** `{CITY_LAT}°N, {CITY_LON}°E`")
    st.markdown("---")

    st.subheader("⚡ Live Operations")
    if st.button("🚀 Ingest Live & Trigger Agent Cycle", use_container_width=True, type="primary"):
        with st.spinner("Fetching live multi-source observations & running 4 LangGraph agents..."):
            live_df = ingest_live_row(city=CITY_NAME)
            eng_df = build_engineered_features(live_df)
            new_state = run_agent_pipeline(eng_df)
            st.cache_data.clear()
            st.success("Hourly Ingestion and Agent Cycle Executed!")
            st.rerun()

    st.markdown("---")
    st.subheader("🛠️ Architecture Nodes")
    st.markdown("""
    * **Node 1:** Forecast Agent (8 Champions)
    * **Node 2:** Alert Agent (Groq / Fallback)
    * **Node 3:** Drift Agent (KS & Model Drift)
    * **Node 4:** Weather Report Agent (Broadcast)
    """)
    st.markdown("---")
    st.caption("Powered by LangGraph, Hopsworks, & Groq")


# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------
agent_state = load_cached_agent_state()
hist_df = load_historical_data()
training_meta = load_training_metadata()

# If agent state not yet run, initialize with historical row
if agent_state is None:
    if hist_df is not None:
        with st.spinner("Initializing first agent run..."):
            agent_state = run_agent_pipeline()
    else:
        st.warning("No data found. Please run backfill and model training first.")
        st.stop()

# Header Banner
st.markdown(f"""
<div class="broadcast-banner">
    <div class="broadcast-headline">📡 AgentWeather Broadcast Center: {agent_state.get('city', CITY_NAME)}</div>
    <div>24-Hour Autonomous Weather & Air Quality Intelligence | Last Cycle: <b>{agent_state.get('timestamp', 'N/A')} UTC</b></div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 1: Weather Report & Hazard Alert Center
# -----------------------------------------------------------------------------
col_report, col_alerts = st.columns([3, 2])

with col_report:
    st.subheader("🎙️ Official 24-Hour Broadcast Weather Report")
    report_text = agent_state.get("weather_report", "Weather report is being compiled...")
    paragraphs = [p for p in report_text.split("\n\n") if p.strip()]
    
    st.markdown(f"""
    <div class="report-card">
        {"".join([f"<p>{p}</p>" for p in paragraphs])}
    </div>
    """, unsafe_allow_html=True)

with col_alerts:
    st.subheader("⚠️ Hazard Warning Center")
    active_alerts = agent_state.get("active_alerts", [])
    advisories = agent_state.get("alert_advisories", {})

    if active_alerts:
        st.markdown(f"""
        <div class="alert-box-danger">
            <b>CRITICAL WARNING: {len(active_alerts)} HAZARD(S) ACTIVE</b><br>
            {", ".join([f"<b>{a['metric'].upper()}:</b> {a['value']} ({a['reason']})" for a in active_alerts])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box-safe">
            <b>STATUS GREEN: ALL METRICS NORMAL</b><br>
            All 8 predicted parameters are within safe municipal thresholds.
        </div>
        """, unsafe_allow_html=True)

    # 3 Color-Coded Hazard Alert Tabs
    tab_gen, tab_sens, tab_work = st.tabs(["👥 General Public", "🩺 Sensitive Groups", "👷 Outdoor Workers"])
    
    with tab_gen:
        st.markdown(f"**Safety Advisory:**\n\n{advisories.get('general', 'Normal precautions apply.')}")
    with tab_sens:
        st.markdown(f"**Health & Respiratory Protocol:**\n\n{advisories.get('sensitive', 'No restrictions for sensitive individuals.')}")
    with tab_work:
        st.markdown(f"**Occupational Work Rules:**\n\n{advisories.get('outdoor', 'Standard outdoor workplace routines apply.')}")


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 2: 24-Hour Predictive Metric Cards
# -----------------------------------------------------------------------------
st.subheader("📊 24-Hour Predictive Metric Categories (t + 24h Outlook)")
forecasts = agent_state.get("forecasts", {})

m_cols = st.columns(6)

cards_data = [
    ("Air Quality (AQI)", forecasts.get("aqi", 0), "AQI", "#e74c3c", "Hazard > 150"),
    ("Temperature", forecasts.get("temp_c", 0), "°C", "#e67e22", "< 0°C or > 40°C"),
    ("Precipitation", forecasts.get("precip_mm", 0), "mm", "#3498db", f"Rain Prob: {forecasts.get('rain_prob', 0)}%"),
    ("Humidity", forecasts.get("humidity_pct", 0), "%", "#2ecc71", "< 20% or > 85%"),
    ("UV Index", forecasts.get("uv_index", 0), "index", "#9b59b6", "Hazard >= 8"),
    ("Wind Velocity", forecasts.get("wind_speed", 0), "km/h", "#34495e", f"Dir: {forecasts.get('wind_dir', 0)}°"),
]

for col, (title, val, unit, color, note) in zip(m_cols, cards_data):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{val} <span style="font-size: 16px; font-weight: 500;">{unit}</span></div>
            <div style="font-size: 12px; color: #718096; margin-top: 4px;">{note}</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Section 3: Interactive Historical & Forecast Chart
# -----------------------------------------------------------------------------
st.subheader("📈 Interactive 7-Day Trend & 24-Hour Horizon")
if hist_df is not None and not hist_df.empty:
    chart_metric = st.selectbox("Select Metric to Visualize:", options=["temp_c", "aqi", "humidity_pct", "wind_speed", "precip_mm", "uv_index"], format_func=lambda x: x.upper())
    
    # Display last 168 hours (7 days)
    recent_history = hist_df.iloc[-168:].copy()
    
    fig = go.Figure()
    # Historical line
    fig.add_trace(go.Scatter(
        x=recent_history["timestamp"],
        y=recent_history[chart_metric],
        mode="lines",
        name=f"Observed {chart_metric.upper()}",
        line=dict(color="#2a5298", width=2),
    ))
    
    # 24h Prediction point
    last_ts = recent_history["timestamp"].iloc[-1]
    pred_ts = last_ts + pd.Timedelta(hours=24)
    pred_val = forecasts.get(chart_metric, recent_history[chart_metric].iloc[-1])
    
    fig.add_trace(go.Scatter(
        x=[last_ts, pred_ts],
        y=[recent_history[chart_metric].iloc[-1], pred_val],
        mode="lines+markers",
        name=f"24h Ahead Forecast ({pred_val})",
        line=dict(color="#e74c3c", width=3, dash="dot"),
        marker=dict(size=[6, 12], color=["#2a5298", "#e74c3c"], symbol=["circle", "star"]),
    ))
    
    # Hazard line if applicable
    thresh = HAZARD_THRESHOLDS.get(chart_metric, {})
    if thresh.get("max") is not None:
        fig.add_hline(y=thresh["max"], line_dash="dash", line_color="red", annotation_text=f"Hazard Max ({thresh['max']})")
    if thresh.get("min") is not None:
        fig.add_hline(y=thresh["min"], line_dash="dash", line_color="orange", annotation_text=f"Hazard Min ({thresh['min']})")
        
    fig.update_layout(
        title=f"7-Day Observed Hourly {chart_metric.upper()} with 24-Hour Predicted Forecast Point",
        xaxis_title="Timestamp (UTC)",
        yaxis_title=chart_metric.upper(),
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# -----------------------------------------------------------------------------
# Section 4: Agent Activity Panel (4 Comprehensive Tabs)
# -----------------------------------------------------------------------------
st.subheader("🤖 LangGraph Agent Activity Panel")

tab_alerts, tab_shap, tab_drift, tab_audit = st.tabs([
    "🚨 Active & Past Alerts",
    "🔬 EDA & SHAP Explanations",
    "📡 Drift Monitor",
    "📝 LLM Call & Failure Logs",
])

# Tab 1: Active and Past Alerts
with tab_alerts:
    st.markdown("### Hazard Rule Evaluation & Prediction Ledger")
    ledger_entries = agent_state.get("forecast_ledger_entries", [])
    if ledger_entries:
        df_ledger = pd.DataFrame(ledger_entries)
        st.dataframe(df_ledger, use_container_width=True)
    else:
        st.info("No prediction ledger records found in current session.")

# Tab 2: EDA & SHAP Explanations
with tab_shap:
    st.markdown("### Machine Learning Interpretability (SHAP) & Historical EDA")
    col_shap_select, col_shap_desc = st.columns([1, 2])
    with col_shap_select:
        selected_metric = st.selectbox("Select Target Variable for SHAP Plot:", options=TARGET_METRICS)
    
    shap_img_path = EDA_ARTIFACTS_DIR / f"shap_importance_{selected_metric}.png"
    if shap_img_path.exists():
        st.image(str(shap_img_path), caption=f"SHAP Global Feature Importance for 24h {selected_metric} Forecast", use_container_width=True)
    else:
        st.info(f"SHAP explanation artifact for {selected_metric} generated during training.")

    st.markdown("#### Baseline Exploratory Data Analysis Charts")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        if (EDA_ARTIFACTS_DIR / "eda_metric_distributions.png").exists():
            st.image(str(EDA_ARTIFACTS_DIR / "eda_metric_distributions.png"), caption="Distributions Across 6 Target Categories", use_container_width=True)
    with eda_col2:
        if (EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png").exists():
            st.image(str(EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png"), caption="24-Hour Diurnal Day/Night Cycles", use_container_width=True)

# Tab 3: Drift Monitor
with tab_drift:
    st.markdown("### Data & Model Drift Surveillance")
    drift_data = agent_state.get("drift_results", {})
    overall_drift = drift_data.get("overall_drift_detected", False)
    
    if overall_drift:
        st.error("⚠️ Active Drift Detected! Automated GitHub Actions retraining trigger has been dispatched.")
    else:
        st.success("✅ Nominal Conditions: No data drift (KS p-values > 0.05) or model drift (MAE degradation < 25%).")

    col_ks, col_md = st.columns(2)
    with col_ks:
        st.markdown("#### Two-Sample Kolmogorov-Smirnov (KS) Data Drift Test")
        ks_details = drift_data.get("data_drift", {}).get("details", {})
        if ks_details:
            ks_rows = []
            for feat, res in ks_details.items():
                ks_rows.append({
                    "Feature": feat,
                    "KS Statistic": res.get("ks_statistic"),
                    "P-Value": res.get("p_value"),
                    "Drift Status": "🚨 DRIFT" if res.get("drift_detected") else "✅ STABLE",
                })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True)

    with col_md:
        st.markdown("#### Model Forecast Error Degradation (Model Drift)")
        md_metrics = drift_data.get("model_drift", {}).get("metrics", {})
        if md_metrics:
            md_rows = []
            for m, res in md_metrics.items():
                md_rows.append({
                    "Target Metric": m,
                    "Recent MAE": res.get("recent_mae"),
                    "Baseline MAE": res.get("baseline_mae"),
                    "Degradation %": f"{res.get('pct_degradation')}%",
                    "Status": "🚨 DRIFT" if res.get("drift_detected") else "✅ ACCURATE",
                })
            st.dataframe(pd.DataFrame(md_rows), use_container_width=True)
        else:
            st.info("Accumulating 24-hour prediction-to-actual pairs for model drift monitoring.")

# Tab 4: LLM Call & Failure Logs
with tab_audit:
    st.markdown("### LangGraph Groq LLM Audit & Failure Log")
    call_logs = agent_state.get("audit_call_logs", [])
    
    filter_option = st.radio("Filter Audit Logs:", options=["All Calls", "Failed Calls Only (Failure Log)", "Successful Calls Only"], horizontal=True)
    
    if call_logs:
        df_calls = pd.DataFrame(call_logs)
        if filter_option == "Failed Calls Only (Failure Log)":
            df_calls = df_calls[df_calls["status"] == "failed"]
        elif filter_option == "Successful Calls Only":
            df_calls = df_calls[df_calls["status"] == "success"]

        if df_calls.empty:
            st.info(f"No logs matching filter '{filter_option}'.")
        else:
            for idx, row in df_calls.iterrows():
                with st.expander(f"[{row.get('status', '').upper()}] {row.get('agent_name')} — {row.get('timestamp')} (Latency: {row.get('latency_ms')}ms)"):
                    st.markdown(f"**Agent:** `{row.get('agent_name')}`")
                    st.markdown(f"**Status:** `{row.get('status')}`")
                    if row.get("error_message"):
                        st.error(f"Error Message: {row.get('error_message')}")
                    st.text_area("Prompt Submitted:", value=row.get("prompt", ""), height=100, key=f"p_{idx}")
                    st.text_area("Response / Output:", value=str(row.get("response", "")), height=150, key=f"r_{idx}")
    else:
        st.info("No LLM calls logged yet in this session.")
