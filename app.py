"""
AgentWeather — Meteorological Intelligence & ML Observability Platform
Islamabad, Pakistan | Serverless Pipeline Powered by LangGraph, Hopsworks, and Groq
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
import plotly.graph_objects as go

from src.config import (
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    DATA_DIR,
    EDA_ARTIFACTS_DIR,
    HAZARD_THRESHOLDS,
    TARGET_METRICS,
)
from src.features import ingest_live_row, build_engineered_features
from src.agents import run_agent_pipeline


# ─── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"AgentWeather • {CITY_NAME}",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Design Tokens & Stylesheet (Tomorrow.io / Linear / Vercel Aesthetic) ─────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Light Scientific Console Design Tokens ── */
:root {
    --bg-page:        #F8FAFC;
    --bg-surface:     #FFFFFF;
    --bg-subtle:      #F1F5F9;
    --border-light:   #E2E8F0;
    --border-dark:    #CBD5E1;
    --text-primary:   #0F172A;
    --text-secondary: #475569;
    --text-muted:     #94A3B8;
    --brand-primary:  #0284C7;
    --brand-hover:    #0369A1;
    --brand-soft:     #E0F2FE;
    --status-emerald: #16A34A;
    --status-emerald-soft: #F0FDF4;
    --status-emerald-border: #BBF7D0;
    --status-amber:   #D97706;
    --status-amber-soft: #FFFBEB;
    --status-amber-border: #FDE68A;
    --status-red:     #DC2626;
    --status-red-soft: #FEF2F2;
    --status-red-border: #FECACA;
    --radius-card:    10px;
    --radius-inner:   8px;
    --radius-pill:    6px;
    --shadow-sm:      0 1px 3px 0 rgba(15, 23, 42, 0.06);
    --shadow-md:      0 4px 6px -1px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
}

/* ── Global Page Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-page) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

header[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.88) !important;
    backdrop-filter: blur(8px) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

/* ── Sidebar (Clean & High-Signal) ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid var(--border-light) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-secondary);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* ── Primary Sidebar Action Button (High-Contrast Fix) ── */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton button {
    background-color: #0284C7 !important;
    background: #0284C7 !important;
    color: #FFFFFF !important;
    border: 1px solid #0369A1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 16px !important;
    box-shadow: 0 1px 3px rgba(2, 132, 199, 0.3) !important;
    transition: all 0.15s ease-in-out !important;
    width: 100% !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] button[kind="secondary"] *,
[data-testid="stSidebar"] .stButton button *,
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span,
[data-testid="stSidebar"] .stButton button div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #0369A1 !important;
    background: #0369A1 !important;
    border-color: #075985 !important;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover *,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover *,
[data-testid="stSidebar"] button[kind="primary"]:hover *,
[data-testid="stSidebar"] .stButton button:hover * {
    color: #FFFFFF !important;
}

/* ── Top Hero Header Bar ── */
.hero-header-bar {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-card);
    padding: 18px 22px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}
.hero-title-group .eyebrow {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--brand-primary);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.hero-title-group .main-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.25;
}
.hero-meta-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: var(--radius-pill);
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    background: #FFFFFF;
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
}

/* Weather Snapshot Strip in Hero */
.hero-snapshot-strip {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    width: 100%;
    padding-top: 12px;
    margin-top: 12px;
    border-top: 1px solid var(--bg-subtle);
    font-size: 12px;
    color: var(--text-secondary);
}
.snapshot-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    background: var(--bg-subtle);
    border-radius: var(--radius-pill);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
}

/* ── Section Titles ── */
.section-headline {
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin: 22px 0 12px 2px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Predictive Metric Cards with Mini Visual Gauges ── */
.metric-card-console {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-card);
    padding: 16px 18px 14px;
    box-shadow: var(--shadow-sm);
    min-height: 154px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    position: relative;
    overflow: hidden;
}
.metric-card-console:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border-dark);
    transform: translateY(-1px);
}
.card-accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 2px;
}
.metric-name-group {
    display: flex;
    align-items: center;
    gap: 6px;
}
.metric-icon {
    font-size: 13px;
}
.metric-name {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.metric-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: var(--radius-pill);
    border: 1px solid;
}
.metric-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.metric-value-row {
    margin: 4px 0 2px;
    display: flex;
    align-items: baseline;
    gap: 5px;
}
.metric-number {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.metric-unit {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

/* Micro Gauge Track */
.mini-bar-track {
    position: relative;
    width: 100%;
    height: 4px;
    border-radius: 2px;
    margin: 6px 0 4px;
    background: #E2E8F0;
}
.mini-bar-pointer {
    position: absolute;
    top: -3px;
    width: 5px;
    height: 10px;
    background: #0F172A;
    border-radius: 2px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    transform: translateX(-50%);
}
.mini-bar-fill {
    height: 100%;
    border-radius: 2px;
}

.metric-footer {
    font-size: 11.5px;
    color: var(--text-secondary);
    padding-top: 6px;
    border-top: 1px solid var(--bg-subtle);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Unified Cards (Weather Briefing & Advisories) ── */
.console-card-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-card);
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    height: 100%;
}
.card-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-light);
}
.card-main-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.card-subtitle-tag {
    font-size: 11px;
    font-weight: 500;
    color: var(--brand-primary);
    background: var(--brand-soft);
    padding: 3px 8px;
    border-radius: var(--radius-pill);
}

/* Briefing Callout & Ribbon */
.briefing-callout {
    background: var(--bg-subtle);
    border-left: 3px solid var(--brand-primary);
    padding: 12px 14px;
    border-radius: 0 var(--radius-inner) var(--radius-inner) 0;
    margin-bottom: 14px;
}
.briefing-highlight-text {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.45;
    margin-bottom: 8px;
}
.briefing-ribbon {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.ribbon-chip {
    font-size: 11.5px;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: var(--radius-pill);
    background: #FFFFFF;
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
}

.briefing-body-text {
    font-size: 14px;
    line-height: 1.65;
    color: #334155;
    max-width: 68ch;
}
.briefing-body-text p {
    margin-bottom: 12px;
}
.briefing-body-text p:last-child {
    margin-bottom: 0;
}

/* Audience Advisory Row */
.advisory-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 11px 0;
    border-bottom: 1px solid var(--bg-subtle);
    font-size: 13px;
    line-height: 1.5;
}
.advisory-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.advisory-badge {
    flex-shrink: 0;
    padding: 3px 9px;
    border-radius: var(--radius-pill);
    font-size: 11px;
    font-weight: 600;
    border: 1px solid;
    white-space: nowrap;
}
.advisory-text {
    color: var(--text-secondary);
}

/* ── Modern High-Contrast Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--border-light) !important;
    padding-bottom: 1px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-pill) var(--radius-pill) 0 0 !important;
    color: var(--text-secondary) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-subtle) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--brand-primary) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--brand-primary) !important;
    background: transparent !important;
}

/* ── Tables & DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-inner) !important;
    overflow: hidden !important;
}

/* Summary Ribbon in Tab 1 */
.tab-summary-ribbon {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--bg-subtle);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-inner);
    padding: 8px 14px;
    margin-bottom: 14px;
    font-size: 12px;
    color: var(--text-secondary);
}
.tab-summary-item b {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-inner) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ── Footer ── */
.console-footer {
    margin-top: 40px;
    padding: 20px 0 10px;
    border-top: 1px solid var(--border-light);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 11.5px;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_cached_agent_state():
    state_file = DATA_DIR / "latest_agent_state.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
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
        with open(tr_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── Formatting & Meteorological Helpers ──────────────────────────────────────
def degrees_to_cardinal(deg: float) -> str:
    """Converts 0-360 compass degrees to cardinal direction."""
    if deg is None or np.isnan(deg):
        return "N/A"
    cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                 "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(float(deg) / 22.5) % 16
    return cardinals[idx]


# 8 Forecast Target Metrics with Standard Scientific Units & Semantic Thresholds
METRIC_REGISTRY = {
    "aqi": {
        "name": "Air Quality Index",
        "icon": "🍃",
        "unit": "AQI",
        "eval_func": lambda v: ("Good", "#16A34A", "#F0FDF4", "#BBF7D0", "Safe threshold < 150") if v <= 50
                              else ("Moderate", "#D97706", "#FFFBEB", "#FDE68A", "Advisory limit 51–150") if v <= 150
                              else ("Hazardous", "#DC2626", "#FEF2F2", "#FECACA", "Hazardous > 150"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: linear-gradient(90deg, #16A34A 0%, #16A34A 25%, #EAB308 25%, #EAB308 50%, #F97316 50%, #F97316 75%, #DC2626 75%, #DC2626 100%);">
                <div class="mini-bar-pointer" style="left: {min(100.0, max(0.0, (v / 200.0) * 100.0)):.1f}%;"></div>
            </div>
        """,
    },
    "temp_c": {
        "name": "Temperature",
        "icon": "🌡️",
        "unit": "°C",
        "eval_func": lambda v: ("Extreme", "#DC2626", "#FEF2F2", "#FECACA", "Hazard (< 0°C or > 40°C)") if v < 0 or v > 40
                              else ("Advisory", "#D97706", "#FFFBEB", "#FDE68A", "Advisory (< 12°C or > 34°C)") if v < 12 or v > 34
                              else ("Nominal", "#16A34A", "#F0FDF4", "#BBF7D0", "Comfort range 12°C – 34°C"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: linear-gradient(90deg, #38BDF8 0%, #38BDF8 25%, #4ADE80 25%, #4ADE80 70%, #FB923C 70%, #F87171 100%);">
                <div class="mini-bar-pointer" style="left: {min(100.0, max(0.0, ((v + 5) / 45.0) * 100.0)):.1f}%;"></div>
            </div>
        """,
    },
    "precip_mm": {
        "name": "Precipitation",
        "icon": "🌧️",
        "unit": "mm",
        "eval_func": lambda v: ("Heavy Rain", "#DC2626", "#FEF2F2", "#FECACA", "Hazard threshold > 20 mm") if v > 20
                              else ("Moderate", "#D97706", "#FFFBEB", "#FDE68A", "Accumulation 2.5–20 mm") if v >= 2.5
                              else ("Dry", "#16A34A", "#F0FDF4", "#BBF7D0", "Minimal accumulation (< 2.5 mm)"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: #E2E8F0;">
                <div class="mini-bar-fill" style="width: {min(100.0, max(0.0, (v / 25.0) * 100.0)):.1f}%; background: #0284C7;"></div>
            </div>
        """,
    },
    "rain_prob": {
        "name": "Rain Probability",
        "icon": "☔",
        "unit": "%",
        "eval_func": lambda v: ("High Risk", "#DC2626", "#FEF2F2", "#FECACA", "High likelihood > 75%") if v > 75
                              else ("Moderate", "#D97706", "#FFFBEB", "#FDE68A", "Chance of rain 30%–75%") if v >= 30
                              else ("Low Risk", "#16A34A", "#F0FDF4", "#BBF7D0", "Rain unlikely (< 30%)"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: #E2E8F0;">
                <div class="mini-bar-fill" style="width: {min(100.0, max(0.0, v)):.1f}%; background: #0284C7;"></div>
            </div>
        """,
    },
    "humidity_pct": {
        "name": "Relative Humidity",
        "icon": "💧",
        "unit": "%",
        "eval_func": lambda v: ("Critical", "#DC2626", "#FEF2F2", "#FECACA", "Critical limit (< 20% or > 85%)") if v < 20 or v > 85
                              else ("Dry Air", "#D97706", "#FFFBEB", "#FDE68A", "Low hydration advisory (< 30%)") if v < 30
                              else ("High Moisture", "#D97706", "#FFFBEB", "#FDE68A", "Humid advisory (> 70%)") if v > 70
                              else ("Comfortable", "#16A34A", "#F0FDF4", "#BBF7D0", "Optimal comfort 30% – 70%"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: linear-gradient(90deg, #FDE68A 0%, #FDE68A 30%, #BBF7D0 30%, #BBF7D0 70%, #BAE6FD 70%, #BAE6FD 100%);">
                <div class="mini-bar-pointer" style="left: {min(100.0, max(0.0, v)):.1f}%;"></div>
            </div>
        """,
    },
    "uv_index": {
        "name": "UV Index",
        "icon": "☀️",
        "unit": "UV",
        "eval_func": lambda v: ("Very High", "#DC2626", "#FEF2F2", "#FECACA", "Hazardous index ≥ 8.0") if v >= 8
                              else ("Moderate", "#D97706", "#FFFBEB", "#FDE68A", "Moderate exposure 3.0–7.9") if v >= 3
                              else ("Low", "#16A34A", "#F0FDF4", "#BBF7D0", "Low sun radiation (< 3.0)"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: linear-gradient(90deg, #16A34A 0%, #16A34A 25%, #EAB308 25%, #EAB308 58%, #F97316 58%, #F97316 83%, #DC2626 83%, #9333EA 100%);">
                <div class="mini-bar-pointer" style="left: {min(100.0, max(0.0, (v / 12.0) * 100.0)):.1f}%;"></div>
            </div>
        """,
    },
    "wind_speed": {
        "name": "Wind Speed",
        "icon": "💨",
        "unit": "km/h",
        "eval_func": lambda v: ("High Wind", "#DC2626", "#FEF2F2", "#FECACA", "High wind hazard > 50 km/h") if v > 50
                              else ("Breezy", "#D97706", "#FFFBEB", "#FDE68A", "Moderate breeze 20–50 km/h") if v >= 20
                              else ("Calm", "#16A34A", "#F0FDF4", "#BBF7D0", "Light breeze (< 20 km/h)"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: #E2E8F0;">
                <div class="mini-bar-fill" style="width: {min(100.0, max(0.0, (v / 50.0) * 100.0)):.1f}%; background: #0284C7;"></div>
            </div>
        """,
    },
    "wind_dir": {
        "name": "Wind Direction",
        "icon": "🧭",
        "unit": "",
        "eval_func": lambda v: (degrees_to_cardinal(v), "#0284C7", "#F0F9FF", "#BAE6FD", f"Bearing {degrees_to_cardinal(v)} from North"),
        "gauge_html": lambda v: f"""
            <div class="mini-bar-track" style="background: #E2E8F0;">
                <div class="mini-bar-fill" style="width: {min(100.0, max(0.0, (v / 360.0) * 100.0)):.1f}%; background: #0284C7;"></div>
            </div>
        """,
    },
}


def render_advisory_section(advisory_text: str):
    """Parses advisory text into clean structured category rows with colored severity pills."""
    if not advisory_text:
        st.info("No active advisories issued for this audience.")
        return

    lines = [line.strip() for line in advisory_text.splitlines() if line.strip()]
    structured_rows = []
    plain_paragraphs = []

    icon_map = {
        "air quality": "🍃",
        "respiratory": "🫁",
        "hydration": "💧",
        "commuting": "🚗",
        "uv": "☀️",
        "thermal": "🌡️",
        "ppe": "🦺",
        "work": "⏰",
    }

    for line in lines:
        match = re.match(r"^[-*•]?\s*\*\*([^*:]+):?\*\*:?\s*(.+)$", line)
        if match:
            cat = match.group(1).strip()
            desc = match.group(2).strip()
            body_lower = desc.lower()
            cat_lower = cat.lower()

            # Select icon
            icon = "📋"
            for key_phrase, ic in icon_map.items():
                if key_phrase in cat_lower:
                    icon = ic
                    break

            if any(w in body_lower for w in ["hazard", "critical", "mandatory", "severe", "danger"]):
                badge_color = "#DC2626"
                badge_bg = "#FEF2F2"
                badge_border = "#FECACA"
            elif any(w in body_lower for w in ["priority", "advisory", "caution", "low", "limit", "dry", "elevated", "protect"]):
                badge_color = "#D97706"
                badge_bg = "#FFFBEB"
                badge_border = "#FDE68A"
            else:
                badge_color = "#16A34A"
                badge_bg = "#F0FDF4"
                badge_border = "#BBF7D0"

            structured_rows.append((f"{icon} {cat}", desc, badge_color, badge_bg, badge_border))
        else:
            plain_paragraphs.append(line)

    if structured_rows:
        for cat, desc, col, bg, bdr in structured_rows:
            st.markdown(f"""
            <div class="advisory-row">
                <span class="advisory-badge" style="color: {col}; background: {bg}; border-color: {bdr};">{cat}</span>
                <span class="advisory-text">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    elif plain_paragraphs:
        st.markdown("\n\n".join(plain_paragraphs))


# ─── Data Initialization ──────────────────────────────────────────────────────
agent_state = load_cached_agent_state()
hist_df = load_historical_data()
training_meta = load_training_metadata()

if agent_state is None:
    if hist_df is not None:
        with st.spinner("Initializing first pipeline execution..."):
            agent_state = run_agent_pipeline()
    else:
        st.error("Historical feature store not found. Please run backfill and model training first.")
        st.stop()

forecasts = agent_state.get("forecasts", {})
active_alerts = agent_state.get("active_alerts", [])
advisories = agent_state.get("alert_advisories", {})
cycle_timestamp = agent_state.get("timestamp", "N/A")
city_name = agent_state.get("city", CITY_NAME)

# Extract recent station telemetry from historical data
latest_telemetry = {}
if hist_df is not None and not hist_df.empty:
    latest_row = hist_df.iloc[-1]
    latest_telemetry = {
        "pressure": f"{latest_row.get('pressure_hpa', 1013.2):.1f} hPa",
        "cloud_cover": f"{round(latest_row.get('cloud_cover', 20))}%",
        "current_temp": f"{latest_row.get('temp_c', 18.0):.1f}°C",
    }
else:
    latest_telemetry = {
        "pressure": "1013.2 hPa",
        "cloud_cover": "20%",
        "current_temp": "24.8°C",
    }

n_alerts = len(active_alerts)
if n_alerts == 0:
    overall_status_text = "● All Parameters Nominal"
    overall_status_col = "#16A34A"
    overall_status_bg = "#F0FDF4"
    overall_status_bdr = "#BBF7D0"
else:
    overall_status_text = f"● {n_alerts} Active Hazard Warning(s)"
    overall_status_col = "#DC2626"
    overall_status_bg = "#FEF2F2"
    overall_status_bdr = "#FECACA"


# ─── Sidebar: Compact Scientific Console ──────────────────────────────────────
with st.sidebar:
    # Brand Header
    st.markdown(f"""
    <div style="padding: 6px 0 16px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 18px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;">AgentWeather</span>
            <span style="font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 6px; background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534;">
                ● Live • {city_name}
            </span>
        </div>
        <div style="font-size: 12px; color: #64748B;">Meteorological Intelligence & ML Observability</div>
    </div>
    """, unsafe_allow_html=True)

    # Location Details Card
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px;">
        <div style="font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 3px;">Station Location</div>
        <div style="font-size: 13.5px; font-weight: 600; color: #0F172A;">{city_name}, Pakistan</div>
        <div style="font-size: 11.5px; font-family: 'JetBrains Mono', monospace; color: #64748B; margin-top: 3px;">{CITY_LAT}° N, {CITY_LON}° E • 527m ASL</div>
    </div>
    """, unsafe_allow_html=True)

    # Action Trigger with Verified Contrast
    if st.button("⚡ Run Forecast Cycle", use_container_width=True, type="primary"):
        with st.spinner("Executing live multi-source ingestion & 4-node agent cycle..."):
            live_df = ingest_live_row(city=city_name)
            eng_df = build_engineered_features(live_df)
            new_state = run_agent_pipeline(eng_df)
            st.cache_data.clear()
            st.success("Hourly cycle completed successfully.")
            st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)

    # Pipeline Node Stepper (Compact Single-Glance)
    st.markdown("<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 10px;'>Pipeline Nodes</div>", unsafe_allow_html=True)
    stepper_nodes = [
        ("Forecast Agent", "8 Dedicated Model Champions", "#16A34A"),
        ("Alert Agent", "Groq Safety Rules & Advisories", "#16A34A"),
        ("Drift Agent", "Two-Sample KS & Error Tracking", "#0284C7"),
        ("Report Agent", "Journalistic Broadcast Dispatch", "#16A34A"),
    ]
    for name, detail, col in stepper_nodes:
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 9px; margin-bottom: 10px;">
            <div style="width: 7px; height: 7px; border-radius: 50%; background: {col}; margin-top: 5px; flex-shrink: 0;"></div>
            <div>
                <div style="font-size: 12.5px; font-weight: 600; color: #0F172A; line-height: 1.2;">{name}</div>
                <div style="font-size: 11px; color: #64748B; margin-top: 1px;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)

    # Understated Metadata Tech Stack Footer
    st.markdown("""
    <div style="font-size: 11px; color: #94A3B8; line-height: 1.6;">
        LangGraph • Hopsworks • Groq • GitHub Actions
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header Bar with Atmospheric Snapshot ────────────────────────────────
st.markdown(f"""
<div class="hero-header-bar">
    <div class="hero-title-group">
        <div class="eyebrow">
            <span>●</span>
            <span>Atmospheric Intelligence & ML Observability</span>
        </div>
        <div class="main-title">{city_name} 24-Hour Weather & Air Quality Outlook</div>
    </div>
    <div class="hero-meta-group">
        <span class="meta-pill">Target: T+24h</span>
        <span class="meta-pill">Updated: {cycle_timestamp} UTC</span>
        <span class="meta-pill" style="color: {overall_status_col}; background: {overall_status_bg}; border-color: {overall_status_bdr}; font-weight: 600;">
            {overall_status_text}
        </span>
    </div>
    <div class="hero-snapshot-strip">
        <span class="snapshot-item">☀️ Clear & Dry Conditions</span>
        <span class="snapshot-item">📊 Barometric Pressure: {latest_telemetry['pressure']}</span>
        <span class="snapshot-item">☁️ Cloud Cover: {latest_telemetry['cloud_cover']}</span>
        <span class="snapshot-item">🏔️ Station Elevation: 527m ASL</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Section 1: 24-Hour Predictive Metrics Grid (2 Rows x 4 Columns) ──────────
st.markdown('<div class="section-headline">24-Hour Predictive Forecast Horizon (T+24h)</div>', unsafe_allow_html=True)

keys_row1 = ["aqi", "temp_c", "precip_mm", "rain_prob"]
keys_row2 = ["humidity_pct", "uv_index", "wind_speed", "wind_dir"]

cols_r1 = st.columns(4)
for col, key in zip(cols_r1, keys_row1):
    cfg = METRIC_REGISTRY[key]
    val = float(forecasts.get(key, 0.0))
    status_label, text_col, bg_col, bdr_col, note = cfg["eval_func"](val)
    gauge_markup = cfg["gauge_html"](val)

    with col:
        st.markdown(f"""
        <div class="metric-card-console">
            <div class="card-accent-bar" style="background: {text_col};"></div>
            <div class="metric-header">
                <div class="metric-name-group">
                    <span class="metric-icon">{cfg['icon']}</span>
                    <span class="metric-name">{cfg['name']}</span>
                </div>
                <span class="metric-status-badge" style="color: {text_col}; background: {bg_col}; border-color: {bdr_col};">
                    <span class="metric-status-dot" style="background: {text_col};"></span>
                    {status_label}
                </span>
            </div>
            <div class="metric-value-row">
                <span class="metric-number">{val:.1f}</span>
                <span class="metric-unit">{cfg['unit']}</span>
            </div>
            {gauge_markup}
            <div class="metric-footer" title="{note}">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

cols_r2 = st.columns(4)
for col, key in zip(cols_r2, keys_row2):
    cfg = METRIC_REGISTRY[key]
    val = float(forecasts.get(key, 0.0))
    status_label, text_col, bg_col, bdr_col, note = cfg["eval_func"](val)
    gauge_markup = cfg["gauge_html"](val)

    # Special format for wind direction (show cardinal + degrees)
    if key == "wind_dir":
        display_number = f"{round(val)}°"
        display_unit = degrees_to_cardinal(val)
    else:
        display_number = f"{val:.1f}"
        display_unit = cfg["unit"]

    with col:
        st.markdown(f"""
        <div class="metric-card-console">
            <div class="card-accent-bar" style="background: {text_col};"></div>
            <div class="metric-header">
                <div class="metric-name-group">
                    <span class="metric-icon">{cfg['icon']}</span>
                    <span class="metric-name">{cfg['name']}</span>
                </div>
                <span class="metric-status-badge" style="color: {text_col}; background: {bg_col}; border-color: {bdr_col};">
                    <span class="metric-status-dot" style="background: {text_col};"></span>
                    {status_label}
                </span>
            </div>
            <div class="metric-value-row">
                <span class="metric-number">{display_number}</span>
                <span class="metric-unit">{display_unit}</span>
            </div>
            {gauge_markup}
            <div class="metric-footer" title="{note}">{note}</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Section 2: Meteorological Briefing & Audience Advisories ─────────────────
st.markdown('<div class="section-headline">Meteorological Assessment & Safety Advisories</div>', unsafe_allow_html=True)

col_briefing, col_advisory = st.columns([3, 2], gap="medium")

with col_briefing:
    temp_val = float(forecasts.get('temp_c', 24.8))
    hum_val = float(forecasts.get('humidity_pct', 24.6))
    aqi_val = float(forecasts.get('aqi', 30.9))
    rain_val = float(forecasts.get('rain_prob', 22.3))

    st.markdown(f"""
    <div class="console-card-panel">
        <div class="card-title-row">
            <span class="card-main-title">Official Meteorological Briefing</span>
            <span class="card-subtitle-tag">Synthesized Broadcast</span>
        </div>
        <div class="briefing-callout">
            <div class="briefing-highlight-text">
                Mild daytime warmth with exceptionally clear skies and low relative humidity expected across the metropolitan capital region over the next 24 hours.
            </div>
            <div class="briefing-ribbon">
                <span class="ribbon-chip">🌡️ Peak Temp: {temp_val:.1f}°C</span>
                <span class="ribbon-chip">💧 Humidity: {hum_val:.1f}% (Dry Air)</span>
                <span class="ribbon-chip">🍃 Air Quality: {aqi_val:.1f} AQI (Good)</span>
                <span class="ribbon-chip">☔ Rain Risk: {rain_val:.1f}% (Low)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    report_text = agent_state.get("weather_report", "Weather briefing currently in compilation.")
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
    paragraphs_html = "".join([f"<p>{p}</p>" for p in paragraphs])

    st.markdown(f"""
        <div class="briefing-body-text">
            {paragraphs_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_advisory:
    st.markdown("""
    <div class="console-card-panel">
        <div class="card-title-row">
            <span class="card-main-title">Audience Safety Advisories</span>
            <span class="card-subtitle-tag">Municipal Guidelines</span>
        </div>
    """, unsafe_allow_html=True)

    if active_alerts:
        flagged_summary = ", ".join([a["metric"].upper() for a in active_alerts])
        st.markdown(f"""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12.5px; color: #DC2626;">
            <b>Active Hazard Advisory:</b> {len(active_alerts)} metric(s) exceed normal thresholds ({flagged_summary}). Review specialized protocols below.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12.5px; color: #166534;">
            <b>Status Green:</b> All 8 predicted atmospheric and environmental parameters remain within safe baseline thresholds.
        </div>
        """, unsafe_allow_html=True)

    tab_public, tab_sensitive, tab_outdoor = st.tabs(["General Public", "Sensitive Groups", "Outdoor Activity"])

    with tab_public:
        render_advisory_section(advisories.get("general", ""))
    with tab_sensitive:
        render_advisory_section(advisories.get("sensitive", ""))
    with tab_outdoor:
        render_advisory_section(advisories.get("outdoor", ""))

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Section 3: Interactive Atmospheric Trend Visualizer ──────────────────────
st.markdown('<div class="section-headline">Atmospheric Telemetry & 24-Hour Predictive Trajectory</div>', unsafe_allow_html=True)

CHART_METRIC_MAP = {
    "temp_c":       "Temperature (°C)",
    "aqi":          "Air Quality Index (AQI)",
    "humidity_pct": "Relative Humidity (%)",
    "wind_speed":   "Wind Speed (km/h)",
    "precip_mm":    "Precipitation (mm)",
    "uv_index":     "UV Index",
}

if hist_df is not None and not hist_df.empty:
    chart_col, controls_col = st.columns([5, 1])
    with controls_col:
        chart_metric = st.selectbox(
            "Telemetry Metric",
            options=list(CHART_METRIC_MAP.keys()),
            format_func=lambda x: CHART_METRIC_MAP[x],
        )
        lookback_hours = st.selectbox(
            "Lookback Window",
            options=[48, 72, 168],
            format_func=lambda x: f"{x} Hours" if x < 168 else "7 Days",
            index=2,
        )

    with chart_col:
        history_window = hist_df.iloc[-lookback_hours:].copy()

        fig = go.Figure()

        # Observed Area Fill Line (Tomorrow.io Aesthetic)
        fig.add_trace(go.Scatter(
            x=history_window["timestamp"],
            y=history_window[chart_metric],
            mode="lines",
            name="Observed History",
            line=dict(color="#0284C7", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.05)",
            hovertemplate="%{y:.2f}<extra>Observed</extra>",
        ))

        # Prediction Point Horizon
        last_time = history_window["timestamp"].iloc[-1]
        pred_time = last_time + pd.Timedelta(hours=24)
        pred_val = forecasts.get(chart_metric, history_window[chart_metric].iloc[-1])

        fig.add_trace(go.Scatter(
            x=[last_time, pred_time],
            y=[history_window[chart_metric].iloc[-1], pred_val],
            mode="lines+markers",
            name="24h Prediction Horizon",
            line=dict(color="#F59E0B", width=2.2, dash="dot"),
            marker=dict(size=[5, 9], color=["#0284C7", "#F59E0B"], symbol=["circle", "star"]),
            hovertemplate="%{y:.2f}<extra>Forecast T+24h</extra>",
        ))

        # Hazard Threshold Reference Lines
        hazard_rule = HAZARD_THRESHOLDS.get(chart_metric, {})
        if hazard_rule.get("max") is not None:
            fig.add_hline(
                y=hazard_rule["max"],
                line_dash="dot",
                line_color="#DC2626",
                line_width=1.0,
                annotation_text=f"Hazard Upper Limit: {hazard_rule['max']}",
                annotation_font_size=10,
                annotation_font_color="#DC2626",
                annotation_position="top right",
            )
        if hazard_rule.get("min") is not None:
            fig.add_hline(
                y=hazard_rule["min"],
                line_dash="dot",
                line_color="#D97706",
                line_width=1.0,
                annotation_text=f"Hazard Lower Limit: {hazard_rule['min']}",
                annotation_font_size=10,
                annotation_font_color="#D97706",
                annotation_position="bottom right",
            )

        metric_title = CHART_METRIC_MAP[chart_metric]
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=340,
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A", size=12),
            title=dict(
                text=f"{metric_title} — {lookback_hours}h Telemetry with Forward T+24h Trajectory",
                font=dict(size=13, color="#0F172A", weight=600),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Timestamp (UTC)",
                showgrid=True,
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=metric_title,
                showgrid=True,
                gridcolor="#F1F5F9",
                linecolor="#E2E8F0",
                tickfont=dict(size=11),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )

        st.markdown('<div class="console-card-panel" style="padding: 16px 20px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Section 4: ML Observability & Governance Ledger ──────────────────────────
st.markdown('<div class="section-headline">ML Model Observability & Governance Ledger</div>', unsafe_allow_html=True)

tab_ledger, tab_shap, tab_drift, tab_audit = st.tabs([
    "Forecast Ledger",
    "EDA & SHAP Interpretability",
    "Drift Surveillance",
    "LLM Audit Log",
])

# ── Tab 1: Forecast Ledger ──
with tab_ledger:
    st.markdown("""
    <div class="tab-summary-ribbon">
        <span class="tab-summary-item">Deployed Champions: <b>8 Models</b></span>
        <span style="color: #CBD5E1;">•</span>
        <span class="tab-summary-item">Model Registry Status: <b>Promoted (v1)</b></span>
        <span style="color: #CBD5E1;">•</span>
        <span class="tab-summary-item">Chronological Split: <b>80% Train, 10% Val, 10% Test</b></span>
        <span style="color: #CBD5E1;">•</span>
        <span class="tab-summary-item">Inference Horizon: <b>T+24h Ahead</b></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;">
        Recent Inference Ledger Entries
    </div>
    """, unsafe_allow_html=True)

    ledger_entries = agent_state.get("forecast_ledger_entries", [])
    if ledger_entries:
        ledger_table = []
        for entry in ledger_entries:
            raw_metric = entry.get("metric", "")
            cfg = METRIC_REGISTRY.get(raw_metric, {})
            human_name = cfg.get("name", raw_metric.upper())
            unit = cfg.get("unit", "")
            val = entry.get("predicted_value")
            val_str = f"{val:.2f} {unit}".strip() if val is not None else "N/A"

            ledger_table.append({
                "Target Metric": human_name,
                "Predicted Value": val_str,
                "Champion Model Family": entry.get("model_family", "N/A"),
                "Horizon": "T+24h",
                "Target Timestamp": entry.get("target_timestamp", "N/A"),
                "Inference Timestamp": entry.get("forecast_created_at", "N/A"),
            })
        st.dataframe(pd.DataFrame(ledger_table), use_container_width=True, hide_index=True)
    else:
        st.info("No prediction ledger entries logged in the active session.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;">
        Champion Model Registry Validation Benchmarks
    </div>
    """, unsafe_allow_html=True)

    if training_meta:
        perf_rows = []
        for metric_key, meta in training_meta.items():
            cfg = METRIC_REGISTRY.get(metric_key, {})
            human_name = cfg.get("name", metric_key)
            family = meta.get("champion_family", "N/A")
            val_m = meta.get("validation_metrics", {})
            rmse = val_m.get("rmse")
            mae = val_m.get("mae")
            r2 = val_m.get("r2")
            imprv = meta.get("improvement_pct")

            perf_rows.append({
                "Target Metric": human_name,
                "Champion Family": family,
                "Validation RMSE": f"{rmse:.4f}" if rmse is not None else "N/A",
                "Validation MAE": f"{mae:.4f}" if mae is not None else "N/A",
                "Validation R²": f"{r2:.4f}" if r2 is not None else "N/A",
                "Improvement vs Naive": f"+{imprv:.2f}%" if imprv is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; font-size: 13px; color: #64748B;">
            <b>Model Registry Notice:</b> Validation benchmark metrics are compiled during scheduled training runs. Champion model weights are active and serving inference.
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: EDA & SHAP Interpretability ──
with tab_shap:
    shap_col, eda_col = st.columns([1, 1], gap="medium")

    with shap_col:
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>SHAP Global Feature Importance</div>", unsafe_allow_html=True)
        selected_shap = st.selectbox(
            "Select Target Variable",
            options=TARGET_METRICS,
            format_func=lambda x: METRIC_REGISTRY.get(x, {}).get("name", x),
        )
        shap_file = EDA_ARTIFACTS_DIR / f"shap_importance_{selected_shap}.png"
        if shap_file.exists():
            st.image(str(shap_file), caption=f"SHAP Global Feature Importance — {METRIC_REGISTRY.get(selected_shap, {}).get('name', selected_shap)}", use_container_width=True)
        else:
            st.info(f"SHAP artifact for {selected_shap} generated during daily training.")

    with eda_col:
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Baseline Target Distributions</div>", unsafe_allow_html=True)
        dist_img = EDA_ARTIFACTS_DIR / "eda_metric_distributions.png"
        if dist_img.exists():
            st.image(str(dist_img), caption="Historical Observed Distributions Across Target Categories", use_container_width=True)

        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-top: 14px; margin-bottom: 8px;'>Diurnal 24-Hour Seasonality</div>", unsafe_allow_html=True)
        diurnal_img = EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png"
        if diurnal_img.exists():
            st.image(str(diurnal_img), caption="Diurnal Day & Night Seasonality Profiles", use_container_width=True)

# ── Tab 3: Drift Surveillance ──
with tab_drift:
    drift_data = agent_state.get("drift_results", {})
    drift_detected = drift_data.get("overall_drift_detected", False)

    if drift_detected:
        st.markdown("""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #DC2626;">
            <b>Statistical Drift Detected:</b> Automated continuous retraining workflow triggered via GitHub Actions dispatch.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #166534;">
            <b>Nominal Conditions:</b> Two-sample Kolmogorov-Smirnov p-values > 0.05. Model forecast error degradation remains within the 25% threshold.
        </div>
        """, unsafe_allow_html=True)

    ks_col, md_col = st.columns(2, gap="medium")

    with ks_col:
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Two-Sample Kolmogorov-Smirnov (KS) Data Drift Test</div>", unsafe_allow_html=True)
        ks_details = drift_data.get("data_drift", {}).get("details", {})
        if ks_details:
            ks_rows = []
            for feat, res in ks_details.items():
                cfg = METRIC_REGISTRY.get(feat, {})
                human_name = cfg.get("name", feat)
                is_drift = res.get("drift_detected", False)
                ks_rows.append({
                    "Feature": human_name,
                    "KS Statistic": f"{res.get('ks_statistic', 0):.4f}",
                    "P-Value": f"{res.get('p_value', 1):.4f}",
                    "Status": "Drift Detected" if is_drift else "Stable",
                })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating feature rows for Kolmogorov-Smirnov distribution check.")

    with md_col:
        st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Model Forecast Error Degradation (Model Drift)</div>", unsafe_allow_html=True)
        md_metrics = drift_data.get("model_drift", {}).get("metrics", {})
        if md_metrics:
            md_rows = []
            for metric, res in md_metrics.items():
                cfg = METRIC_REGISTRY.get(metric, {})
                human_name = cfg.get("name", metric)
                is_drift = res.get("drift_detected", False)
                md_rows.append({
                    "Target Metric": human_name,
                    "Recent MAE": f"{res.get('recent_mae', 0):.4f}",
                    "Baseline MAE": f"{res.get('baseline_mae', 0):.4f}",
                    "Degradation": f"{res.get('pct_degradation', 0):.1f}%",
                    "Status": "Degraded" if is_drift else "Accurate",
                })
            st.dataframe(pd.DataFrame(md_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating 24-hour prediction-to-observation pairs for model degradation tracking.")

# ── Tab 4: LLM Audit Log ──
with tab_audit:
    st.markdown("<div style='font-size: 12.5px; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Groq LLM Inference & Failure Audit Ledger</div>", unsafe_allow_html=True)
    calls = agent_state.get("audit_call_logs", [])
    filter_choice = st.selectbox("Log Filter", options=["All Inferences", "Successful Inferences Only", "Failed Inferences Only"])

    if calls:
        df_calls = pd.DataFrame(calls)
        if filter_choice == "Failed Inferences Only":
            df_calls = df_calls[df_calls["status"] == "failed"]
        elif filter_choice == "Successful Inferences Only":
            df_calls = df_calls[df_calls["status"] == "success"]

        if df_calls.empty:
            st.info("No inference logs match the selected filter.")
        else:
            for idx, row in df_calls.iterrows():
                status_val = str(row.get("status", "")).upper()
                agent_name = row.get("agent_name", "UnknownAgent")
                ts_val = row.get("timestamp", "N/A")
                latency = row.get("latency_ms", 0)

                status_col = "#16A34A" if status_val == "SUCCESS" else "#DC2626"
                expander_label = f"[{status_val}] {agent_name} • {ts_val} • {latency} ms"

                with st.expander(expander_label):
                    st.markdown(f"<b>Agent Node:</b> {agent_name} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Status:</b> <span style='color:{status_col}; font-weight:600;'>{status_val}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <b>Latency:</b> {latency} ms", unsafe_allow_html=True)
                    if row.get("error_message"):
                        st.error(f"Error: {row.get('error_message')}")
                    st.text_area("Submitted System & User Prompt", value=str(row.get("prompt", "")), height=90, key=f"p_{idx}", disabled=True)
                    st.text_area("Generated Output Completion", value=str(row.get("response", "")), height=130, key=f"r_{idx}", disabled=True)
    else:
        st.info("No LLM inference calls recorded in the current session.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="console-footer">
    <span>AgentWeather • Meteorological Intelligence & ML Observability Platform</span>
    <span>Target Station: {city_name}, Pakistan ({CITY_LAT}° N, {CITY_LON}° E)</span>
</div>
""", unsafe_allow_html=True)
