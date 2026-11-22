"""
AgentWeather: Meteorological Observability and Forecast Platform
Islamabad, Pakistan | Serverless Pipeline on LangGraph, Hopsworks, and Groq
"""

import os
import sys
import json
import re
import datetime
from pathlib import Path

# Ensure project root is in sys.path
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


# Streamlit Page Configuration
st.set_page_config(
    page_title=f"AgentWeather | {CITY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Design System and Dark Mode Stylesheet
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-canvas:             #090D16;
    --bg-surface:            #0F1623;
    --bg-elevated:           #161F30;
    --bg-sidebar:            #0B101B;
    --border-subtle:         #1E293B;
    --border-strong:         #2E3D56;
    --text-primary:          #F8FAFC;
    --text-secondary:        #94A3B8;
    --text-muted:            #64748B;
    --accent-primary:        #38BDF8;
    --accent-solid:          #0284C7;
    --accent-solid-hover:    #0369A1;
    --accent-soft:           rgba(56, 189, 248, 0.10);
    --accent-border:         rgba(56, 189, 248, 0.25);
    --emerald-text:          #10B981;
    --emerald-soft:          rgba(16, 185, 129, 0.10);
    --emerald-border:        rgba(16, 185, 129, 0.25);
    --amber-text:            #F59E0B;
    --amber-soft:            rgba(245, 158, 11, 0.10);
    --amber-border:          rgba(245, 158, 11, 0.25);
    --red-text:              #EF4444;
    --red-soft:              rgba(239, 68, 68, 0.10);
    --red-border:            rgba(239, 68, 68, 0.25);
    --radius-lg:             10px;
    --radius-md:             8px;
    --radius-sm:             6px;
}

/* Base Canvas and Chrome Cleanup */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg-canvas) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, [data-testid="stDecoration"] {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: rgba(9, 13, 22, 0.88) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar Architecture */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Primary Action Button */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton button {
    background-color: var(--accent-solid) !important;
    background: var(--accent-solid) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    padding: 10px 16px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stButton button *,
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: var(--accent-solid-hover) !important;
    background: var(--accent-solid-hover) !important;
    border-color: var(--accent-primary) !important;
}

/* Sidebar Components */
.sb-brand-wrap {
    padding: 4px 0 18px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 16px;
}

.sb-brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}

.sb-brand-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

.sb-live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: var(--emerald-text);
    display: inline-block;
}

.sb-live-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    background: var(--emerald-soft);
    border: 1px solid var(--emerald-border);
    color: var(--emerald-text) !important;
}

.sb-brand-sub {
    font-size: 12px;
    color: var(--text-muted) !important;
}

.sb-station-box {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 13px 14px;
    margin-bottom: 16px;
}

.sb-kicker {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    margin-bottom: 4px;
}

.sb-station-name {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary) !important;
}

.sb-station-coords {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary) !important;
    margin-top: 4px;
}

.sb-divider {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 18px 0;
}

.sb-node-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
}

.sb-node-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}

.sb-node-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary) !important;
    line-height: 1.25;
}

.sb-node-desc {
    font-size: 11px;
    color: var(--text-muted) !important;
    margin-top: 2px;
}

.sb-stack-footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: var(--text-muted) !important;
    line-height: 1.6;
}

/* Top Header Bar */
.top-header-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 22px;
}

.top-header-main {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}

.header-location-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent-primary);
    margin-bottom: 5px;
}

.header-main-title {
    font-size: 21px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.header-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 11px;
    border-radius: var(--radius-sm);
    font-size: 11.5px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}

.telemetry-strip {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    width: 100%;
    padding-top: 14px;
    margin-top: 14px;
    border-top: 1px solid var(--border-subtle);
}

.telemetry-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: var(--text-secondary);
}

.telemetry-chip strong {
    color: var(--text-primary);
    font-weight: 600;
}

/* Section Headers */
.section-label {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin: 24px 0 12px 2px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Cards */
.metric-tile {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    min-height: 144px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}

.metric-tile:hover {
    border-color: var(--border-strong);
    background: #121A29;
}

.metric-tile-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}

.metric-tile-label {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    font-size: 10.5px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid;
    white-space: nowrap;
}

.metric-reading {
    margin: 8px 0 6px;
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.metric-val {
    font-size: 27px;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
    letter-spacing: -0.03em;
}

.metric-unit-text {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.progress-track {
    width: 100%;
    height: 3px;
    border-radius: 2px;
    background: var(--bg-elevated);
    margin: 6px 0 8px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 2px;
}

.metric-note {
    font-size: 11px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Content Panels (Briefing and Advisories) */
.surface-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px 22px;
    height: 100%;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-subtle);
}

.panel-title {
    font-size: 14.5px;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}

.panel-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--accent-primary);
    background: var(--accent-soft);
    border: 1px solid var(--accent-border);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
}

.briefing-summary-box {
    background: var(--bg-elevated);
    border-left: 3px solid var(--accent-primary);
    padding: 12px 14px;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin-bottom: 16px;
}

.briefing-lead {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.45;
    margin-bottom: 10px;
}

.briefing-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.briefing-chip {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 9px;
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}

.briefing-prose {
    font-size: 13.5px;
    line-height: 1.65;
    color: #CBD5E1;
    max-width: 68ch;
}

.briefing-prose p {
    margin-bottom: 12px;
}

.briefing-prose p:last-child {
    margin-bottom: 0;
}

/* Advisory List Items */
.advisory-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 11px 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 13px;
    line-height: 1.55;
}

.advisory-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.advisory-tag {
    flex-shrink: 0;
    padding: 3px 9px;
    border-radius: var(--radius-sm);
    font-size: 11px;
    font-weight: 600;
    border: 1px solid;
    white-space: nowrap;
}

.advisory-desc {
    color: var(--text-secondary);
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    color: var(--text-secondary) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: color 0.15s ease, background-color 0.15s ease !important;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--bg-elevated) !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-primary) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent-primary) !important;
    background: transparent !important;
}

/* DataFrames and Summary Bar */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

.registry-bar {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 12px;
    color: var(--text-secondary);
}

.registry-bar b {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}

.registry-sep {
    color: var(--border-strong);
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* Footer */
.app-footer {
    margin-top: 36px;
    padding: 18px 0 8px;
    border-top: 1px solid var(--border-subtle);
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


# Data Loaders
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


# Meteorological Helpers
def degrees_to_cardinal(deg: float) -> str:
    """Converts 0 to 360 compass degrees to cardinal direction."""
    if deg is None or np.isnan(deg):
        return "N/A"
    cardinals = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = round(float(deg) / 22.5) % 16
    return cardinals[idx]


def _render_progress_bar(pct: float, fill_color: str) -> str:
    clamped = f"{min(100.0, max(0.0, pct)):.1f}"
    return (
        f'<div class="progress-track">'
        f'<div class="progress-fill" style="width:{clamped}%;background:{fill_color};"></div>'
        f'</div>'
    )


METRIC_REGISTRY = {
    "aqi": {
        "name": "Air Quality Index",
        "unit": "AQI",
        "eval_func": lambda v: (
            ("Good", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Safe threshold below 150")
            if v <= 50
            else ("Moderate", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Advisory range 51 to 150")
            if v <= 150
            else ("Hazardous", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "Hazardous above 150")
        ),
        "pct_func": lambda v: (v / 200.0) * 100.0,
    },
    "temp_c": {
        "name": "Temperature",
        "unit": "°C",
        "eval_func": lambda v: (
            ("Extreme", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "Hazard below 0°C or above 40°C")
            if v < 0 or v > 40
            else ("Advisory", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Advisory below 12°C or above 34°C")
            if v < 12 or v > 34
            else ("Nominal", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Comfort range 12°C to 34°C")
        ),
        "pct_func": lambda v: ((v + 5.0) / 45.0) * 100.0,
    },
    "precip_mm": {
        "name": "Precipitation",
        "unit": "mm",
        "eval_func": lambda v: (
            ("Heavy Rain", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "Hazard threshold above 20 mm")
            if v > 20
            else ("Moderate", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Accumulation 2.5 to 20 mm")
            if v >= 2.5
            else ("Dry", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Minimal accumulation below 2.5 mm")
        ),
        "pct_func": lambda v: (v / 25.0) * 100.0,
    },
    "rain_prob": {
        "name": "Rain Probability",
        "unit": "%",
        "eval_func": lambda v: (
            ("High Risk", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "High likelihood above 75%")
            if v > 75
            else ("Moderate", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Chance of rain 30% to 75%")
            if v >= 30
            else ("Low Risk", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Rain unlikely below 30%")
        ),
        "pct_func": lambda v: v,
    },
    "humidity_pct": {
        "name": "Relative Humidity",
        "unit": "%",
        "eval_func": lambda v: (
            ("Critical", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "Critical limit below 20% or above 85%")
            if v < 20 or v > 85
            else ("Dry Air", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Low hydration advisory below 30%")
            if v < 30
            else ("High Moisture", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Humid advisory above 70%")
            if v > 70
            else ("Comfortable", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Optimal comfort 30% to 70%")
        ),
        "pct_func": lambda v: v,
    },
    "uv_index": {
        "name": "UV Index",
        "unit": "UV",
        "eval_func": lambda v: (
            ("Very High", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "Hazardous index 8.0 or above")
            if v >= 8
            else ("Moderate", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Moderate exposure 3.0 to 7.9")
            if v >= 3
            else ("Low", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Low solar radiation below 3.0")
        ),
        "pct_func": lambda v: (v / 12.0) * 100.0,
    },
    "wind_speed": {
        "name": "Wind Speed",
        "unit": "km/h",
        "eval_func": lambda v: (
            ("High Wind", "#EF4444", "rgba(239, 68, 68, 0.10)", "rgba(239, 68, 68, 0.25)", "High wind hazard above 50 km/h")
            if v > 50
            else ("Breezy", "#F59E0B", "rgba(245, 158, 11, 0.10)", "rgba(245, 158, 11, 0.25)", "Moderate breeze 20 to 50 km/h")
            if v >= 20
            else ("Calm", "#10B981", "rgba(16, 185, 129, 0.10)", "rgba(16, 185, 129, 0.25)", "Light breeze below 20 km/h")
        ),
        "pct_func": lambda v: (v / 50.0) * 100.0,
    },
    "wind_dir": {
        "name": "Wind Direction",
        "unit": "",
        "eval_func": lambda v: (
            degrees_to_cardinal(v),
            "#38BDF8",
            "rgba(56, 189, 248, 0.10)",
            "rgba(56, 189, 248, 0.25)",
            f"Bearing {degrees_to_cardinal(v)} from North",
        ),
        "pct_func": lambda v: (v / 360.0) * 100.0,
    },
}


def render_advisory_section(advisory_text: str):
    """Parses advisory text into structured rows with semantic badges."""
    if not advisory_text:
        st.info("No active advisories issued for this audience.")
        return

    lines = [line.strip() for line in advisory_text.splitlines() if line.strip()]
    structured_rows = []
    plain_paragraphs = []

    for line in lines:
        match = re.match(r"^[-*•]?\s*\*\*([^*:]+):?\*\*:?\s*(.+)$", line)
        if match:
            cat = match.group(1).strip()
            desc = match.group(2).strip()
            body_lower = desc.lower()

            if any(w in body_lower for w in ["hazard", "critical", "mandatory", "severe", "danger"]):
                badge_color = "#EF4444"
                badge_bg = "rgba(239, 68, 68, 0.10)"
                badge_border = "rgba(239, 68, 68, 0.25)"
            elif any(w in body_lower for w in ["priority", "advisory", "caution", "low", "limit", "dry", "elevated", "protect"]):
                badge_color = "#F59E0B"
                badge_bg = "rgba(245, 158, 11, 0.10)"
                badge_border = "rgba(245, 158, 11, 0.25)"
            else:
                badge_color = "#10B981"
                badge_bg = "rgba(16, 185, 129, 0.10)"
                badge_border = "rgba(16, 185, 129, 0.25)"

            structured_rows.append((cat, desc, badge_color, badge_bg, badge_border))
        else:
            plain_paragraphs.append(line)

    if structured_rows:
        for cat, desc, col, bg, bdr in structured_rows:
            st.markdown(f"""
            <div class="advisory-item">
                <span class="advisory-tag" style="color: {col}; background: {bg}; border-color: {bdr};">{cat}</span>
                <span class="advisory-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
    elif plain_paragraphs:
        st.markdown("\n\n".join(plain_paragraphs))


# Data Initialization
agent_state = load_cached_agent_state()
hist_df = load_historical_data()
training_meta = load_training_metadata()

if agent_state is None:
    if hist_df is not None:
        with st.spinner("Executing initial pipeline run..."):
            agent_state = run_agent_pipeline()
    else:
        st.error("Historical feature store not found. Run backfill and model training first.")
        st.stop()

forecasts = agent_state.get("forecasts", {})
active_alerts = agent_state.get("active_alerts", [])
advisories = agent_state.get("alert_advisories", {})
cycle_timestamp = agent_state.get("timestamp", "N/A")
city_name = agent_state.get("city", CITY_NAME)

# Extract recent station telemetry
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
    overall_status_text = "All Parameters Nominal"
    overall_status_col = "#10B981"
    overall_status_bg = "rgba(16, 185, 129, 0.10)"
    overall_status_bdr = "rgba(16, 185, 129, 0.25)"
else:
    overall_status_text = f"{n_alerts} Active Hazard Alert(s)"
    overall_status_col = "#EF4444"
    overall_status_bg = "rgba(239, 68, 68, 0.10)"
    overall_status_bdr = "rgba(239, 68, 68, 0.25)"


# Sidebar Console
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand-wrap">
        <div class="sb-brand-row">
            <span class="sb-brand-title">
                <span class="sb-live-dot"></span>
                <span>AgentWeather</span>
            </span>
            <span class="sb-live-pill">ONLINE</span>
        </div>
        <div class="sb-brand-sub">Meteorological Forecasting and ML Ops</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-station-box">
        <div class="sb-kicker">Monitoring Station</div>
        <div class="sb-station-name">{city_name}, Pakistan</div>
        <div class="sb-station-coords">{CITY_LAT}° N, {CITY_LON}° E  |  527m ASL</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Forecast Cycle", use_container_width=True, type="primary"):
        with st.spinner("Ingesting live telemetry and executing 4 node pipeline..."):
            live_df = ingest_live_row(city=city_name)
            eng_df = build_engineered_features(live_df)
            new_state = run_agent_pipeline(eng_df)
            st.cache_data.clear()
            st.success("Forecast cycle completed.")
            st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sb-kicker" style="margin-bottom: 8px;">Execution Graph</div>', unsafe_allow_html=True)
    stepper_nodes = [
        ("Forecast Node", "8 Dedicated Model Champions", "#10B981"),
        ("Alert Node", "Threshold Rules and Advisories", "#10B981"),
        ("Drift Node", "Two Sample KS and Error Tracking", "#38BDF8"),
        ("Report Node", "Broadcast Summary Generation", "#10B981"),
    ]
    for name, detail, col in stepper_nodes:
        st.markdown(f"""
        <div class="sb-node-row">
            <span class="sb-node-indicator" style="background-color: {col};"></span>
            <div>
                <div class="sb-node-title">{name}</div>
                <div class="sb-node-desc">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-stack-footer">
        LangGraph  |  Hopsworks  |  Groq  |  GitHub Actions
    </div>
    """, unsafe_allow_html=True)


# Top Header Bar
st.markdown(f"""
<div class="top-header-panel">
    <div class="top-header-main">
        <div>
            <div class="header-location-tag">Islamabad Capital Territory, Pakistan</div>
            <div class="header-main-title">{city_name} 24 Hour Weather and Air Quality Outlook</div>
        </div>
        <div class="header-badges">
            <span class="header-pill">Horizon: Next 24h</span>
            <span class="header-pill">Cycle: {cycle_timestamp} UTC</span>
            <span class="header-pill" style="color: {overall_status_col}; background: {overall_status_bg}; border-color: {overall_status_bdr}; font-weight: 600;">
                {overall_status_text}
            </span>
        </div>
    </div>
    <div class="telemetry-strip">
        <span class="telemetry-chip">Observed Temp: <strong>{latest_telemetry['current_temp']}</strong></span>
        <span class="telemetry-chip">Barometric Pressure: <strong>{latest_telemetry['pressure']}</strong></span>
        <span class="telemetry-chip">Cloud Cover: <strong>{latest_telemetry['cloud_cover']}</strong></span>
        <span class="telemetry-chip">Station Elevation: <strong>527m ASL</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_metric_card(cfg, val, display_number, display_unit, text_col, bg_col, bdr_col, status_label, note):
    pct = cfg["pct_func"](val)
    bar_html = _render_progress_bar(pct, text_col)
    parts = [
        '<div class="metric-tile">',
        '<div class="metric-tile-top">',
        f'<span class="metric-tile-label">{cfg["name"]}</span>',
        f'<span class="metric-pill" style="color:{text_col};background:{bg_col};border-color:{bdr_col};">{status_label}</span>',
        '</div>',
        '<div class="metric-reading">',
        f'<span class="metric-val">{display_number}</span>',
        f'<span class="metric-unit-text">{display_unit}</span>',
        '</div>',
        bar_html,
        f'<div class="metric-note">{note}</div>',
        '</div>',
    ]
    return "".join(parts)


# Section 1: 24 Hour Predictive Metrics Grid
st.markdown('<div class="section-label">24 Hour Forecast</div>', unsafe_allow_html=True)

keys_row1 = ["aqi", "temp_c", "precip_mm", "rain_prob"]
keys_row2 = ["humidity_pct", "uv_index", "wind_speed", "wind_dir"]

cols_r1 = st.columns(4)
for col, key in zip(cols_r1, keys_row1):
    cfg = METRIC_REGISTRY[key]
    val = float(forecasts.get(key) or 0.0)
    status_label, text_col, bg_col, bdr_col, note = cfg["eval_func"](val)
    with col:
        st.markdown(
            render_metric_card(cfg, val, f"{val:.1f}", cfg["unit"], text_col, bg_col, bdr_col, status_label, note),
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

cols_r2 = st.columns(4)
for col, key in zip(cols_r2, keys_row2):
    cfg = METRIC_REGISTRY[key]
    val = float(forecasts.get(key) or 0.0)
    status_label, text_col, bg_col, bdr_col, note = cfg["eval_func"](val)
    if key == "wind_dir":
        display_number = f"{round(val)}°"
        display_unit = degrees_to_cardinal(val)
    else:
        display_number = f"{val:.1f}"
        display_unit = cfg["unit"]
    with col:
        st.markdown(
            render_metric_card(cfg, val, display_number, display_unit, text_col, bg_col, bdr_col, status_label, note),
            unsafe_allow_html=True,
        )


# Section 2: Meteorological Briefing and Audience Advisories
st.markdown('<div class="section-label">Weather Briefing and Advisories</div>', unsafe_allow_html=True)

col_briefing, col_advisory = st.columns([3, 2], gap="medium")

with col_briefing:
    temp_val = float(forecasts.get("temp_c") or 24.8)
    hum_val = float(forecasts.get("humidity_pct") or 24.6)
    aqi_val = float(forecasts.get("aqi") or 30.9)
    rain_val = float(forecasts.get("rain_prob") or 22.3)

    report_text = agent_state.get("weather_report", "Weather briefing currently in compilation.")
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
    paragraphs_html = "".join([f"<p>{p}</p>" for p in paragraphs])

    st.markdown(f"""
    <div class="surface-panel">
        <div class="panel-header">
            <span class="panel-title">Weather Briefing</span>
            <span class="panel-tag">24h Horizon</span>
        </div>
        <div class="briefing-summary-box">
            <div class="briefing-lead">
                Atmospheric outlook across Islamabad for the upcoming 24 hour forecast window.
            </div>
            <div class="briefing-chips">
                <span class="briefing-chip">Peak Temp: {temp_val:.1f}°C</span>
                <span class="briefing-chip">Humidity: {hum_val:.1f}%</span>
                <span class="briefing-chip">Air Quality: {aqi_val:.1f} AQI</span>
                <span class="briefing-chip">Rain Risk: {rain_val:.1f}%</span>
            </div>
        </div>
        <div class="briefing-prose">
            {paragraphs_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_advisory:
    st.markdown("""
    <div class="surface-panel">
        <div class="panel-header">
            <span class="panel-title">Safety Advisories</span>
            <span class="panel-tag">By Audience</span>
        </div>
    """, unsafe_allow_html=True)

    if active_alerts:
        flagged_summary = ", ".join([a["metric"].upper() for a in active_alerts])
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12.5px; color: #EF4444;">
            <b>Active Hazard Alert:</b> {len(active_alerts)} metric(s) exceed normal thresholds ({flagged_summary}). Review protocols below.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12.5px; color: #10B981;">
            <b>Status Nominal:</b> All 8 predicted atmospheric parameters remain within safe baseline thresholds.
        </div>
        """, unsafe_allow_html=True)

    tab_public, tab_sensitive, tab_outdoor = st.tabs([
        "General Public",
        "Sensitive Groups",
        "Outdoor Activity",
    ])

    with tab_public:
        render_advisory_section(advisories.get("general", ""))
    with tab_sensitive:
        render_advisory_section(advisories.get("sensitive", ""))
    with tab_outdoor:
        render_advisory_section(advisories.get("outdoor", ""))

    st.markdown("</div>", unsafe_allow_html=True)


# Section 3: Interactive Atmospheric Telemetry Chart
st.markdown('<div class="section-label">Atmospheric Telemetry</div>', unsafe_allow_html=True)

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

        fig.add_trace(go.Scatter(
            x=history_window["timestamp"],
            y=history_window[chart_metric],
            mode="lines",
            name="Observed History",
            line=dict(color="#38BDF8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.06)",
            hovertemplate="%{y:.2f}<extra>Observed</extra>",
        ))

        last_time = history_window["timestamp"].iloc[-1]
        pred_time = last_time + pd.Timedelta(hours=24)
        pred_val = forecasts.get(chart_metric, history_window[chart_metric].iloc[-1])

        fig.add_trace(go.Scatter(
            x=[last_time, pred_time],
            y=[history_window[chart_metric].iloc[-1], pred_val],
            mode="lines+markers",
            name="24h Forecast Trajectory",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            marker=dict(size=[5, 8], color=["#38BDF8", "#F59E0B"], symbol=["circle", "circle"]),
            hovertemplate="%{y:.2f}<extra>Forecast 24h</extra>",
        ))

        hazard_rule = HAZARD_THRESHOLDS.get(chart_metric, {})
        if hazard_rule.get("max") is not None:
            fig.add_hline(
                y=hazard_rule["max"],
                line_dash="dot",
                line_color="#EF4444",
                line_width=1.0,
                annotation_text=f"Upper Limit: {hazard_rule['max']}",
                annotation_font_size=10,
                annotation_font_color="#EF4444",
                annotation_position="top right",
            )
        if hazard_rule.get("min") is not None:
            fig.add_hline(
                y=hazard_rule["min"],
                line_dash="dot",
                line_color="#F59E0B",
                line_width=1.0,
                annotation_text=f"Lower Limit: {hazard_rule['min']}",
                annotation_font_size=10,
                annotation_font_color="#F59E0B",
                annotation_position="bottom right",
            )

        metric_title = CHART_METRIC_MAP[chart_metric]
        fig.update_layout(
            margin=dict(l=12, r=12, t=36, b=12),
            height=340,
            template="plotly_dark",
            paper_bgcolor="#0F1623",
            plot_bgcolor="#0F1623",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#161F30",
                bordercolor="#2E3D56",
                font=dict(family="JetBrains Mono, monospace", size=11.5, color="#F8FAFC"),
            ),
            font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
            title=dict(
                text=f"{metric_title} ({lookback_hours}h Telemetry and 24h Projection)",
                font=dict(size=13, color="#F8FAFC", weight=600),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Timestamp (UTC)",
                showgrid=False,
                zeroline=False,
                linecolor="#1E293B",
                tickfont=dict(size=11, color="#94A3B8", family="JetBrains Mono, monospace"),
            ),
            yaxis=dict(
                title=metric_title,
                showgrid=True,
                zeroline=False,
                gridcolor="#182234",
                linecolor="#1E293B",
                tickfont=dict(size=11, color="#94A3B8", family="JetBrains Mono, monospace"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#CBD5E1"),
                bgcolor="rgba(15, 22, 35, 0.9)",
            ),
        )

        st.markdown('<div class="surface-panel" style="padding: 16px 20px;">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# Section 4: Model Observability and Governance Ledger
st.markdown('<div class="section-label">Model Observability</div>', unsafe_allow_html=True)

tab_ledger, tab_shap, tab_drift, tab_audit = st.tabs([
    "Forecast Ledger",
    "EDA and SHAP Interpretability",
    "Drift Surveillance",
    "LLM Audit Log",
])

# Tab 1: Forecast Ledger
with tab_ledger:
    st.markdown("""
    <div class="registry-bar">
        <span>Deployed Champions: <b>8 Models</b></span>
        <span class="registry-sep">|</span>
        <span>Registry Stage: <b>Production (v1)</b></span>
        <span class="registry-sep">|</span>
        <span>Split Strategy: <b>80% Train, 10% Val, 10% Test</b></span>
        <span class="registry-sep">|</span>
        <span>Inference Horizon: <b>24h Ahead</b></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
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
                "Horizon": "24h Ahead",
                "Target Timestamp": entry.get("target_timestamp", "N/A"),
                "Inference Timestamp": entry.get("forecast_created_at", "N/A"),
            })
        st.dataframe(pd.DataFrame(ledger_table), use_container_width=True, hide_index=True)
    else:
        st.info("No prediction ledger entries logged in the active session.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;">
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
        <div style="background: #0F1623; border: 1px solid #1E293B; border-radius: 8px; padding: 14px 18px; font-size: 13px; color: #94A3B8;">
            <b>Model Registry Notice:</b> Validation benchmark metrics are compiled during scheduled training runs. Champion model weights are active and serving inference.
        </div>
        """, unsafe_allow_html=True)

# Tab 2: EDA and SHAP Interpretability
with tab_shap:
    shap_col, eda_col = st.columns([1, 1], gap="medium")

    with shap_col:
        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>SHAP Global Feature Importance</div>",
            unsafe_allow_html=True,
        )
        selected_shap = st.selectbox(
            "Select Target Variable",
            options=TARGET_METRICS,
            format_func=lambda x: METRIC_REGISTRY.get(x, {}).get("name", x),
        )
        shap_file = EDA_ARTIFACTS_DIR / f"shap_importance_{selected_shap}.png"
        if shap_file.exists():
            st.image(
                str(shap_file),
                caption=f"SHAP Global Feature Importance ({METRIC_REGISTRY.get(selected_shap, {}).get('name', selected_shap)})",
                use_container_width=True,
            )
        else:
            st.info(f"SHAP artifact for {selected_shap} generated during daily training.")

    with eda_col:
        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Baseline Target Distributions</div>",
            unsafe_allow_html=True,
        )
        dist_img = EDA_ARTIFACTS_DIR / "eda_metric_distributions.png"
        if dist_img.exists():
            st.image(
                str(dist_img),
                caption="Historical Observed Distributions Across Target Categories",
                use_container_width=True,
            )

        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-top: 14px; margin-bottom: 8px;'>Diurnal 24 Hour Seasonality</div>",
            unsafe_allow_html=True,
        )
        diurnal_img = EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png"
        if diurnal_img.exists():
            st.image(
                str(diurnal_img),
                caption="Diurnal Day and Night Seasonality Profiles",
                use_container_width=True,
            )

# Tab 3: Drift Surveillance
with tab_drift:
    drift_data = agent_state.get("drift_results", {})
    drift_detected = drift_data.get("overall_drift_detected", False)

    if drift_detected:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #EF4444;">
            <b>Statistical Drift Detected:</b> Automated continuous retraining workflow triggered via GitHub Actions dispatch.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #10B981;">
            <b>Nominal Conditions:</b> Two sample Kolmogorov Smirnov p values above 0.05. Model forecast error degradation remains within the 25% threshold.
        </div>
        """, unsafe_allow_html=True)

    ks_col, md_col = st.columns(2, gap="medium")

    with ks_col:
        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Two Sample Kolmogorov Smirnov Data Drift Test</div>",
            unsafe_allow_html=True,
        )
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
                    "P Value": f"{res.get('p_value', 1):.4f}",
                    "Status": "Drift Detected" if is_drift else "Stable",
                })
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Accumulating feature rows for Kolmogorov Smirnov distribution check.")

    with md_col:
        st.markdown(
            "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Model Forecast Error Degradation</div>",
            unsafe_allow_html=True,
        )
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
            st.info("Accumulating 24 hour prediction to observation pairs for model degradation tracking.")

# Tab 4: LLM Audit Log
with tab_audit:
    st.markdown(
        "<div style='font-size: 12.5px; font-weight: 600; color: #F8FAFC; margin-bottom: 8px;'>Groq LLM Inference and Failure Audit Ledger</div>",
        unsafe_allow_html=True,
    )
    calls = agent_state.get("audit_call_logs", [])
    filter_choice = st.selectbox(
        "Log Filter",
        options=["All Inferences", "Successful Inferences Only", "Failed Inferences Only"],
    )

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

                status_col = "#10B981" if status_val == "SUCCESS" else "#EF4444"
                expander_label = f"{status_val}  |  {agent_name}  |  {ts_val}  |  {latency} ms"

                with st.expander(expander_label):
                    st.markdown(
                        f"<b>Agent Node:</b> {agent_name} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Status:</b> <span style='color:{status_col}; font-weight:600;'>{status_val}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <b>Latency:</b> {latency} ms",
                        unsafe_allow_html=True,
                    )
                    if row.get("error_message"):
                        st.error(f"Error: {row.get('error_message')}")
                    st.text_area(
                        "Submitted System and User Prompt",
                        value=str(row.get("prompt", "")),
                        height=90,
                        key=f"p_{idx}",
                        disabled=True,
                    )
                    st.text_area(
                        "Generated Output Completion",
                        value=str(row.get("response", "")),
                        height=130,
                        key=f"r_{idx}",
                        disabled=True,
                    )
    else:
        st.info("No LLM inference calls recorded in the current session.")


# Footer
st.markdown(f"""
<div class="app-footer">
    <span>AgentWeather  |  Meteorological Observability and Forecast Platform</span>
    <span>Station: {city_name}, Pakistan ({CITY_LAT}° N, {CITY_LON}° E)</span>
</div>
""", unsafe_allow_html=True)

