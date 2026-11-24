"""
AgentWeather Configuration Module
Centralized settings for API endpoints, city coordinates, feature definitions,
hazard thresholds, Hopsworks feature groups, and model targets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env file if present
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EDA_ARTIFACTS_DIR = DATA_DIR / "eda_artifacts"
EDA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# City and Location Settings (Islamabad, Pakistan)
# -----------------------------------------------------------------------------
CITY_NAME = os.getenv("CITY_NAME", "Islamabad")
CITY_LAT = float(os.getenv("CITY_LAT", "33.6844"))
CITY_LON = float(os.getenv("CITY_LON", "73.0479"))

# -----------------------------------------------------------------------------
# API Credentials and Project Settings
# -----------------------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "agent_weather")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

GH_PAT = os.getenv("GH_PAT", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "Shehzad3684/agentweather")

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
AQICN_FEED_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"

# -----------------------------------------------------------------------------
# Targets to Predict (24 Hours Ahead: t + 24h)
# -----------------------------------------------------------------------------
TARGET_METRICS = [
    "aqi",            # US EPA scale (0 to 500)
    "temp_c",         # Degrees Celsius
    "precip_mm",      # Millimeters
    "rain_prob",      # Probability (0 to 100%)
    "humidity_pct",   # Percentage (0 to 100%)
    "uv_index",       # Index (0 to 11+)
    "wind_speed",     # Speed (km/h)
    "wind_dir",       # Compass degrees (0 to 360°)
]

TARGET_COLUMNS = [f"target_{m}" for m in TARGET_METRICS]

# -----------------------------------------------------------------------------
# Fixed Hazard Thresholds & US EPA AQI Standards
# -----------------------------------------------------------------------------
HAZARD_THRESHOLDS = {
    "aqi": {
        "min": None,
        "max": 150.0,
        "unit": "AQI",
        "desc": "Unhealthy Air Quality (US EPA > 150)",
        "advisory_sensitive": 100.0,
        "advisory_unhealthy": 150.0,
        "advisory_very_unhealthy": 200.0,
        "advisory_hazardous": 300.0,
    },
    "temp_c": {"min": 0.0, "max": 40.0, "unit": "°C", "desc": "Extreme Temperature (< 0°C Freezing or > 40°C Extreme Heat)"},
    "precip_mm": {"min": None, "max": 20.0, "unit": "mm", "desc": "Heavy Precipitation (> 20 mm/h)"},
    "rain_prob": {"min": None, "max": 80.0, "unit": "%", "desc": "High Rain Probability (> 80%)"},
    "humidity_pct": {"min": 20.0, "max": 85.0, "unit": "%", "desc": "Extreme Humidity (< 20% or > 85%)"},
    "uv_index": {"min": None, "max": 8.0, "unit": "index", "desc": "Very High UV Hazard (>= 8)"},
    "wind_speed": {"min": None, "max": 50.0, "unit": "km/h", "desc": "Gale / High Wind (> 50 km/h)"},
}

def classify_epa_aqi(v: float):
    """
    Standardizes on the official 6-tier US EPA Air Quality Index scale:
    - 0 to 50: Good
    - 51 to 100: Moderate
    - 101 to 150: Unhealthy for Sensitive Groups
    - 151 to 200: Unhealthy
    - 201 to 300: Very Unhealthy
    - 301 to 500: Hazardous
    Returns: (category, text_color, bg_color, border_color, note)
    """
    if v is None:
        v = 0.0
    val = float(v)
    if val <= 50.0:
        return (
            "Good",
            "#10B981",
            "rgba(16, 185, 129, 0.10)",
            "rgba(16, 185, 129, 0.25)",
            "Good: 0 to 50",
        )
    elif val <= 100.0:
        return (
            "Moderate",
            "#F59E0B",
            "rgba(245, 158, 11, 0.10)",
            "rgba(245, 158, 11, 0.25)",
            "Moderate: 51 to 100",
        )
    elif val <= 150.0:
        return (
            "Unhealthy for Sensitive Groups",
            "#FB923C",
            "rgba(251, 146, 60, 0.10)",
            "rgba(251, 146, 60, 0.25)",
            "101 to 150: Sensitive Groups",
        )
    elif val <= 200.0:
        return (
            "Unhealthy",
            "#EF4444",
            "rgba(239, 68, 68, 0.10)",
            "rgba(239, 68, 68, 0.25)",
            "Unhealthy: 151 to 200",
        )
    elif val <= 300.0:
        return (
            "Very Unhealthy",
            "#A855F7",
            "rgba(168, 85, 247, 0.10)",
            "rgba(168, 85, 247, 0.25)",
            "Very Unhealthy: 201 to 300",
        )
    else:
        return (
            "Hazardous",
            "#E11D48",
            "rgba(225, 29, 72, 0.10)",
            "rgba(225, 29, 72, 0.25)",
            "Hazardous: 301 to 500",
        )


def pm25_to_epa_aqi(pm25: float) -> float:
    """
    Computes US EPA AQI from PM2.5 concentration (ug/m3) using official 2024 breakpoints.
    """
    if pm25 is None or pm25 < 0:
        return 0.0
    c = round(float(pm25), 1)

    breakpoints = [
        (0.0, 9.0, 0.0, 50.0),
        (9.1, 35.4, 51.0, 100.0),
        (35.5, 55.4, 101.0, 150.0),
        (55.5, 125.4, 151.0, 200.0),
        (125.5, 225.4, 201.0, 300.0),
        (225.5, 325.4, 301.0, 500.0),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low, 1)

    if c > 325.4:
        # Above top breakpoint
        return min(500.0, round(((500.0 - 301.0) / (500.0 - 325.5)) * (c - 325.5) + 301.0, 1))

    return 0.0

# -----------------------------------------------------------------------------
# Hopsworks Feature Group Names
# -----------------------------------------------------------------------------
FEATURE_GROUP_WEATHER = "weather_features"
FEATURE_GROUP_FORECAST_LEDGER = "forecast_ledger"
FEATURE_GROUP_AGENT_CALL_LOG = "agent_call_log"
FEATURE_GROUP_DRIFT_LOG = "drift_log"
FEATURE_GROUP_VERSION = 1

# -----------------------------------------------------------------------------
# Core Base Features
# -----------------------------------------------------------------------------
BASE_METEOROLOGICAL_FEATURES = [
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed",
    "wind_dir",
    "precip_mm",
    "rain_prob",
    "uv_index",
    "cloud_cover",
]

BASE_AIR_QUALITY_FEATURES = [
    "aqi",
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
]
