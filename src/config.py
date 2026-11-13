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
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")

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
# Fixed Hazard Thresholds
# -----------------------------------------------------------------------------
HAZARD_THRESHOLDS = {
    "aqi": {"min": None, "max": 150.0, "unit": "AQI", "desc": "Unhealthy Air Quality (US EPA > 150)"},
    "temp_c": {"min": 0.0, "max": 40.0, "unit": "°C", "desc": "Extreme Temperature (< 0°C Freezing or > 40°C Extreme Heat)"},
    "precip_mm": {"min": None, "max": 20.0, "unit": "mm", "desc": "Heavy Precipitation (> 20 mm/h)"},
    "rain_prob": {"min": None, "max": 80.0, "unit": "%", "desc": "High Rain Probability (> 80%)"},
    "humidity_pct": {"min": 20.0, "max": 85.0, "unit": "%", "desc": "Extreme Humidity (< 20% or > 85%)"},
    "uv_index": {"min": None, "max": 8.0, "unit": "index", "desc": "Very High UV Hazard (>= 8)"},
    "wind_speed": {"min": None, "max": 50.0, "unit": "km/h", "desc": "Gale / High Wind (> 50 km/h)"},
}

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
