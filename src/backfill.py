"""
AgentWeather Historical Backfill Module
Downloads 6 to 12 months of hourly meteorological and air quality historical records
from OpenMeteo APIs (free, no API key required), aligns timestamps, computes engineered
features and 24-hour ahead targets, saves locally, and populates Hopsworks Feature Store.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from src.config import (
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    DATA_DIR,
    OPEN_METEO_HISTORICAL_URL,
    OPEN_METEO_AIR_QUALITY_URL,
    OPEN_METEO_FORECAST_URL,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_WEATHER,
    FEATURE_GROUP_VERSION,
    TARGET_METRICS,
    TARGET_COLUMNS,
)
from src.features import build_engineered_features, save_features_to_hopsworks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentWeather.Backfill")


def fetch_openmeteo_historical_weather(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    start_date: str = "2026-03-01",
    end_date: str = "2026-11-08",
) -> pd.DataFrame:
    """
    Fetches hourly historical meteorological data from Open-Meteo Archive API.
    """
    logger.info(f"Downloading historical weather for ({lat}, {lon}) from {start_date} to {end_date}...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "cloud_cover",
        ],
        "timezone": "UTC",
    }
    res = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=30)
    res.raise_for_status()
    data = res.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "temp_c": data["temperature_2m"],
        "humidity_pct": data["relative_humidity_2m"],
        "pressure_hpa": data["surface_pressure"],
        "wind_speed": data["wind_speed_10m"],
        "wind_dir": data["wind_direction_10m"],
        "precip_mm": data["precipitation"],
        "cloud_cover": data["cloud_cover"],
    })
    # Compute approximate rain_prob and UV index based on precip & solar radiation / time
    df["rain_prob"] = np.where(df["precip_mm"] > 0.05, np.minimum(100.0, df["precip_mm"] * 30.0 + 40.0), 5.0)
    
    # Calculate synthetic realistic UV index for historical hours based on solar hour and cloud cover
    hour = df["timestamp"].dt.hour
    day_fraction = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    df["uv_index"] = np.round(day_fraction * (1.0 - df["cloud_cover"] / 100.0 * 0.7) * 9.0, 1)

    logger.info(f"Downloaded {len(df)} historical weather hourly rows.")
    return df


def fetch_openmeteo_historical_air_quality(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    start_date: str = "2026-03-01",
    end_date: str = "2026-11-08",
) -> pd.DataFrame:
    """
    Fetches hourly historical air quality data from Open-Meteo Air Quality API.
    """
    logger.info(f"Downloading historical air quality for ({lat}, {lon}) from {start_date} to {end_date}...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10",
            "pm2_5",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "carbon_monoxide",
            "us_aqi",
        ],
        "timezone": "UTC",
    }
    res = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=30)
    res.raise_for_status()
    data = res.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"], utc=True),
        "aqi": data.get("us_aqi", [50.0] * len(data["time"])),
        "pm2_5": data.get("pm2_5", [12.0] * len(data["time"])),
        "pm10": data.get("pm10", [20.0] * len(data["time"])),
        "no2": data.get("nitrogen_dioxide", [15.0] * len(data["time"])),
        "so2": data.get("sulphur_dioxide", [5.0] * len(data["time"])),
        "co": data.get("carbon_monoxide", [250.0] * len(data["time"])),
        "o3": data.get("ozone", [40.0] * len(data["time"])),
    })
    logger.info(f"Downloaded {len(df)} historical air quality hourly rows.")
    return df


def fetch_recent_days_openmeteo(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    past_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the most recent days (bridging historical archive with today)
    using the OpenMeteo forecast endpoint with past_days parameter.
    """
    logger.info(f"Fetching recent {past_days} days to bridge up to current hour...")
    w_url = OPEN_METEO_FORECAST_URL
    w_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,precipitation_probability,uv_index,cloud_cover",
        "timezone": "UTC",
    }
    w_res = requests.get(w_url, params=w_params, timeout=20)
    w_res.raise_for_status()
    w_data = w_res.json().get("hourly", {})
    df_w = pd.DataFrame({
        "timestamp": pd.to_datetime(w_data["time"], utc=True),
        "temp_c": w_data["temperature_2m"],
        "humidity_pct": w_data["relative_humidity_2m"],
        "pressure_hpa": w_data["surface_pressure"],
        "wind_speed": w_data["wind_speed_10m"],
        "wind_dir": w_data["wind_direction_10m"],
        "precip_mm": w_data["precipitation"],
        "rain_prob": w_data.get("precipitation_probability", [10.0] * len(w_data["time"])),
        "uv_index": w_data.get("uv_index", [1.0] * len(w_data["time"])),
        "cloud_cover": w_data["cloud_cover"],
    })

    aq_url = OPEN_METEO_AIR_QUALITY_URL
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide,us_aqi",
        "timezone": "UTC",
    }
    aq_res = requests.get(aq_url, params=aq_params, timeout=20)
    aq_res.raise_for_status()
    aq_data = aq_res.json().get("hourly", {})
    df_aq = pd.DataFrame({
        "timestamp": pd.to_datetime(aq_data["time"], utc=True),
        "aqi": aq_data.get("us_aqi", [50.0] * len(aq_data["time"])),
        "pm2_5": aq_data.get("pm2_5", [12.0] * len(aq_data["time"])),
        "pm10": aq_data.get("pm10", [20.0] * len(aq_data["time"])),
        "no2": aq_data.get("nitrogen_dioxide", [15.0] * len(aq_data["time"])),
        "so2": aq_data.get("sulphur_dioxide", [5.0] * len(aq_data["time"])),
        "co": aq_data.get("carbon_monoxide", [250.0] * len(aq_data["time"])),
        "o3": aq_data.get("ozone", [40.0] * len(aq_data["time"])),
    })

    return df_w, df_aq


def run_backfill(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    city: str = CITY_NAME,
    months_back: int = 8,
) -> pd.DataFrame:
    """
    Executes end-to-end backfill:
    1. Determines date range (8 months of hourly data = ~5,800 records).
    2. Fetches weather and air quality historical and recent records.
    3. Merges on timestamp and adds city identifier.
    4. Applies imputation and feature engineering with 24h shifted targets.
    5. Saves to local parquet and populates Hopsworks if key is configured.
    """
    logger.info(f"Starting historical backfill for {city} ({lat}, {lon}) spanning past {months_back} months...")
    today = datetime.datetime.now(datetime.timezone.utc).date()
    # Archive is complete up to 5 days ago
    archive_end = today - datetime.timedelta(days=5)
    archive_start = today - datetime.timedelta(days=months_back * 30)

    start_str = archive_start.strftime("%Y-%m-%d")
    end_str = archive_end.strftime("%Y-%m-%d")

    # 1. Fetch archive
    df_w_arch = fetch_openmeteo_historical_weather(lat, lon, start_str, end_str)
    df_aq_arch = fetch_openmeteo_historical_air_quality(lat, lon, start_str, end_str)

    # 2. Fetch recent bridge
    df_w_rec, df_aq_rec = fetch_recent_days_openmeteo(lat, lon, past_days=6)

    # 3. Concatenate and deduplicate
    df_w = pd.concat([df_w_arch, df_w_rec], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df_aq = pd.concat([df_aq_arch, df_aq_rec], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    # Merge weather and air quality
    df_merged = pd.merge(df_w, df_aq, on="timestamp", how="inner")
    df_merged["city"] = city

    # Limit timestamps up to current hour
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
    df_merged = df_merged[df_merged["timestamp"] <= now_utc].reset_index(drop=True)

    logger.info(f"Raw merged dataset contains {len(df_merged)} hourly records.")

    # 4. Feature engineering and 24h targets
    df_features = build_engineered_features(df_merged, is_training=True)

    # Check nulls in non-target features
    feature_cols = [c for c in df_features.columns if not c.startswith("target_")]
    null_counts = df_features[feature_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Unexpected nulls found in feature columns:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("Clean check passed: EXACTLY 0 null values across all feature columns.")

    # 5. Save local copy for offline training, EDA, and caching
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_parquet_path = DATA_DIR / "historical_features.parquet"
    local_csv_path = DATA_DIR / "historical_features.csv"
    df_features.to_parquet(local_parquet_path, index=False)
    # Save a lighter CSV sample or full CSV
    df_features.to_csv(local_csv_path, index=False)
    logger.info(f"Saved local backfill cache to {local_parquet_path} ({len(df_features)} rows).")

    # 6. Push to Hopsworks Feature Store if configured
    if HOPSWORKS_API_KEY and HOPSWORKS_API_KEY != "your_hopsworks_api_key_here":
        try:
            logger.info("Populating Hopsworks Feature Store 'weather_features'...")
            save_features_to_hopsworks(df_features, feature_group_name=FEATURE_GROUP_WEATHER, version=FEATURE_GROUP_VERSION)
            logger.info("Successfully pushed historical features to Hopsworks Feature Store.")
        except Exception as exc:
            logger.error(f"Failed to push to Hopsworks: {exc}")
    else:
        logger.info("HOPSWORKS_API_KEY not configured yet. Cached features locally for subsequent push.")

    return df_features


if __name__ == "__main__":
    df_backfill = run_backfill()
    print("Backfill completed successfully:")
    print("Shape:", df_backfill.shape)
    print("Date range:", df_backfill["timestamp"].min(), "to", df_backfill["timestamp"].max())
    print("Sample targets:\n", df_backfill[TARGET_COLUMNS].tail(30))
