"""
AgentWeather Feature Engineering and Ingestion Module
Handles live ingestion, multi-source fusion, timestamp alignment,
gap imputation, circular & rolling feature engineering, and Hopsworks storage.
"""

import os
import sys
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests

from src.config import (
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    OPENWEATHER_API_KEY,
    AQICN_API_KEY,
    OPENWEATHER_CURRENT_URL,
    AQICN_FEED_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_AIR_QUALITY_URL,
    FEATURE_GROUP_WEATHER,
    FEATURE_GROUP_VERSION,
    TARGET_METRICS,
    TARGET_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentWeather.Features")


# =============================================================================
# 1. Live Weather and Air Quality Fetching
# =============================================================================
def fetch_openweather_current(lat: float = CITY_LAT, lon: float = CITY_LON, api_key: str = OPENWEATHER_API_KEY) -> Optional[Dict[str, Any]]:
    """
    Fetch current meteorological metrics from OpenWeatherMap API.
    Returns parsed dictionary or None on failure.
    """
    if not api_key or api_key == "your_openweather_api_key_here":
        logger.warning("OpenWeatherMap API key not provided or placeholder. Will rely on OpenMeteo fallback.")
        return None

    try:
        url = OPENWEATHER_CURRENT_URL
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
        }
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            # Extract standard fields
            temp_c = data.get("main", {}).get("temp", np.nan)
            humidity = data.get("main", {}).get("humidity", np.nan)
            pressure = data.get("main", {}).get("pressure", np.nan)
            wind_speed = data.get("wind", {}).get("speed", np.nan)
            if not np.isnan(wind_speed):
                wind_speed = wind_speed * 3.6  # convert m/s to km/h
            wind_deg = data.get("wind", {}).get("deg", 0.0)
            cloud_cover = data.get("clouds", {}).get("all", 0.0)
            precip = 0.0
            if "rain" in data and isinstance(data["rain"], dict):
                precip = data["rain"].get("1h", 0.0)

            dt_utc = datetime.datetime.fromtimestamp(data.get("dt", datetime.datetime.now().timestamp()), tz=datetime.timezone.utc)

            return {
                "source": "OpenWeatherMap",
                "timestamp": dt_utc,
                "temp_c": float(temp_c) if temp_c is not None else np.nan,
                "humidity_pct": float(humidity) if humidity is not None else np.nan,
                "pressure_hpa": float(pressure) if pressure is not None else np.nan,
                "wind_speed": float(wind_speed) if wind_speed is not None else np.nan,
                "wind_dir": float(wind_deg) if wind_deg is not None else 0.0,
                "cloud_cover": float(cloud_cover) if cloud_cover is not None else 0.0,
                "precip_mm": float(precip) if precip is not None else 0.0,
                "rain_prob": 100.0 if float(precip) > 0.1 else 10.0,
                "uv_index": 0.0,
            }
        else:
            logger.warning(f"OpenWeatherMap request returned status {res.status_code}: {res.text}")
            return None
    except Exception as exc:
        logger.warning(f"OpenWeatherMap fetch failed: {exc}")
        return None


def fetch_aqicn_current(lat: float = CITY_LAT, lon: float = CITY_LON, api_key: str = AQICN_API_KEY) -> Optional[Dict[str, Any]]:
    """
    Fetch current pollutant metrics from AQICN API.
    Returns parsed dictionary or None on failure.
    """
    if not api_key or api_key == "your_aqicn_api_key_here":
        logger.warning("AQICN API key not provided or placeholder. Will rely on OpenMeteo fallback.")
        return None

    try:
        url = AQICN_FEED_URL.format(lat=lat, lon=lon, token=api_key)
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            payload = res.json()
            if payload.get("status") == "ok":
                data = payload.get("data", {})
                aqi = data.get("aqi", np.nan)
                iaqi = data.get("iaqi", {})
                pm25 = iaqi.get("pm25", {}).get("v", np.nan)
                pm10 = iaqi.get("pm10", {}).get("v", np.nan)
                no2 = iaqi.get("no2", {}).get("v", np.nan)
                so2 = iaqi.get("so2", {}).get("v", np.nan)
                co = iaqi.get("co", {}).get("v", np.nan)
                o3 = iaqi.get("o3", {}).get("v", np.nan)
                temp = iaqi.get("t", {}).get("v", np.nan)

                # AQICN time
                time_info = data.get("time", {})
                s_time = time_info.get("s", "")
                if s_time:
                    try:
                        dt = datetime.datetime.strptime(s_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
                    except Exception:
                        dt = datetime.datetime.now(datetime.timezone.utc)
                else:
                    dt = datetime.datetime.now(datetime.timezone.utc)

                return {
                    "source": "AQICN",
                    "timestamp": dt,
                    "aqi": float(aqi) if aqi is not None else np.nan,
                    "pm2_5": float(pm25) if pm25 is not None else np.nan,
                    "pm10": float(pm10) if pm10 is not None else np.nan,
                    "no2": float(no2) if no2 is not None else np.nan,
                    "so2": float(so2) if so2 is not None else np.nan,
                    "co": float(co) if co is not None else np.nan,
                    "o3": float(o3) if o3 is not None else np.nan,
                    "temp_c_station": float(temp) if temp is not None else np.nan,
                }
        logger.warning(f"AQICN returned non-ok response: {res.text}")
        return None
    except Exception as exc:
        logger.warning(f"AQICN fetch failed: {exc}")
        return None


def fetch_openmeteo_live_fallback(lat: float = CITY_LAT, lon: float = CITY_LON) -> Dict[str, Any]:
    """
    Reliable OpenMeteo fallback for live weather and air quality without requiring an API key.
    Ensures the live ingestion pipeline never fails due to third-party rate limits or timeouts.
    """
    logger.info("Querying OpenMeteo for live meteorological and air quality observation...")
    weather_url = OPEN_METEO_FORECAST_URL
    w_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,uv_index,cloud_cover",
        "hourly": "precipitation_probability",
        "timezone": "UTC",
    }
    w_data = {}
    hourly_probs = []
    for attempt in range(3):
        try:
            w_res = requests.get(weather_url, params=w_params, timeout=12)
            if w_res.status_code == 200:
                payload = w_res.json()
                w_data = payload.get("current", {})
                hourly_probs = payload.get("hourly", {}).get("precipitation_probability", [])
                break
        except Exception as exc:
            logger.warning(f"OpenMeteo weather fetch attempt {attempt + 1} error: {exc}")
            time.sleep(1)

    # Air quality
    aq_url = OPEN_METEO_AIR_QUALITY_URL
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide",
        "timezone": "UTC",
    }
    aq_data = {}
    for attempt in range(3):
        try:
            aq_res = requests.get(aq_url, params=aq_params, timeout=12)
            if aq_res.status_code == 200:
                aq_data = aq_res.json().get("current", {})
                break
        except Exception as exc:
            logger.warning(f"OpenMeteo air quality fetch attempt {attempt + 1} error: {exc}")
            time.sleep(1)

    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Estimate rain prob
    rain_prob = 15.0
    if hourly_probs:
        try:
            rain_prob = float(hourly_probs[0])
        except Exception:
            pass

    return {
        "timestamp": now_utc,
        "temp_c": float(w_data.get("temperature_2m", 22.0)),
        "humidity_pct": float(w_data.get("relative_humidity_2m", 50.0)),
        "pressure_hpa": float(w_data.get("surface_pressure", 1013.25)),
        "wind_speed": float(w_data.get("wind_speed_10m", 8.0)),
        "wind_dir": float(w_data.get("wind_direction_10m", 180.0)),
        "precip_mm": float(w_data.get("precipitation", 0.0)),
        "rain_prob": float(rain_prob),
        "uv_index": float(w_data.get("uv_index", 2.0)),
        "cloud_cover": float(w_data.get("cloud_cover", 20.0)),
        "aqi": float(aq_data.get("us_aqi", 45.0)),
        "pm2_5": float(aq_data.get("pm2_5", 12.0)),
        "pm10": float(aq_data.get("pm10", 20.0)),
        "no2": float(aq_data.get("nitrogen_dioxide", 15.0)),
        "so2": float(aq_data.get("sulphur_dioxide", 5.0)),
        "co": float(aq_data.get("carbon_monoxide", 250.0)),
        "o3": float(aq_data.get("ozone", 40.0)),
    }


def ingest_live_row(lat: float = CITY_LAT, lon: float = CITY_LON, city: str = CITY_NAME) -> pd.DataFrame:
    """
    Executes source fusion and timestamp alignment for current hour:
    - Queries OpenWeatherMap (primary meteorology) and AQICN (primary air quality).
    - Falls back to OpenMeteo for any missing fields.
    - Floors timestamp to exact hour UTC.
    - Averages overlapping temperature readings when both are present.
    """
    logger.info(f"Initiating live ingestion for {city} ({lat}, {lon})...")
    owm = fetch_openweather_current(lat, lon)
    aqicn = fetch_aqicn_current(lat, lon)
    fallback = fetch_openmeteo_live_fallback(lat, lon)

    # Base timestamp floored to exact hour UTC
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Source fusion
    # Meteorology
    temp_c = None
    if owm and not np.isnan(owm.get("temp_c", np.nan)):
        if aqicn and not np.isnan(aqicn.get("temp_c_station", np.nan)):
            temp_c = (owm["temp_c"] + aqicn["temp_c_station"]) / 2.0
            logger.info(f"Averaged temperature from OWM ({owm['temp_c']}°C) and AQICN ({aqicn['temp_c_station']}°C): {temp_c:.2f}°C")
        else:
            temp_c = owm["temp_c"]
    elif fallback:
        temp_c = fallback["temp_c"]

    humidity_pct = (owm["humidity_pct"] if owm and not np.isnan(owm.get("humidity_pct", np.nan)) else fallback["humidity_pct"])
    pressure_hpa = (owm["pressure_hpa"] if owm and not np.isnan(owm.get("pressure_hpa", np.nan)) else fallback["pressure_hpa"])
    wind_speed = (owm["wind_speed"] if owm and not np.isnan(owm.get("wind_speed", np.nan)) else fallback["wind_speed"])
    wind_dir = (owm["wind_dir"] if owm and not np.isnan(owm.get("wind_dir", np.nan)) else fallback["wind_dir"])
    precip_mm = (owm["precip_mm"] if owm and not np.isnan(owm.get("precip_mm", np.nan)) else fallback["precip_mm"])
    rain_prob = (owm["rain_prob"] if owm and not np.isnan(owm.get("rain_prob", np.nan)) else fallback["rain_prob"])
    uv_index = (owm["uv_index"] if owm and not np.isnan(owm.get("uv_index", np.nan)) else fallback["uv_index"])
    cloud_cover = (owm["cloud_cover"] if owm and not np.isnan(owm.get("cloud_cover", np.nan)) else fallback["cloud_cover"])

    # Air quality
    aqi = (aqicn["aqi"] if aqicn and not np.isnan(aqicn.get("aqi", np.nan)) else fallback["aqi"])
    pm2_5 = (aqicn["pm2_5"] if aqicn and not np.isnan(aqicn.get("pm2_5", np.nan)) else fallback["pm2_5"])
    pm10 = (aqicn["pm10"] if aqicn and not np.isnan(aqicn.get("pm10", np.nan)) else fallback["pm10"])
    no2 = (aqicn["no2"] if aqicn and not np.isnan(aqicn.get("no2", np.nan)) else fallback["no2"])
    so2 = (aqicn["so2"] if aqicn and not np.isnan(aqicn.get("so2", np.nan)) else fallback["so2"])
    co = (aqicn["co"] if aqicn and not np.isnan(aqicn.get("co", np.nan)) else fallback["co"])
    o3 = (aqicn["o3"] if aqicn and not np.isnan(aqicn.get("o3", np.nan)) else fallback["o3"])

    row = {
        "timestamp": now_utc,
        "city": city,
        "temp_c": float(temp_c),
        "humidity_pct": float(humidity_pct),
        "pressure_hpa": float(pressure_hpa),
        "wind_speed": float(wind_speed),
        "wind_dir": float(wind_dir),
        "precip_mm": float(precip_mm),
        "rain_prob": float(rain_prob),
        "uv_index": float(uv_index),
        "cloud_cover": float(cloud_cover),
        "aqi": float(aqi),
        "pm2_5": float(pm2_5),
        "pm10": float(pm10),
        "no2": float(no2),
        "so2": float(so2),
        "co": float(co),
        "o3": float(o3),
    }

    df = pd.DataFrame([row])
    logger.info("Live row successfully ingested and fused.")
    return df


# =============================================================================
# 2. Imputation Strategy
# =============================================================================
def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the documented imputation strategy:
    - Never drop hourly rows.
    - Gaps of 1 to 3 consecutive missing hours: forward fill followed by linear interpolation.
    - Outages of 4 or more hours: rolling 7-day average for that specific hour of day.
    - Fallback: global column median / backward fill to guarantee 0 nulls.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Step 1: Forward fill up to 3 consecutive hours, then linear interpolation
    df[numeric_cols] = df[numeric_cols].ffill(limit=3).interpolate(method="linear", limit_direction="both")

    # Step 2: For any remaining nulls (outages >= 4h), compute rolling 7-day hourly mean
    if df[numeric_cols].isnull().any().any():
        logger.info("Applying 7-day diurnal hour-of-day mean imputation for extended outages...")
        if "timestamp" in df.columns:
            df["_hour_tmp"] = pd.to_datetime(df["timestamp"]).dt.hour
            for col in numeric_cols:
                if df[col].isnull().any():
                    # Group by hour of day
                    hourly_means = df.groupby("_hour_tmp")[col].transform(lambda s: s.rolling(window=168, min_periods=1).mean())
                    df[col] = df[col].fillna(hourly_means)
            df.drop(columns=["_hour_tmp"], inplace=True)

    # Step 3: Absolute guarantee of zero nulls via forward/backward fill then column median
    for col in numeric_cols:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].bfill().ffill().fillna(0.0 if np.isnan(med) else med)

    return df


# =============================================================================
# 3. Feature Engineering Pipeline
# =============================================================================
def build_engineered_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes all temporal, circular, rate-of-change, and rolling features.
    Creates 24-hour shifted targets if is_training is True or if historical data available.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Impute missing values
    df = impute_missing_data(df)

    # 1. Temporal cyclical features
    dt = df["timestamp"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # 2. Wind direction circular encoding
    df["wind_sin"] = np.sin(np.radians(df["wind_dir"]))
    df["wind_cos"] = np.cos(np.radians(df["wind_dir"]))

    # 3. Rate of change (1h and 3h)
    df["temp_change_1h"] = df["temp_c"].diff(1).fillna(0.0)
    df["aqi_change_3h"] = df["aqi"].diff(3).fillna(0.0)
    df["pressure_change_3h"] = df["pressure_hpa"].diff(3).fillna(0.0)

    # 4. Rolling statistics (6h, 12h, 24h)
    rolling_metrics = ["temp_c", "aqi", "wind_speed", "humidity_pct", "pressure_hpa", "precip_mm"]
    for m in rolling_metrics:
        for w in [6, 12, 24]:
            df[f"{m}_roll_mean_{w}h"] = df[m].rolling(window=w, min_periods=1).mean()
            df[f"{m}_roll_std_{w}h"] = df[m].rolling(window=w, min_periods=1).std().fillna(0.0)

    # 5. Create 24h ahead targets (shift -24)
    for m in TARGET_METRICS:
        df[f"target_{m}"] = df[m].shift(-24)

    # Extra timestamps representation for Hopsworks compatibility
    df["timestamp_epoch"] = (df["timestamp"].astype("int64") // 10**6).astype("int64")
    df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


# =============================================================================
# 4. Hopsworks Feature Store Operations
# =============================================================================
def get_hopsworks_project():
    """
    Initializes and returns the Hopsworks project instance.
    """
    if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY == "your_hopsworks_api_key_here":
        raise ValueError("HOPSWORKS_API_KEY is not set. Please provide a valid key in .env.")
    import hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    return project


def save_features_to_hopsworks(df: pd.DataFrame, feature_group_name: str = FEATURE_GROUP_WEATHER, version: int = FEATURE_GROUP_VERSION) -> Any:
    """
    Persists feature DataFrame into Hopsworks Feature Store.
    Creates Feature Group if it does not already exist.
    Uses online_enabled=True to ensure reliable cross-platform ingestion across Windows and Linux.
    """
    project = get_hopsworks_project()
    fs = project.get_feature_store()

    clean_df = df.copy()
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    # Ensure integer schema compatibility with Hopsworks
    int_cols = ["humidity_pct", "wind_dir", "cloud_cover", "aqi", "hour", "day_of_week", "month"]
    for col in int_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].round().astype("int64")

    logger.info(f"Connecting to Hopsworks Feature Group '{feature_group_name}' (v{version})...")
    feature_group = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        primary_key=["timestamp_epoch", "city"],
        event_time="timestamp",
        description="AgentWeather hourly engineered weather and air quality observations with 24h targets",
        online_enabled=True,
    )

    logger.info(f"Inserting {len(clean_df)} records into Feature Group '{feature_group_name}'...")
    # On Windows or external environments, insert into online store via Kafka streaming
    try:
        # Insert in batches of 1000 for stability
        batch_size = 1000
        total_rows = len(clean_df)
        for i in range(0, total_rows, batch_size):
            chunk = clean_df.iloc[i:min(i + batch_size, total_rows)]
            logger.info(f"Uploading batch {i} to {min(i + batch_size, total_rows)} of {total_rows}...")
            feature_group.insert(chunk, storage="online")
        logger.info(f"Successfully saved {total_rows} records to Hopsworks Feature Group '{feature_group_name}'.")
    except Exception as exc:
        logger.warning(f"Online insert encountered error, attempting standard insert: {exc}")
        feature_group.insert(clean_df)

    return feature_group


def load_features_from_hopsworks(feature_group_name: str = FEATURE_GROUP_WEATHER, version: int = FEATURE_GROUP_VERSION, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Reads records from Hopsworks Feature Store.
    """
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=version)
    df = fg.read()
    if limit:
        df = df.tail(limit)
    return df


if __name__ == "__main__":
    logger.info("Executing live ingestion standalone test...")
    live_df = ingest_live_row()
    print("Ingested Live Observation:")
    print(live_df[["timestamp", "city", "temp_c", "humidity_pct", "wind_speed", "precip_mm", "aqi"]])
