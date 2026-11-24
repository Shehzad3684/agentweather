"""
AgentWeather 4-Node LangGraph Agent Layer Module
Implements:
1. Forecast Agent (Node 1): Loads features, evaluates 8 champion models for t+24h, updates state & forecast_ledger.
2. Alert Agent (Node 2): Evaluates hazard rules, calls Groq LLM (with template fallback) for tailored advisories.
3. Drift Agent (Node 3): Runs KS data drift & model drift, triggers GitHub retraining if needed, logs to drift_log.
4. Weather Report Agent (Node 4): Generates 3-paragraph broadcast news report combining forecasts & active hazards.
5. Hopsworks Feature Group audit logging for forecast_ledger, agent_call_log, and drift_log.
"""

import os
import sys
import json
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
from langgraph.graph import StateGraph, END
from groq import Groq

from src.config import (
    BASE_DIR,
    DATA_DIR,
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    GROQ_API_KEY,
    GROQ_MODEL,
    GH_PAT,
    GITHUB_REPOSITORY,
    HAZARD_THRESHOLDS,
    TARGET_METRICS,
    FEATURE_GROUP_WEATHER,
    FEATURE_GROUP_FORECAST_LEDGER,
    FEATURE_GROUP_AGENT_CALL_LOG,
    FEATURE_GROUP_DRIFT_LOG,
    FEATURE_GROUP_VERSION,
    classify_epa_aqi,
)
from src.features import get_hopsworks_project, ingest_live_row, build_engineered_features
from src.drift import run_ks_data_drift_test, evaluate_model_drift, trigger_retraining_pipeline, log_drift_to_hopsworks
from src.train import WeatherLSTM

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentWeather.Agents")

MODELS_DIR = BASE_DIR / "models"


# =============================================================================
# 1. Shared State Definition
# =============================================================================
class AgentState(TypedDict):
    city: str
    timestamp: str
    observed: Dict[str, Any]
    current_features: Dict[str, Any]
    forecasts: Dict[str, float]
    forecast_ledger_entries: List[Dict[str, Any]]
    active_alerts: List[Dict[str, Any]]
    alert_advisories: Dict[str, str]  # general, sensitive, outdoor
    weather_report: str
    drift_results: Dict[str, Any]
    audit_call_logs: List[Dict[str, Any]]
    errors: List[str]


# =============================================================================
# 2. Node 1: Forecast Agent
# =============================================================================
def forecast_node(state: AgentState) -> AgentState:
    """
    Loads latest engineered features and champion models,
    generates 24-hour ahead predictions across all 8 target metrics,
    computes 24h temperature extrema (temp_max_24h and temp_min_24h),
    and constructs ledger entries.
    """
    logger.info("Executing Node 1: Forecast Agent...")
    city = state.get("city", CITY_NAME)
    now_dt = datetime.datetime.fromisoformat(state.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()))
    target_dt = now_dt + datetime.timedelta(hours=24)

    # 1. Load latest feature row and observed metadata
    current_features = state.get("current_features", {})
    observed = state.get("observed", {})
    temp_max_24h = None
    temp_min_24h = None

    if not current_features or not observed:
        live_df = ingest_live_row(city=city)
        eng_df = build_engineered_features(live_df)
        current_features = eng_df.iloc[-1].to_dict()
        observed = eng_df.attrs.get("observed", {})
        temp_max_24h = eng_df.attrs.get("temp_max_24h")
        temp_min_24h = eng_df.attrs.get("temp_min_24h")
        state["current_features"] = current_features
        state["observed"] = observed

    if temp_max_24h is None:
        temp_max_24h = observed.get("temp_max_24h", observed.get("temp_c", 20.0) + 4.0)
    if temp_min_24h is None:
        temp_min_24h = observed.get("temp_min_24h", observed.get("temp_c", 20.0) - 5.0)

    # 2. Load training results to inspect features and model types
    results_path = DATA_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            training_metadata = json.load(f)
    else:
        training_metadata = {}

    forecasts = {}
    ledger_entries = []

    # Load scaler if available
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Predict for each of the 8 targets
    for metric in TARGET_METRICS:
        meta = training_metadata.get(metric, {})
        feature_cols = meta.get("feature_columns", [])
        champion_family = meta.get("champion_family", "Classical ML (LightGBM)")

        # Prepare X feature vector
        if feature_cols:
            x_vec = np.array([[current_features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
        else:
            excluded = {"timestamp", "timestamp_str", "timestamp_epoch", "city"} | {f"target_{m}" for m in TARGET_METRICS}
            cols = [k for k in current_features.keys() if k not in excluded and isinstance(current_features[k], (int, float))]
            x_vec = np.array([[current_features.get(c, 0.0) for c in cols]], dtype=np.float32)

        pred_val = None
        metric_dir = MODELS_DIR / metric
        model_joblib = metric_dir / "model.joblib"
        model_pt = metric_dir / "model.pt"

        try:
            if model_pt.exists() and "LSTM" in champion_family:
                lstm = WeatherLSTM(input_dim=x_vec.shape[1], hidden_dim=32, num_layers=1)
                lstm.load_state_dict(torch.load(model_pt, weights_only=True))
                lstm.eval()
                x_scaled = scaler.transform(x_vec) if scaler else x_vec
                x_seq = np.repeat(x_scaled[np.newaxis, :, :], 24, axis=1)
                with torch.no_grad():
                    pred_val = float(lstm(torch.from_numpy(x_seq)).item())
            elif model_joblib.exists():
                model = joblib.load(model_joblib)
                if hasattr(model, "forecast"):
                    try:
                        f_vals = model.forecast(steps=1)
                        pred_val = float(f_vals.iloc[0] if hasattr(f_vals, "iloc") else f_vals[0])
                    except Exception:
                        pred_val = float(current_features.get(f"{metric}_roll_mean_6h", current_features.get(metric, 0.0)))
                elif "Ridge" in champion_family and scaler:
                    x_scaled = scaler.transform(x_vec)
                    pred_val = float(model.predict(x_scaled)[0])
                else:
                    pred_val = float(model.predict(x_vec)[0])
        except Exception as exc:
            logger.warning(f"Error predicting with {metric} champion: {exc}. Using persistence fallback.")

        # Fallback to current observation if prediction failed
        if pred_val is None or np.isnan(pred_val):
            pred_val = float(observed.get(metric, current_features.get(metric, 20.0)))

        # Enforce physical reality boundaries & calibration
        if metric in ["aqi", "precip_mm", "uv_index", "wind_speed"]:
            pred_val = max(0.0, pred_val)
        elif metric in ["humidity_pct", "rain_prob"]:
            pred_val = min(100.0, max(0.0, pred_val))
        elif metric == "wind_dir":
            pred_val = float(pred_val % 360.0)

        # Temperature bias fix: ensure t+24h point forecast respects local diurnal limits
        if metric == "temp_c":
            if pred_val < temp_min_24h - 3.0:
                pred_val = max(temp_min_24h, pred_val)
            elif pred_val > temp_max_24h + 3.0:
                pred_val = min(temp_max_24h, pred_val)

        pred_val = round(pred_val, 2)
        forecasts[metric] = pred_val

        # Add to ledger
        ledger_entries.append({
            "forecast_created_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "metric": metric,
            "predicted_value": pred_val,
            "model_family": champion_family,
            "model_version": 1,
            "city": city,
        })

    # Window extrema targets for 24h outlook
    forecasts["temp_max_24h"] = round(float(temp_max_24h), 1)
    forecasts["temp_min_24h"] = round(float(temp_min_24h), 1)

    state["forecasts"] = forecasts
    state["forecast_ledger_entries"] = ledger_entries
    logger.info(f"Node 1 completed. 24h Predictions: {forecasts}, Extrema: High={temp_max_24h}°C, Low={temp_min_24h}°C")
    return state


# =============================================================================
# 3. Node 2: Alert Agent
# =============================================================================
def alert_node(state: AgentState) -> AgentState:
    """
    Evaluates 24-hour predictions against official US EPA AQI 6-tier standards
    and fixed meteorological hazard thresholds.
    Calls Groq LLM (with template fallback) to generate tailored safety advisories
    for General Public, Sensitive Groups, and Outdoor Workers.
    Logs LLM call into audit ledger.
    """
    logger.info("Executing Node 2: Alert Agent...")
    forecasts = state.get("forecasts", {})
    city = state.get("city", CITY_NAME)
    active_alerts = []

    # 1. Evaluate AQI under official US EPA standards
    aqi_val = float(forecasts.get("aqi", 0.0))
    epa_cat, text_col, bg_col, bdr_col, note = classify_epa_aqi(aqi_val)

    if aqi_val > 100.0:
        if aqi_val <= 150.0:
            active_alerts.append({
                "metric": "aqi",
                "value": aqi_val,
                "category": "Unhealthy for Sensitive Groups",
                "threshold": "101 - 150 AQI",
                "description": f"US EPA AQI {aqi_val:.1f} is Unhealthy for Sensitive Groups (101 to 150)",
                "reason": "Active advisory for children, elderly, and respiratory/cardiovascular patients.",
            })
        elif aqi_val <= 200.0:
            active_alerts.append({
                "metric": "aqi",
                "value": aqi_val,
                "category": "Unhealthy",
                "threshold": "151 - 200 AQI",
                "description": f"US EPA AQI {aqi_val:.1f} is Unhealthy (151 to 200)",
                "reason": "Air quality is Unhealthy according to US EPA standards. General public should limit prolonged outdoor exertion and consider N95/FFP2 masks.",
            })
        elif aqi_val <= 300.0:
            active_alerts.append({
                "metric": "aqi",
                "value": aqi_val,
                "category": "Very Unhealthy",
                "threshold": "201 - 300 AQI",
                "description": f"US EPA AQI {aqi_val:.1f} is Very Unhealthy (201 to 300)",
                "reason": "Health alert: serious risk of respiratory health effects for everyone. Avoid strenuous outdoor activities.",
            })
        else:
            active_alerts.append({
                "metric": "aqi",
                "value": aqi_val,
                "category": "Hazardous",
                "threshold": "> 300 AQI",
                "description": f"US EPA AQI {aqi_val:.1f} is Hazardous (> 300)",
                "reason": "Emergency warning: entire population is likely to experience severe health impacts.",
            })

    # 2. Evaluate remaining metrics against HAZARD_THRESHOLDS
    for metric, val in forecasts.items():
        if metric in ["aqi", "temp_max_24h", "temp_min_24h"]:
            continue
        rule = HAZARD_THRESHOLDS.get(metric)
        if not rule:
            continue
        min_thresh = rule.get("min")
        max_thresh = rule.get("max")
        triggered = False
        hazard_desc = ""

        if min_thresh is not None and val < min_thresh:
            triggered = True
            hazard_desc = f"{metric.upper()} value of {val} is below safe lower limit ({min_thresh} {rule['unit']})"
        elif max_thresh is not None and val > max_thresh:
            triggered = True
            hazard_desc = f"{metric.upper()} value of {val} exceeds safety hazard threshold ({max_thresh} {rule['unit']})"

        if triggered:
            active_alerts.append({
                "metric": metric,
                "value": val,
                "category": "Hazard",
                "threshold": f"{min_thresh} - {max_thresh} {rule['unit']}",
                "description": rule["desc"],
                "reason": hazard_desc,
            })

    state["active_alerts"] = active_alerts
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Construct advisories via Groq or Deterministic Fallback
    advisories = {}
    call_log_entry = {
        "timestamp": now_iso,
        "agent_name": "AlertAgent",
        "prompt": "",
        "response": "",
        "latency_ms": 0,
        "status": "success",
        "error_message": "",
        "city": city,
    }

    prompt = f"""You are the Chief Safety Meteorologist for {city}.
Review these 24-hour weather predictions:
- Air Quality Index (AQI): {forecasts.get('aqi')} ({epa_cat} per US EPA scale)
- Temperature (t+24h): {forecasts.get('temp_c')} °C (24h Peak: {forecasts.get('temp_max_24h')} °C, 24h Low: {forecasts.get('temp_min_24h')} °C)
- Precipitation: {forecasts.get('precip_mm')} mm (Rain Probability: {forecasts.get('rain_prob')} %)
- Humidity: {forecasts.get('humidity_pct')} %
- UV Index: {forecasts.get('uv_index')}
- Wind: {forecasts.get('wind_speed')} km/h from {forecasts.get('wind_dir')} degrees

Active Hazard Triggers: {json.dumps(active_alerts)}

Provide concise, highly actionable safety advisories separated strictly into these 3 sections:
[GENERAL PUBLIC]
Concise practical guidance for daily life, commuting, and standard activities. If AQI is >= 151 (Unhealthy), advise reducing prolonged exertion and recommending masks if sensitive.

[SENSITIVE GROUPS]
Targeted precautions for children, the elderly, asthma sufferers, and individuals with cardiovascular conditions. If AQI is >= 101, advise limiting strenuous outdoor activity.

[OUTDOOR WORKERS]
Mandatory occupational safety guidance for construction laborers, delivery riders, and agricultural workers. If AQI >= 151, recommend N95 masks and scheduled breaks.
"""
    call_log_entry["prompt"] = prompt

    start_time = time.time()
    llm_succeeded = False
    raw_response = ""

    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional meteorological safety advisor. Be concise, structured, factual, and strictly follow official US EPA AQI standards."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=650,
                temperature=0.3,
            )
            raw_response = resp.choices[0].message.content.strip()
            llm_succeeded = True
        except Exception as exc:
            logger.warning(f"Groq AlertAgent call encountered error: {exc}. Utilizing deterministic fallback.")
            call_log_entry["status"] = "failed"
            call_log_entry["error_message"] = str(exc)

    latency_ms = int((time.time() - start_time) * 1000)
    call_log_entry["latency_ms"] = latency_ms

    if llm_succeeded and raw_response:
        call_log_entry["response"] = raw_response
        parts = {"general": "", "sensitive": "", "outdoor": ""}
        current_section = "general"
        for line in raw_response.splitlines():
            line_upper = line.upper()
            if "GENERAL PUBLIC" in line_upper:
                current_section = "general"
            elif "SENSITIVE GROUPS" in line_upper:
                current_section = "sensitive"
            elif "OUTDOOR WORKERS" in line_upper:
                current_section = "outdoor"
            else:
                parts[current_section] += line + "\n"
        advisories = {k: v.strip() for k, v in parts.items()}
    else:
        # Deterministic Python String Template Fallback
        logger.info("Generating deterministic fallback advisories adhering to US EPA tiers...")
        if aqi_val > 150.0:
            aq_public = f"Air quality is classified as Unhealthy (AQI {aqi_val:.1f}). General public should reduce prolonged or heavy outdoor exertion."
            aq_sensitive = f"Children, elderly individuals, and people with respiratory or heart disease must avoid prolonged outdoor exertion and wear N95/FFP2 masks outdoors."
            aq_outdoor = f"Outdoor crews should wear protective N95 masks, take frequent breaks in indoor ventilated zones, and rotate heavy tasks."
        elif aqi_val > 100.0:
            aq_public = f"Air quality is Unhealthy for Sensitive Groups (AQI {aqi_val:.1f}). General public can engage in normal outdoor activities."
            aq_sensitive = f"Sensitive individuals (asthma, heart disease, elderly, children) should limit prolonged or strenuous outdoor exertion."
            aq_outdoor = f"Standard operational protocols with monitoring for sensitive crew members."
        else:
            aq_public = f"Air quality is {epa_cat} (AQI {aqi_val:.1f}). Standard routines and commutes can proceed normally."
            aq_sensitive = f"Air quality is within acceptable limits. No special restrictions for sensitive groups."
            aq_outdoor = f"Standard workplace safety rules apply."

        advisories["general"] = f"- **Air Quality:** {aq_public}\n- **Thermal Outlook:** Daytime high near {forecasts.get('temp_max_24h')}°C and low near {forecasts.get('temp_min_24h')}°C.\n- **Commuting:** Rain probability is {forecasts.get('rain_prob')}%; roads clear."
        advisories["sensitive"] = f"- **Respiratory Precautions:** {aq_sensitive}\n- **Hydration:** Maintain regular fluid intake with humidity around {forecasts.get('humidity_pct')}%."
        advisories["outdoor"] = f"- **Occupational Guidelines:** {aq_outdoor}\n- **Wind & Sun:** Winds at {forecasts.get('wind_speed')} km/h, UV index at {forecasts.get('uv_index')}."

        call_log_entry["response"] = json.dumps(advisories)

    state["alert_advisories"] = advisories
    state.setdefault("audit_call_logs", []).append(call_log_entry)
    logger.info(f"Node 2 completed. Active alerts: {len(active_alerts)}.")
    return state
    return state


# =============================================================================
# 4. Node 3: Drift Agent
# =============================================================================
def drift_node(state: AgentState) -> AgentState:
    """
    Evaluates data drift via two-sample KS test and model drift via recent MAE comparison.
    Logs checks to Hopsworks drift_log.
    Dispatches GitHub Actions training pipeline if drift crosses thresholds.
    """
    logger.info("Executing Node 3: Drift Agent...")
    city = state.get("city", CITY_NAME)
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    # 1. Load historical reference data
    parquet_path = DATA_DIR / "historical_features.parquet"
    if parquet_path.exists():
        df_hist = pd.read_parquet(parquet_path)
    else:
        df_hist = pd.DataFrame([state.get("current_features", {})])

    # Reference window: first 80% (training set)
    n_train = int(len(df_hist) * 0.8)
    ref_df = df_hist.iloc[:n_train]
    # Current window: latest 24 hours
    curr_df = df_hist.iloc[-24:] if len(df_hist) >= 24 else df_hist

    # Run KS Data Drift Test
    ks_results = run_ks_data_drift_test(current_df=curr_df, reference_df=ref_df)

    # Run Model Drift Test
    training_res_path = DATA_DIR / "training_results.json"
    baseline_maes = {}
    if training_res_path.exists():
        with open(training_res_path, "r") as f:
            tr_meta = json.load(f)
            baseline_maes = {m: tr_meta[m]["validation_metrics"]["mae"] for m in tr_meta}

    model_drift_results = evaluate_model_drift(
        predictions_history=pd.DataFrame(state.get("forecast_ledger_entries", [])),
        actuals_df=df_hist,
        baseline_maes=baseline_maes,
        threshold_pct=25.0,
    )

    overall_drift_detected = ks_results.get("drift_detected", False) or model_drift_results.get("model_drift_detected", False)

    # Prepare audit records for Hopsworks drift_log
    drift_records = []
    for feat, det in ks_results.get("details", {}).items():
        drift_records.append({
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "metric": feat,
            "drift_type": "data_ks_test",
            "drift_score": det.get("ks_statistic", 0.0),
            "threshold": det.get("threshold_alpha", 0.05),
            "action_taken": "retraining_dispatched" if overall_drift_detected else "monitored",
            "city": city,
        })

    for m, det in model_drift_results.get("metrics", {}).items():
        drift_records.append({
            "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "metric": m,
            "drift_type": "model_mae_degradation",
            "drift_score": det.get("pct_degradation", 0.0),
            "threshold": det.get("threshold_pct", 25.0),
            "action_taken": "retraining_dispatched" if det.get("drift_detected", False) else "monitored",
            "city": city,
        })

    # Log to Hopsworks drift_log
    log_drift_to_hopsworks(drift_records)

    # Trigger GitHub Actions continuous retraining if drift detected
    if overall_drift_detected:
        logger.warning("Significant drift detected! Dispatching GitHub Actions training workflow...")
        trigger_retraining_pipeline(
            pat=GH_PAT,
            repo=GITHUB_REPOSITORY,
            reason=f"Automated trigger: Data drift ({ks_results.get('drifted_features')}) or Model drift detected.",
        )

    state["drift_results"] = {
        "data_drift": ks_results,
        "model_drift": model_drift_results,
        "overall_drift_detected": overall_drift_detected,
    }
    logger.info(f"Node 3 completed. Drift detected: {overall_drift_detected}.")
    return state


def validate_report_grounding(report_text: str, payload: Dict[str, Any]) -> bool:
    """
    Validates that the generated LLM text does not contradict canonical payload facts:
    - If cloud_cover_pct > 10, cannot claim clear skies or unbroken sunshine.
    - Ensures report is substantial (at least 100 characters).
    """
    if not report_text or len(report_text) < 100:
        return False
    txt_lower = report_text.lower()
    obs = payload.get("observed") or payload.get("observed_current", {})
    cloud_pct = obs.get("cloud_cover_pct", 0)

    # Cloud cover contradiction check (Bug 4)
    if cloud_pct > 10:
        clear_terms = ["clear skies", "clear sky", "crystal clear", "completely clear", "unbroken by cloud"]
        if any(term in txt_lower for term in clear_terms):
            logger.warning(f"Report validation rejected: text claims clear skies but observed cloud cover is {cloud_pct}%.")
            return False

    return True


def generate_grounded_report(payload: Dict[str, Any]) -> str:
    """
    Generates a deterministic, journalistic, 100% grounded 3-paragraph weather report
    strictly adhering to canonical pre-rounded payload values.
    """
    city = payload["city"]
    day_period = payload.get("day_period", "day")
    greeting = f"Good {day_period}, {city}."
    obs = payload.get("observed") or payload.get("observed_current", {})
    fc = payload.get("forecast_24h") or payload.get("forecast_next_24h", {})
    alerts = payload.get("active_alerts", [])

    cloud_pct = obs.get("cloud_cover_pct", 20)
    cloud_desc = (
        "clear skies" if cloud_pct <= 10
        else f"partly cloudy skies with {cloud_pct}% cloud cover" if cloud_pct <= 50
        else f"mostly overcast conditions with {cloud_pct}% cloud cover"
    )

    temp_c = obs.get("temp_c", obs.get("temperature_c", 18.0))
    hum_pct = obs.get("humidity_pct", obs.get("relative_humidity_pct", 50))
    press_hpa = obs.get("pressure_hpa", obs.get("surface_pressure_hpa", 1013.2))
    obs_aqi = obs.get("aqi", obs.get("air_quality_index", 50.0))
    obs_aqi_cat = obs.get("aqi_category", obs.get("air_quality_category", "Good"))

    fc_max = fc.get("temp_max_24h", fc.get("daytime_high_temperature_c", 22.0))
    fc_min = fc.get("temp_min_24h", fc.get("overnight_low_temperature_c", 11.0))
    fc_target = fc.get("temp_c", fc.get("target_temperature_c", 15.0))
    fc_aqi = fc.get("aqi", fc.get("air_quality_index", 50.0))
    fc_aqi_cat = fc.get("aqi_category", fc.get("air_quality_category", "Good"))
    fc_rain = fc.get("rain_prob_pct", fc.get("rain_probability_pct", 0))
    fc_precip = fc.get("precip_mm", fc.get("precipitation_mm", 0.0))
    fc_uv = fc.get("uv_index", 2.0)
    fc_wind_spd = fc.get("wind_speed_kmh", 5.0)
    fc_wind_dir = fc.get("wind_dir_deg", 180)

    p1 = (
        f"{greeting} Current monitoring station telemetry records an observed temperature of {temp_c}°C "
        f"under {cloud_desc}, with relative humidity at {hum_pct}% and surface pressure steady at {press_hpa} hPa. "
        f"Over the upcoming 24-hour cycle, atmospheric models indicate daytime warmth peaking at a 24-hour high of {fc_max}°C, "
        f"with overnight cooling descending toward {fc_min}°C, and the t+24h forward point forecast settling near {fc_target}°C."
    )

    p2 = (
        f"Turning to environmental and air quality surveillance, the current regional Air Quality Index is measured at {obs_aqi} ({obs_aqi_cat}) "
        f"and is predicted to average around {fc_aqi} ({fc_aqi_cat}) over the next 24 hours under official US EPA criteria. "
        f"Precipitation chances remain at {fc_rain}% with expected accumulations limited to {fc_precip} mm, "
        f"while surface winds circulate from {fc_wind_dir}° at approximately {fc_wind_spd} km/h."
    )

    if alerts:
        alert_summary = "; ".join([a.get("description", str(a)) for a in alerts])
        p3 = (
            f"Public safety advisories have been issued across the capital region: {alert_summary}. "
            f"Given that air quality is categorized as {fc_aqi_cat}, sensitive individuals and outdoor personnel should follow recommended precautions. "
            f"The solar UV index is anticipated at {fc_uv}, and citizens are encouraged to monitor live updates on the AgentWeather terminal."
        )
    else:
        p3 = (
            f"All predicted atmospheric and environmental parameters remain within nominal safety baselines for the capital region. "
            f"The solar radiation index is forecast at {fc_uv}, indicating minimal sunburn risk outside peak solar hours. "
            f"Standard municipal workflows, outdoor labor schedules, and daily commutes may proceed according to normal guidelines."
        )

    return f"{p1}\n\n{p2}\n\n{p3}"


# =============================================================================
# 5. Node 4: Weather Report Agent
# =============================================================================
def weather_report_node(state: AgentState) -> AgentState:
    """
    Drafts a 3-paragraph broadcast news-style weather report combining forecasts,
    live observed conditions, and active hazard warnings.
    Strictly grounds all numerical figures in a canonical pre-rounded payload.
    Logs LLM call into agent_call_log.
    """
    logger.info("Executing Node 4: Weather Report Agent...")
    city = state.get("city", CITY_NAME)
    forecasts = state.get("forecasts", {})
    observed = state.get("observed", {})
    active_alerts = state.get("active_alerts", [])

    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    pkt_dt = now_dt + datetime.timedelta(hours=5)
    pkt_str = pkt_dt.strftime("%Y-%m-%d %H:%M:%S PKT")

    local_hour = pkt_dt.hour
    if 5 <= local_hour < 12:
        day_period = "morning"
    elif 12 <= local_hour < 17:
        day_period = "afternoon"
    elif 17 <= local_hour < 21:
        day_period = "evening"
    else:
        day_period = "night"

    obs_aqi = float(observed.get("aqi", forecasts.get("aqi", 45.0)))
    fc_aqi = float(forecasts.get("aqi", 45.0))

    canonical_payload = {
        "city": city,
        "cycle_utc": now_iso + " UTC",
        "local_time_pkt": pkt_str,
        "day_period": day_period,
        "observed": {
            "temp_c": round(float(observed.get("temp_c", forecasts.get("temp_c", 18.0))), 1),
            "humidity_pct": int(round(float(observed.get("humidity_pct", forecasts.get("humidity_pct", 50.0))))),
            "pressure_hpa": round(float(observed.get("pressure_hpa", 1013.2)), 1),
            "cloud_cover_pct": int(round(float(observed.get("cloud_cover", 20.0)))),
            "wind_speed_kmh": round(float(observed.get("wind_speed", forecasts.get("wind_speed", 5.0))), 1),
            "aqi": round(obs_aqi, 1),
            "aqi_category": classify_epa_aqi(obs_aqi)[0],
        },
        "forecast_24h": {
            "temp_c": round(float(forecasts.get("temp_c", 20.0)), 1),
            "temp_max_24h": round(float(forecasts.get("temp_max_24h", forecasts.get("temp_c", 20.0) + 4.0)), 1),
            "temp_min_24h": round(float(forecasts.get("temp_min_24h", forecasts.get("temp_c", 20.0) - 4.0)), 1),
            "humidity_pct": int(round(float(forecasts.get("humidity_pct", 50.0)))),
            "precip_mm": round(float(forecasts.get("precip_mm", 0.0)), 1),
            "rain_prob_pct": int(round(float(forecasts.get("rain_prob", 10.0)))),
            "uv_index": round(float(forecasts.get("uv_index", 2.0)), 1),
            "wind_speed_kmh": round(float(forecasts.get("wind_speed", 5.0)), 1),
            "wind_dir_deg": int(round(float(forecasts.get("wind_dir", 180.0)))),
            "aqi": round(fc_aqi, 1),
            "aqi_category": classify_epa_aqi(fc_aqi)[0],
        },
        "active_alerts": active_alerts,
    }

    call_log_entry = {
        "timestamp": now_iso,
        "agent_name": "WeatherReportAgent",
        "prompt": "",
        "response": "",
        "latency_ms": 0,
        "status": "success",
        "error_message": "",
        "city": city,
    }

    prompt = f"""You are a television broadcast meteorologist delivering the official 24-hour weather forecast for {city}.
You must strictly follow these grounding rules:
1. Ground greetings and time-of-day remarks in local time ({pkt_str}, period: {day_period}).
2. Distinguish current observed conditions ({canonical_payload['observed']['temp_c']}°C, {canonical_payload['observed']['cloud_cover_pct']}% cloud cover) from the 24-hour forecast.
3. If cloud_cover_pct is greater than 10%, DO NOT state the sky is "unbroken by cloud" or "completely clear". Report cloud cover as {canonical_payload['observed']['cloud_cover_pct']}%.
4. Clearly distinguish the 24h Peak Temperature ({canonical_payload['forecast_24h']['temp_max_24h']}°C) and Overnight Low ({canonical_payload['forecast_24h']['temp_min_24h']}°C) from the t+24h point forecast ({canonical_payload['forecast_24h']['temp_c']}°C). Never call an overnight/morning low the "Peak Temp".
5. Use the exact US EPA Air Quality category: '{canonical_payload['forecast_24h']['aqi_category']}' for forecast AQI {canonical_payload['forecast_24h']['aqi']}.
6. Use exact numbers matching this canonical payload:
{json.dumps(canonical_payload, indent=2)}

Write a professional, compelling, 3-paragraph broadcast news script:
- Paragraph 1: Headline and general weather outlook (current temp, sky conditions, 24h Peak and Low temperatures).
- Paragraph 2: Environmental conditions and Air Quality analysis (US EPA AQI, particulate safety, precipitation, wind).
- Paragraph 3: Public precautions, active hazard guidance, and planning advice for citizens.
Do not use bullet points. Write purely in clean, broadcast-ready journalistic prose.
"""
    call_log_entry["prompt"] = prompt

    start_time = time.time()
    llm_succeeded = False
    report_text = ""

    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional television meteorologist delivering grounded weather news."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=650,
                temperature=0.3,
            )
            candidate_text = resp.choices[0].message.content.strip()
            if validate_report_grounding(candidate_text, canonical_payload):
                report_text = candidate_text
                llm_succeeded = True
            else:
                logger.warning("Groq WeatherReportAgent output failed grounding validation. Falling back to verified generator.")
                report_text = generate_grounded_report(canonical_payload)
                llm_succeeded = True
        except Exception as exc:
            logger.warning(f"Groq WeatherReportAgent error: {exc}. Utilizing deterministic fallback.")
            call_log_entry["status"] = "failed"
            call_log_entry["error_message"] = str(exc)

    latency_ms = int((time.time() - start_time) * 1000)
    call_log_entry["latency_ms"] = latency_ms

    if not report_text:
        report_text = generate_grounded_report(canonical_payload)

    call_log_entry["response"] = report_text
    state["weather_report"] = report_text
    state.setdefault("audit_call_logs", []).append(call_log_entry)
    logger.info("Node 4 completed. Weather report compiled.")
    return state


# =============================================================================
# 6. Audit Logging to Hopsworks Feature Store
# =============================================================================
def save_audit_ledgers_to_hopsworks(state: AgentState):
    """
    Persists forecast_ledger and agent_call_log into Hopsworks Feature Groups.
    """
    city = state.get("city", CITY_NAME)

    # 1. Forecast Ledger
    ledger_entries = state.get("forecast_ledger_entries", [])
    if ledger_entries:
        try:
            df_ledger = pd.DataFrame(ledger_entries)
            df_ledger["forecast_created_at"] = pd.to_datetime(df_ledger["forecast_created_at"])
            df_ledger["target_timestamp"] = pd.to_datetime(df_ledger["target_timestamp"])
            df_ledger["timestamp_epoch"] = (df_ledger["forecast_created_at"].astype("int64") // 10**6).astype("int64")

            project = get_hopsworks_project()
            fs = project.get_feature_store()
            fg_ledger = fs.get_or_create_feature_group(
                name=FEATURE_GROUP_FORECAST_LEDGER,
                version=FEATURE_GROUP_VERSION,
                primary_key=["timestamp_epoch", "metric", "city"],
                event_time="forecast_created_at",
                description="AgentWeather 24-hour prediction ledger for model accuracy and drift auditing",
                online_enabled=True,
            )
            fg_ledger.insert(df_ledger, storage="online")
            logger.info(f"Persisted {len(df_ledger)} prediction entries to Hopsworks '{FEATURE_GROUP_FORECAST_LEDGER}'.")
        except Exception as exc:
            logger.warning(f"Failed to persist forecast ledger to Hopsworks: {exc}")

    # 2. Agent Call Log
    call_logs = state.get("audit_call_logs", [])
    if call_logs:
        try:
            df_calls = pd.DataFrame(call_logs)
            df_calls["timestamp"] = pd.to_datetime(df_calls["timestamp"])
            df_calls["timestamp_epoch"] = (df_calls["timestamp"].astype("int64") // 10**6).astype("int64")
            if "prompt" in df_calls.columns:
                df_calls["prompt"] = df_calls["prompt"].astype(str).str.slice(0, 850)
            if "response" in df_calls.columns:
                df_calls["response"] = df_calls["response"].astype(str).str.slice(0, 1850)

            project = get_hopsworks_project()
            fs = project.get_feature_store()
            fg_calls = fs.get_or_create_feature_group(
                name=FEATURE_GROUP_AGENT_CALL_LOG,
                version=FEATURE_GROUP_VERSION,
                primary_key=["timestamp_epoch", "agent_name", "city"],
                event_time="timestamp",
                description="AgentWeather LLM call performance and failure audit log",
                online_enabled=True,
            )
            fg_calls.insert(df_calls, storage="online")
            logger.info(f"Persisted {len(df_calls)} agent call records to Hopsworks '{FEATURE_GROUP_AGENT_CALL_LOG}'.")
        except Exception as exc:
            logger.warning(f"Failed to persist agent call logs to Hopsworks: {exc}")


# =============================================================================
# 7. LangGraph Graph Assembly & Orchestration
# =============================================================================
def build_agent_graph():
    """
    Constructs the 4-node LangGraph StateGraph workflow:
    forecast_agent -> alert_agent -> drift_agent -> weather_report_agent -> END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("forecast_agent", forecast_node)
    workflow.add_node("alert_agent", alert_node)
    workflow.add_node("drift_agent", drift_node)
    workflow.add_node("weather_report_agent", weather_report_node)

    workflow.set_entry_point("forecast_agent")
    workflow.add_edge("forecast_agent", "alert_agent")
    workflow.add_edge("alert_agent", "drift_agent")
    workflow.add_edge("drift_agent", "weather_report_agent")
    workflow.add_edge("weather_report_agent", END)

    return workflow.compile()


def run_agent_pipeline(initial_df: Optional[pd.DataFrame] = None) -> AgentState:
    """
    Runs the complete 4-node agentic workflow:
    - Accepts optional DataFrame directly in memory from live ingestion to bypass offline/online sync lag.
    - Preserves live observed telemetry snapshot in state.
    - Executes LangGraph pipeline.
    - Saves audit ledgers into Hopsworks.
    - Caches state locally for Streamlit UI consumption.
    """
    logger.info("Initializing AgentWeather LangGraph orchestration pipeline...")
    app = build_agent_graph()

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    current_features = {}
    observed = {}
    if initial_df is not None and not initial_df.empty:
        current_features = initial_df.iloc[-1].to_dict()
        if hasattr(initial_df, "attrs") and "observed" in initial_df.attrs:
            observed = initial_df.attrs["observed"]

    if not observed:
        live_df = ingest_live_row(city=CITY_NAME)
        observed = live_df.attrs.get("observed", {})
        if not current_features:
            eng_df = build_engineered_features(live_df)
            current_features = eng_df.iloc[-1].to_dict()

    initial_state: AgentState = {
        "city": CITY_NAME,
        "timestamp": now_iso,
        "observed": observed,
        "current_features": current_features,
        "forecasts": {},
        "forecast_ledger_entries": [],
        "active_alerts": [],
        "alert_advisories": {},
        "weather_report": "",
        "drift_results": {},
        "audit_call_logs": [],
        "errors": [],
    }

    final_state = app.invoke(initial_state)

    # Persist audit ledgers to Hopsworks
    save_audit_ledgers_to_hopsworks(final_state)

    # Cache state locally in data/latest_agent_state.json for fast Streamlit UI rendering
    state_to_cache = {
        "city": final_state["city"],
        "timestamp": final_state["timestamp"],
        "observed": final_state.get("observed", {}),
        "forecasts": final_state["forecasts"],
        "forecast_ledger_entries": final_state["forecast_ledger_entries"],
        "active_alerts": final_state["active_alerts"],
        "alert_advisories": final_state["alert_advisories"],
        "weather_report": final_state["weather_report"],
        "drift_results": final_state["drift_results"],
        "audit_call_logs": final_state["audit_call_logs"],
    }
    with open(DATA_DIR / "latest_agent_state.json", "w") as f:
        json.dump(state_to_cache, f, indent=2, default=str)

    logger.info("AgentWeather LangGraph pipeline run completed successfully.")
    return final_state


if __name__ == "__main__":
    final_output = run_agent_pipeline()
    print("\n--- Pipeline Execution Summary ---")
    print("Forecasts:", final_output["forecasts"])
    print("Active Alerts:", len(final_output["active_alerts"]))
    print("Weather Report Snippet:", final_output["weather_report"][:200] + "...")
