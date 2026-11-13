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
    and constructs ledger entries.
    """
    logger.info("Executing Node 1: Forecast Agent...")
    city = state.get("city", CITY_NAME)
    now_dt = datetime.datetime.fromisoformat(state.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()))
    target_dt = now_dt + datetime.timedelta(hours=24)

    # 1. Load latest feature row
    current_features = state.get("current_features", {})
    if not current_features:
        # Load from local cache or fetch live
        parquet_path = DATA_DIR / "historical_features.parquet"
        if parquet_path.exists():
            df_hist = pd.read_parquet(parquet_path)
            latest_row = df_hist.iloc[-1].to_dict()
            current_features = latest_row
        else:
            live_df = ingest_live_row(city=city)
            eng_df = build_engineered_features(live_df)
            current_features = eng_df.iloc[-1].to_dict()
        state["current_features"] = current_features

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
                # Create a 24-step synthetic sequence repeating current scaled feature vector
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

        # Fallback to persistence if prediction failed or model missing
        if pred_val is None or np.isnan(pred_val):
            pred_val = float(current_features.get(metric, 20.0))

        # Enforce physical reality boundaries
        if metric in ["aqi", "precip_mm", "uv_index", "wind_speed"]:
            pred_val = max(0.0, pred_val)
        elif metric in ["humidity_pct", "rain_prob"]:
            pred_val = min(100.0, max(0.0, pred_val))
        elif metric == "wind_dir":
            pred_val = float(pred_val % 360.0)

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

    state["forecasts"] = forecasts
    state["forecast_ledger_entries"] = ledger_entries
    logger.info(f"Node 1 completed. 24h Predictions: {forecasts}")
    return state


# =============================================================================
# 3. Node 2: Alert Agent
# =============================================================================
def alert_node(state: AgentState) -> AgentState:
    """
    Evaluates 24-hour predictions against fixed hazard thresholds.
    Calls Groq LLM (with template fallback) to generate tailored safety advisories
    for General Public, Sensitive Groups, and Outdoor Workers.
    Logs LLM call into audit ledger.
    """
    logger.info("Executing Node 2: Alert Agent...")
    forecasts = state.get("forecasts", {})
    city = state.get("city", CITY_NAME)
    active_alerts = []

    # Evaluate hazard rules
    for metric, val in forecasts.items():
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
- Air Quality Index (AQI): {forecasts.get('aqi')} (US EPA scale)
- Temperature: {forecasts.get('temp_c')} °C
- Precipitation: {forecasts.get('precip_mm')} mm (Rain Probability: {forecasts.get('rain_prob')} %)
- Humidity: {forecasts.get('humidity_pct')} %
- UV Index: {forecasts.get('uv_index')}
- Wind: {forecasts.get('wind_speed')} km/h from {forecasts.get('wind_dir')} degrees

Active Hazard Triggers: {json.dumps(active_alerts)}

Provide concise, highly actionable safety advisories separated strictly into these 3 sections:
[GENERAL PUBLIC]
Concise practical guidance for daily life, commuting, and standard activities.

[SENSITIVE GROUPS]
Targeted precautions for children, the elderly, asthma sufferers, and individuals with cardiovascular conditions.

[OUTDOOR WORKERS]
Mandatory occupational safety guidance for construction laborers, delivery riders, and agricultural workers.
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
                    {"role": "system", "content": "You are a professional meteorological safety advisor. Be concise, structured, and factual."},
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
        # Parse sections
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
        logger.info("Generating deterministic fallback advisories...")
        if active_alerts:
            hazard_summary = ", ".join([a["description"] for a in active_alerts])
            advisories["general"] = (
                f"Hazard Alert in {city}: {hazard_summary}. Expected temperature is {forecasts.get('temp_c')}°C with AQI at {forecasts.get('aqi')}. "
                f"Stay informed, plan outdoor activities during cooler or cleaner hours, and keep rain gear on hand if rain probability ({forecasts.get('rain_prob')}%) is elevated."
            )
            advisories["sensitive"] = (
                f"Sensitive Health Warning: Due to {hazard_summary}, children, older adults, and individuals with respiratory or heart conditions "
                f"should limit strenuous outdoor exertion. Keep prescribed inhalers and medications accessible and monitor indoor air ventilation."
            )
            advisories["outdoor"] = (
                f"Occupational Hazard Protocol: Outdoor crews in {city} must adhere to safety limits. With wind speed at {forecasts.get('wind_speed')} km/h "
                f"and active hazards ({hazard_summary}), establish scheduled hydration and rest breaks in shaded or ventilated quarters."
            )
        else:
            advisories["general"] = f"Normal Weather Conditions for {city}. Forecast temperature is {forecasts.get('temp_c')}°C and AQI is {forecasts.get('aqi')}. Standard daily routines and commutes may proceed normally."
            advisories["sensitive"] = f"Air quality and temperatures are within standard ranges. No special health restrictions required today."
            advisories["outdoor"] = f"Standard workplace safety rules apply. Normal operational workflows for outdoor and field teams."

        call_log_entry["response"] = json.dumps(advisories)

    state["alert_advisories"] = advisories
    state.setdefault("audit_call_logs", []).append(call_log_entry)
    logger.info(f"Node 2 completed. Active alerts: {len(active_alerts)}.")
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


# =============================================================================
# 5. Node 4: Weather Report Agent
# =============================================================================
def weather_report_node(state: AgentState) -> AgentState:
    """
    Runs daily at 00:00 UTC or immediately upon active hazard triggers.
    Drafts a 3-paragraph broadcast news-style weather report combining forecasts and active warnings.
    Logs LLM call into agent_call_log.
    """
    logger.info("Executing Node 4: Weather Report Agent...")
    city = state.get("city", CITY_NAME)
    forecasts = state.get("forecasts", {})
    active_alerts = state.get("active_alerts", [])
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S")

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
Current Date/Time: {now_iso} UTC.

Key Forecast Metrics for the next 24 Hours:
- Temperature: {forecasts.get('temp_c')} °C
- Air Quality Index (AQI): {forecasts.get('aqi')}
- Precipitation: {forecasts.get('precip_mm')} mm (Rain Probability: {forecasts.get('rain_prob')} %)
- Relative Humidity: {forecasts.get('humidity_pct')} %
- UV Index: {forecasts.get('uv_index')}
- Wind: {forecasts.get('wind_speed')} km/h ({forecasts.get('wind_dir')}°)

Active Warnings: {json.dumps(active_alerts)}

Write a professional, compelling, 3-paragraph broadcast news script:
- Paragraph 1: Headline and general weather outlook (temperature, sky conditions, feel).
- Paragraph 2: Environmental conditions and Air Quality analysis (AQI, breathing safety, PM levels).
- Paragraph 3: Hazard warnings, public precautions, and daily planning advice for citizens.
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
                    {"role": "system", "content": "You are a professional television meteorologist delivering weather news."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=650,
                temperature=0.4,
            )
            report_text = resp.choices[0].message.content.strip()
            llm_succeeded = True
        except Exception as exc:
            logger.warning(f"Groq WeatherReportAgent error: {exc}. Utilizing deterministic fallback.")
            call_log_entry["status"] = "failed"
            call_log_entry["error_message"] = str(exc)

    latency_ms = int((time.time() - start_time) * 1000)
    call_log_entry["latency_ms"] = latency_ms

    if not llm_succeeded or not report_text:
        # Deterministic Broadcast Script Fallback
        hazard_note = f"Special weather statements remain active regarding {active_alerts[0]['description']}." if active_alerts else "Stable atmospheric patterns prevail with no severe alerts issued."
        report_text = (
            f"Good day, Islamabad. Here is your official 24-hour meteorological outlook. "
            f"Temperatures across the capital region are projected to hold near {forecasts.get('temp_c')}°C over the next 24 hours, accompanied by humidity levels around {forecasts.get('humidity_pct')}%. "
            f"Wind velocities are anticipated near {forecasts.get('wind_speed')} km/h, keeping the ambient conditions well-ventilated through the morning and afternoon hours.\n\n"
            f"Turning to environmental and air quality readings, the regional Air Quality Index is trending near {forecasts.get('aqi')}. "
            f"{'Particulate concentrations remain somewhat elevated, requiring caution during peak travel hours.' if forecasts.get('aqi', 0) > 100 else 'Air quality remains comfortably within moderate limits across the metropolitan area.'} "
            f"Precipitation chances stand at {forecasts.get('rain_prob')}%, with expected hourly rainfall accumulations measuring approximately {forecasts.get('precip_mm')} millimeters.\n\n"
            f"For commuters and outdoor workers, {hazard_note} "
            f"The UV Index is forecast to peak at {forecasts.get('uv_index')}, meaning moderate sun protection is advised during midday hours. "
            f"Stay tuned to the AgentWeather broadcast terminal for continuous automated hourly updates."
        )

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
    - Executes LangGraph pipeline.
    - Saves audit ledgers into Hopsworks.
    - Caches state locally for Streamlit UI consumption.
    """
    logger.info("Initializing AgentWeather LangGraph orchestration pipeline...")
    app = build_agent_graph()

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    current_features = {}
    if initial_df is not None and not initial_df.empty:
        current_features = initial_df.iloc[-1].to_dict()

    initial_state: AgentState = {
        "city": CITY_NAME,
        "timestamp": now_iso,
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
