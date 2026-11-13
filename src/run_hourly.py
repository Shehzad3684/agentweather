"""
AgentWeather Hourly Pipeline Runner
Executes:
1. Live multi-source ingestion (OpenWeatherMap + AQICN + OpenMeteo fallback)
2. Feature engineering & timestamp alignment
3. Direct in-memory pass to LangGraph 4-node agent layer (bypassing Hopsworks sync latency)
4. Audit ledger persistence (forecast_ledger, agent_call_log, drift_log)
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CITY_NAME, CITY_LAT, CITY_LON
from src.features import ingest_live_row, build_engineered_features, save_features_to_hopsworks
from src.agents import run_agent_pipeline

def run_hourly_cycle():
    print(f"=== Starting Hourly AgentWeather Pipeline for {CITY_NAME} ===")
    
    # 1. Ingest live observation
    live_df = ingest_live_row(lat=CITY_LAT, lon=CITY_LON, city=CITY_NAME)
    
    # 2. Build engineered features
    eng_df = build_engineered_features(live_df)
    
    # 3. Stream to Hopsworks Feature Store
    try:
        save_features_to_hopsworks(eng_df)
    except Exception as e:
        print(f"Notice: Streaming to Hopsworks encountered: {e}")
        
    # 4. Pass DataFrame directly in memory to LangGraph Agent pipeline
    final_state = run_agent_pipeline(initial_df=eng_df)
    
    print("=== Hourly Pipeline Completed Successfully ===")
    print(f"24h Forecasts Generated: {final_state['forecasts']}")
    print(f"Active Alerts Triggered: {len(final_state['active_alerts'])}")
    print(f"Drift Status: {final_state.get('drift_results', {}).get('overall_drift_detected')}")
    return final_state

if __name__ == "__main__":
    run_hourly_cycle()
