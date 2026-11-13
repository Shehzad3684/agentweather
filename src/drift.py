"""
AgentWeather Drift Detection and Continuous Retraining Trigger Module
Implements:
1. Two-sample Kolmogorov-Smirnov (KS) Data Drift Test comparing incoming features against training distribution.
2. Model Drift Detection comparing historical predictions (t-24h) against actual observations.
3. GitHub Actions Continuous Retraining Trigger via GitHub REST API (workflow_dispatch).
4. Hopsworks audit logging to 'drift_log' feature group.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from src.config import (
    CITY_NAME,
    GH_PAT,
    GITHUB_REPOSITORY,
    FEATURE_GROUP_DRIFT_LOG,
    FEATURE_GROUP_VERSION,
    TARGET_METRICS,
)
from src.features import get_hopsworks_project

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentWeather.Drift")


# =============================================================================
# 1. Data Drift: Two-Sample Kolmogorov-Smirnov Test
# =============================================================================
def run_ks_data_drift_test(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    features_to_check: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Performs a two-sample Kolmogorov-Smirnov test comparing the latest feature distribution
    against the baseline training distribution.
    Flags data drift if p-value < alpha (default 0.05).
    """
    if features_to_check is None:
        features_to_check = ["temp_c", "humidity_pct", "pressure_hpa", "wind_speed", "precip_mm", "aqi"]

    drift_report = {}
    drift_detected = False

    for feature in features_to_check:
        if feature in current_df.columns and feature in reference_df.columns:
            ref_vals = reference_df[feature].dropna().values
            curr_vals = current_df[feature].dropna().values

            if len(ref_vals) > 10 and len(curr_vals) > 0:
                ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
                is_drifted = bool(p_val < alpha)
                if is_drifted:
                    drift_detected = True

                drift_report[feature] = {
                    "ks_statistic": float(round(ks_stat, 4)),
                    "p_value": float(round(p_val, 6)),
                    "drift_detected": is_drifted,
                    "threshold_alpha": alpha,
                }
            else:
                drift_report[feature] = {"status": "insufficient_data", "drift_detected": False}

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "drift_detected": drift_detected,
        "features_evaluated": len(features_to_check),
        "drifted_features": [f for f, r in drift_report.items() if r.get("drift_detected", False)],
        "details": drift_report,
    }
    return summary


# =============================================================================
# 2. Model Drift: Forecast Error Degradation Tracking
# =============================================================================
def evaluate_model_drift(
    predictions_history: pd.DataFrame,
    actuals_df: pd.DataFrame,
    baseline_maes: Dict[str, float],
    threshold_pct: float = 25.0,
) -> Dict[str, Any]:
    """
    Compares 24-hour predictions made 24 hours ago against today's actual observations.
    Flags model drift if recent MAE exceeds validation baseline MAE by more than 25%.
    """
    model_drift_report = {}
    any_model_drifted = False

    if predictions_history.empty or actuals_df.empty:
        logger.info("Insufficient forecast history to compute model drift.")
        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_drift_detected": False,
            "metrics": {},
        }

    # Match actuals with predicted ledger on target_timestamp and metric
    for metric in TARGET_METRICS:
        pred_subset = predictions_history[predictions_history["metric"] == metric].copy()
        if not pred_subset.empty and metric in actuals_df.columns:
            # Standardize timestamps to datetime UTC for clean merge
            pred_subset["target_timestamp"] = pd.to_datetime(pred_subset["target_timestamp"], utc=True)
            actual_subset = actuals_df[["timestamp", metric]].copy()
            actual_subset["target_timestamp"] = pd.to_datetime(actual_subset["timestamp"], utc=True)
            actual_subset = actual_subset.rename(columns={metric: "actual_value"}).drop(columns=["timestamp"])
            
            merged = pd.merge(pred_subset, actual_subset, on="target_timestamp", how="inner")
            if len(merged) >= 5:
                recent_mae = float(np.mean(np.abs(merged["actual_value"] - merged["predicted_value"])))
                base_mae = baseline_maes.get(metric, 10.0)
                pct_degradation = ((recent_mae - base_mae) / (base_mae + 1e-6)) * 100.0

                is_drifted = bool(pct_degradation > threshold_pct)
                if is_drifted:
                    any_model_drifted = True

                model_drift_report[metric] = {
                    "recent_mae": round(recent_mae, 4),
                    "baseline_mae": round(base_mae, 4),
                    "pct_degradation": round(pct_degradation, 2),
                    "threshold_pct": threshold_pct,
                    "drift_detected": is_drifted,
                }

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_drift_detected": any_model_drifted,
        "metrics": model_drift_report,
    }


# =============================================================================
# 3. GitHub Actions Retraining Dispatch Trigger
# =============================================================================
def trigger_retraining_pipeline(
    pat: str = GH_PAT,
    repo: str = GITHUB_REPOSITORY,
    workflow_file: str = "training_pipeline.yml",
    branch: str = "main",
    reason: str = "Automated drift threshold crossed",
) -> bool:
    """
    Triggers GitHub Actions training_pipeline.yml workflow via workflow_dispatch REST API.
    """
    if not pat or not repo or pat == "your_github_personal_access_token_here":
        logger.warning("GitHub PAT or GITHUB_REPOSITORY not configured. Skipping workflow_dispatch trigger.")
        return False

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "ref": branch,
        "inputs": {
            "trigger_reason": reason,
        },
    }
    try:
        logger.info(f"Triggering GitHub Actions retraining workflow ({url})...")
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code in [204, 201]:
            logger.info("Successfully triggered GitHub Actions training pipeline dispatch!")
            return True
        else:
            logger.warning(f"Failed to dispatch GitHub workflow: HTTP {res.status_code} - {res.text}")
            return False
    except Exception as exc:
        logger.error(f"Error triggering GitHub workflow: {exc}")
        return False


# =============================================================================
# 4. Hopsworks Audit Logging to drift_log
# =============================================================================
def log_drift_to_hopsworks(
    records: List[Dict[str, Any]],
    feature_group_name: str = FEATURE_GROUP_DRIFT_LOG,
    version: int = FEATURE_GROUP_VERSION,
) -> bool:
    """
    Logs drift events to the Hopsworks 'drift_log' Feature Group.
    """
    if not records:
        return True

    df_drift = pd.DataFrame(records)
    if "timestamp" not in df_drift.columns:
        df_drift["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
    else:
        df_drift["timestamp"] = pd.to_datetime(df_drift["timestamp"])

    df_drift["timestamp_epoch"] = (df_drift["timestamp"].astype("int64") // 10**6).astype("int64")
    df_drift["city"] = df_drift.get("city", CITY_NAME)

    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            primary_key=["timestamp_epoch", "metric", "drift_type"],
            event_time="timestamp",
            description="AgentWeather audit log for KS data drift and MAE model drift evaluations",
            online_enabled=True,
        )
        fg.insert(df_drift, storage="online")
        logger.info(f"Logged {len(df_drift)} drift audit records to Hopsworks '{feature_group_name}'.")
        return True
    except Exception as exc:
        logger.warning(f"Failed to log drift to Hopsworks: {exc}")
        return False
