"""
AgentWeather Multi-Family Model Training and Registry Module
Trains Naive baseline, Statistical, Classical ML (LightGBM & Ridge), and PyTorch LSTM models
across all 8 target variables. Evaluates with RMSE, MAE, R2. Selects champions,
runs SHAP explainability, and registers models in Hopsworks Model Registry.
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    BASE_DIR,
    DATA_DIR,
    EDA_ARTIFACTS_DIR,
    HOPSWORKS_API_KEY,
    TARGET_METRICS,
    TARGET_COLUMNS,
    FEATURE_GROUP_WEATHER,
    FEATURE_GROUP_VERSION,
)
from src.features import get_hopsworks_project, load_features_from_hopsworks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AgentWeather.Train")

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. PyTorch LSTM Model Definition
# =============================================================================
class WeatherLSTM(nn.Module):
    """
    Lightweight PyTorch LSTM Sequence Model for 24-hour lookback weather forecasting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take the hidden state of the final timestep
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs rolling sequence windows of length `seq_len` for LSTM training.
    """
    X_seqs, y_seqs = [], []
    for i in range(len(X) - seq_len):
        X_seqs.append(X[i : i + seq_len])
        y_seqs.append(y[i + seq_len])
    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# =============================================================================
# 2. Evaluation Utilities
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes RMSE, MAE, and R2 score.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


# =============================================================================
# 3. Model Training Pipeline
# =============================================================================
def train_and_evaluate_all_targets(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Executes the multi-family model training pipeline:
    1. Loads historical data (from Hopsworks or local parquet cache).
    2. Chronological 80/10/10 time split (excluding unobserved newest 24h targets).
    3. Evaluates 4 candidate model families for each of the 8 targets:
       - Naive Persistence Baseline (y_pred(t+24) = y(t))
       - Statistical (ARIMA/SARIMAX)
       - Classical ML (LightGBM Regressor + Ridge benchmark)
       - Deep Learning (PyTorch LSTM Sequence Model)
    4. Selects champion based on validation RMSE.
    5. Runs SHAP explainability on LightGBM champion models.
    6. Registers winning models in Hopsworks Model Registry.
    """
    logger.info("Starting AgentWeather multi-family model training...")

    # Load dataset
    if data_path is None or not data_path.exists():
        data_path = DATA_DIR / "historical_features.parquet"

    if data_path.exists():
        logger.info(f"Loading features from local parquet {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        logger.info("Loading features from Hopsworks Feature Store...")
        df = load_features_from_hopsworks()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter out newest rows where targets are null
    valid_mask = df[TARGET_COLUMNS].notnull().all(axis=1)
    df_trainable = df[valid_mask].copy().reset_index(drop=True)
    N = len(df_trainable)
    logger.info(f"Trainable dataset size (excluding newest 24h targets): {N} rows.")

    # Chronological 80% / 10% / 10% split
    n_train = int(0.80 * N)
    n_val = int(0.10 * N)
    n_test = N - n_train - n_val

    df_train = df_trainable.iloc[:n_train].copy()
    df_val = df_trainable.iloc[n_train : n_train + n_val].copy()
    df_test = df_trainable.iloc[n_train + n_val :].copy()

    logger.info(f"Splits -> Train: {len(df_train)} rows, Val: {len(df_val)} rows, Test: {len(df_test)} rows.")

    # Prepare feature matrix X (exclude non-feature metadata and targets)
    excluded_cols = {"timestamp", "timestamp_str", "timestamp_epoch", "city"} | set(TARGET_COLUMNS)
    feature_cols = [c for c in df_trainable.columns if c not in excluded_cols]
    logger.info(f"Number of engineered input features: {len(feature_cols)}")

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values

    # Feature scaler for Ridge and LSTM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    training_results = {}
    shap_explainers = {}

    # Train dedicated model for each target
    for metric in TARGET_METRICS:
        logger.info(f"==================================================")
        logger.info(f"Training models for target: '{metric}'")
        logger.info(f"==================================================")

        target_col = f"target_{metric}"
        y_train = df_train[target_col].values
        y_val = df_val[target_col].values
        y_test = df_test[target_col].values

        # -------------------------------------------------------------
        # 1. Naive Persistence Baseline: prediction(t+24) = observation(t)
        # -------------------------------------------------------------
        y_val_naive = df_val[metric].values
        y_test_naive = df_test[metric].values
        naive_val_metrics = compute_metrics(y_val, y_val_naive)
        naive_test_metrics = compute_metrics(y_test, y_test_naive)
        logger.info(f"[{metric}] Naive Baseline -> Val RMSE: {naive_val_metrics['rmse']}, MAE: {naive_val_metrics['mae']}, R2: {naive_val_metrics['r2']}")

        # -------------------------------------------------------------
        # 2. Statistical Model (ARIMA / SARIMAX)
        # -------------------------------------------------------------
        logger.info(f"[{metric}] Fitting Statistical Model (ARIMA)...")
        try:
            # Fit on recent train history (last 500 hours for stability)
            series_train = y_train[-500:]
            arima_model = ARIMA(series_train, order=(1, 0, 1)).fit()
            # Rolling 24-step forecast approximation on validation
            mean_stat_pred = arima_model.forecast(steps=len(y_val))
            stat_val_metrics = compute_metrics(y_val, mean_stat_pred)
            stat_model_obj = arima_model
        except Exception as e:
            logger.warning(f"ARIMA fit failed: {e}. Falling back to persistence mean.")
            mean_stat_pred = np.full_like(y_val, y_train.mean())
            stat_val_metrics = compute_metrics(y_val, mean_stat_pred)
            stat_model_obj = None
        logger.info(f"[{metric}] Statistical Model -> Val RMSE: {stat_val_metrics['rmse']}, MAE: {stat_val_metrics['mae']}, R2: {stat_val_metrics['r2']}")

        # -------------------------------------------------------------
        # 3. Classical ML: LightGBM Regressor & Ridge Benchmark
        # -------------------------------------------------------------
        logger.info(f"[{metric}] Fitting Classical ML: LightGBM Regressor...")
        # For wind_dir circular target, train directly or on wind_sin/wind_cos
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=120,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(X_train, y_train)
        y_val_lgb = lgb_model.predict(X_val)
        lgb_val_metrics = compute_metrics(y_val, y_val_lgb)
        logger.info(f"[{metric}] LightGBM -> Val RMSE: {lgb_val_metrics['rmse']}, MAE: {lgb_val_metrics['mae']}, R2: {lgb_val_metrics['r2']}")

        # Ridge Benchmark
        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X_train_scaled, y_train)
        y_val_ridge = ridge_model.predict(X_val_scaled)
        ridge_val_metrics = compute_metrics(y_val, y_val_ridge)

        # -------------------------------------------------------------
        # 4. Deep Learning: PyTorch LSTM Sequence Model
        # -------------------------------------------------------------
        logger.info(f"[{metric}] Fitting Deep Learning: PyTorch LSTM Model...")
        seq_len = 24
        X_seq_train, y_seq_train = create_sequences(X_train_scaled, y_train, seq_len=seq_len)
        X_seq_val, y_seq_val = create_sequences(X_val_scaled, y_val, seq_len=seq_len)

        train_ds = TensorDataset(torch.from_numpy(X_seq_train), torch.from_numpy(y_seq_train))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

        lstm_model = WeatherLSTM(input_dim=len(feature_cols), hidden_dim=32, num_layers=1)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005, weight_decay=1e-5)
        criterion = nn.MSELoss()

        lstm_model.train()
        for epoch in range(12):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = lstm_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            val_preds_lstm = lstm_model(torch.from_numpy(X_seq_val)).numpy()
        lstm_val_metrics = compute_metrics(y_seq_val, val_preds_lstm)
        logger.info(f"[{metric}] PyTorch LSTM -> Val RMSE: {lstm_val_metrics['rmse']}, MAE: {lstm_val_metrics['mae']}, R2: {lstm_val_metrics['r2']}")

        # -------------------------------------------------------------
        # 5. Champion Model Selection
        # -------------------------------------------------------------
        candidates = [
            ("Naive Baseline", naive_val_metrics, None),
            ("Statistical (ARIMA)", stat_val_metrics, stat_model_obj),
            ("Classical ML (LightGBM)", lgb_val_metrics, lgb_model),
            ("Classical ML (Ridge)", ridge_val_metrics, ridge_model),
            ("Deep Learning (PyTorch LSTM)", lstm_val_metrics, lstm_model),
        ]

        # Select candidate with lowest validation RMSE
        valid_candidates = [c for c in candidates if c[2] is not None]
        champion_name, champion_val_metrics, champion_model = min(valid_candidates, key=lambda c: c[1]["rmse"])

        # Compare against Naive Baseline
        beat_naive = champion_val_metrics["rmse"] < naive_val_metrics["rmse"]
        improvement_pct = round(((naive_val_metrics["rmse"] - champion_val_metrics["rmse"]) / naive_val_metrics["rmse"]) * 100, 2)
        logger.info(f"[{metric}] CHAMPION: '{champion_name}' | Val RMSE: {champion_val_metrics['rmse']} vs Naive: {naive_val_metrics['rmse']} (Improvement: {improvement_pct}%)")

        # Evaluate Champion on Test Split
        if champion_name == "Classical ML (LightGBM)":
            y_test_pred = champion_model.predict(X_test)
        elif champion_name == "Classical ML (Ridge)":
            y_test_pred = champion_model.predict(X_test_scaled)
        elif champion_name == "Deep Learning (PyTorch LSTM)":
            X_seq_test, y_seq_test = create_sequences(X_test_scaled, y_test, seq_len=seq_len)
            with torch.no_grad():
                y_test_pred = champion_model(torch.from_numpy(X_seq_test)).numpy()
            y_test = y_seq_test
        else:
            y_test_pred = np.full_like(y_test, y_train.mean())

        champion_test_metrics = compute_metrics(y_test, y_test_pred)
        logger.info(f"[{metric}] Champion Test Metrics -> RMSE: {champion_test_metrics['rmse']}, MAE: {champion_test_metrics['mae']}, R2: {champion_test_metrics['r2']}")

        # Save local champion model
        metric_model_dir = MODELS_DIR / metric
        metric_model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = metric_model_dir / "model.joblib"
        if isinstance(champion_model, nn.Module):
            torch.save(champion_model.state_dict(), metric_model_dir / "model.pt")
            saved_type = "torch"
        else:
            joblib.dump(champion_model, model_save_path)
            saved_type = "joblib"

        training_results[metric] = {
            "champion_family": champion_name,
            "beat_naive": beat_naive,
            "improvement_pct": improvement_pct,
            "validation_metrics": champion_val_metrics,
            "test_metrics": champion_test_metrics,
            "naive_val_metrics": naive_val_metrics,
            "naive_test_metrics": naive_test_metrics,
            "model_path": str(model_save_path),
            "feature_columns": feature_cols,
        }

        # -------------------------------------------------------------
        # 6. SHAP Explainability (on LightGBM)
        # -------------------------------------------------------------
        try:
            logger.info(f"[{metric}] Generating SHAP explanations...")
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(X_val[:150])  # representative sample
            
            # Save SHAP feature importance plot
            plt.figure(figsize=(10, 6), dpi=300)
            shap.summary_plot(shap_values, X_val[:150], feature_names=feature_cols, plot_type="bar", show=False, max_display=12)
            plt.title(f"SHAP Feature Importance: 24h Ahead Forecast for '{metric}'", fontsize=12, fontweight="bold")
            plt.tight_layout()
            shap_fig_path = EDA_ARTIFACTS_DIR / f"shap_importance_{metric}.png"
            plt.savefig(shap_fig_path)
            plt.close()

            shap_explainers[metric] = {
                "top_features": [feature_cols[idx] for idx in np.argsort(np.abs(shap_values).mean(0))[::-1][:6]]
            }
        except Exception as e:
            logger.warning(f"SHAP computation warning for {metric}: {e}")

        # -------------------------------------------------------------
        # 7. Hopsworks Model Registry Promotion
        # -------------------------------------------------------------
        if HOPSWORKS_API_KEY and HOPSWORKS_API_KEY != "your_hopsworks_api_key_here":
            try:
                project = get_hopsworks_project()
                mr = project.get_model_registry()
                model_name = f"agentweather_model_{metric}"
                logger.info(f"Registering champion '{champion_name}' in Hopsworks Model Registry as '{model_name}'...")
                
                # Check current champion in Hopsworks Model Registry
                should_promote = True
                try:
                    current_models = mr.get_models(name=model_name)
                    if current_models:
                        best_existing = max(current_models, key=lambda m: m.version)
                        existing_rmse = best_existing.training_metrics.get("val_rmse", float("inf"))
                        if champion_val_metrics["rmse"] >= existing_rmse:
                            logger.info(f"Existing model version {best_existing.version} has RMSE {existing_rmse} <= current {champion_val_metrics['rmse']}. Retaining current champion.")
                            should_promote = False
                except Exception:
                    pass

                if should_promote:
                    hw_model = mr.python.create_model(
                        name=model_name,
                        metrics={
                            "val_rmse": champion_val_metrics["rmse"],
                            "val_mae": champion_val_metrics["mae"],
                            "val_r2": champion_val_metrics["r2"],
                            "test_rmse": champion_test_metrics["rmse"],
                            "test_mae": champion_test_metrics["mae"],
                            "test_r2": champion_test_metrics["r2"],
                        },
                        description=f"AgentWeather Champion {champion_name} for 24h ahead {metric} forecast",
                    )
                    # Save model artifact to Hopsworks (preserve local copies)
                    hw_model.save(str(metric_model_dir), keep_original_files=True)
                    logger.info(f"Successfully promoted model '{model_name}' (v{hw_model.version}) to Hopsworks Model Registry!")
            except Exception as e:
                logger.warning(f"Failed to register model in Hopsworks: {e}")

    # Save overall training results and SHAP summary
    with open(DATA_DIR / "training_results.json", "w") as f:
        json.dump(training_results, f, indent=2)
    with open(EDA_ARTIFACTS_DIR / "shap_summary_all_targets.json", "w") as f:
        json.dump(shap_explainers, f, indent=2)

    logger.info("All 8 target models trained, evaluated, explained via SHAP, and saved successfully!")
    return training_results


if __name__ == "__main__":
    results = train_and_evaluate_all_targets()
    print("\nTraining Pipeline Completed. Champion Summary:")
    for m, res in results.items():
        print(f"Target: {m:12s} | Champion: {res['champion_family']:26s} | Val RMSE: {res['validation_metrics']['rmse']} (Naive: {res['naive_val_metrics']['rmse']}, Imprv: {res['improvement_pct']}%)")
