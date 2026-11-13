# ⛅ AgentWeather

> **An End-to-End Serverless Weather & Air Quality Forecasting Platform Powered by a 4-Node LangGraph Agent Layer, Hopsworks Feature Store & Model Registry, and GitHub Actions CI/CD.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentweather-islamabad.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Hopsworks](https://img.shields.io/badge/Feature%20Store-Hopsworks-FF5722)](https://hopsworks.ai)
[![LangGraph](https://img.shields.io/badge/Agent%20Orchestration-LangGraph-24292e)](https://langchain.com)
[![Groq Cloud](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-f55036)](https://groq.com)

**Live Broadcast Dashboard:** [https://agentweather-islamabad.streamlit.app/](https://agentweather-islamabad.streamlit.app/)  
**Target City:** Islamabad, Pakistan (Lat: `33.6844°N`, Lon: `73.0479°E`)

---

## 🏛️ System Architecture

```
                              +---------------------------------------------+
                              |         Multi-Source Live Ingestion         |
                              |   (OpenWeatherMap + AQICN + OpenMeteo)      |
                              +---------------------------------------------+
                                                     |
                                                     v
                              +---------------------------------------------+
                              |       Source Fusion & Alignment Engine      |
                              |  - UTC Floor Hour | Temperature Averaging   |
                              |  - Dual-Stage Diurnal Imputation Strategy   |
                              +---------------------------------------------+
                                                     |
                                                     v
                              +---------------------------------------------+
                              |         Hopsworks Feature Store             |
                              |    (Feature Group: 'weather_features')      |
                              +---------------------------------------------+
                                                     |
                                                     v
                              +---------------------------------------------+
                              |         4-Node LangGraph Agent Layer        |
                              |                                             |
                              |  [Node 1: Forecast Agent (8 Champions)]     |
                              |                       |                     |
                              |  [Node 2: Alert Agent (Groq / Fallback)]    |
                              |                       |                     |
                              |  [Node 3: Drift Agent (KS & Model Drift)]   |
                              |                       |                     |
                              |  [Node 4: Weather Report Agent (Broadcast)] |
                              +---------------------------------------------+
                                 |            |                     |
                                 v            v                     v
                         +----------------+ +-----------------+  +-----------------+
                         |forecast_ledger | | agent_call_log  |  |   drift_log     |
                         | (Hopsworks FG) | | (Hopsworks FG)  |  | (Hopsworks FG)  |
                         +----------------+ +-----------------+  +-----------------+
                                                     |
                                                     v
                              +---------------------------------------------+
                              |    Continuous Retraining / CI/CD Actions    |
                              |       (workflow_dispatch REST Trigger)      |
                              +---------------------------------------------+
                                                     |
                                                     v
                              +---------------------------------------------+
                              |         Streamlit Broadcast Portal          |
                              | (News Card, Hazard Tabs, Forecasts, Panel)  |
                              +---------------------------------------------+
```

---

## 🚀 Key Features

1. **Multi-Source Sensor Fusion & Alignment:**
   - Ingests atmospheric metrics from **OpenWeatherMap** and pollutant concentrations (`aqi`, `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`) from **AQICN**.
   - Dual-layer fallback to **OpenMeteo APIs** guarantees 100% ingestion uptime.
   - All timestamps floored to the UTC hour with sensor temperature averaging.
2. **Documented Dual-Stage Imputation Strategy:**
   - Short gaps (1–3 hours): Forward fill + linear interpolation.
   - Long outages ($\ge 4$ hours): Rolling 7-day diurnal hour-of-day mean ($h \in [0, 23]$).
   - Zero nulls across all feature columns.
3. **8 Dedicated ML Models Predicting 24 Hours Ahead ($t + 24\text{h}$):**
   - Air Quality Index (`aqi`), Temperature (`temp_c`), Precipitation (`precip_mm`), Rain Probability (`rain_prob`), Humidity (`humidity_pct`), UV Index (`uv_index`), Wind Speed (`wind_speed`), and Circular Wind Direction (`wind_dir`).
   - Every champion model beat the Naive Persistence baseline on chronological validation data.
4. **4-Node LangGraph Agentic Layer:**
   - **Forecast Agent:** Ingests observations and computes $t+24\text{h}$ forecasts.
   - **Alert Agent:** Evaluates fixed hazard rules and calls **Groq LLM** (`qwen/qwen3.8-27b`) for tailored advisories (General Public, Sensitive Groups, Outdoor Workers) with deterministic template fallback.
   - **Drift Agent:** Performs two-sample Kolmogorov-Smirnov (`KS`) data drift tests and model MAE degradation checks. Dispatches GitHub Actions retraining via `workflow_dispatch` API.
   - **Weather Report Agent:** Drafts a broadcast television news script.
5. **Hopsworks Feature Store & Model Registry:**
   - Feature Group `weather_features` stores 64 engineered features and 24h targets.
   - Audit Feature Groups: `forecast_ledger`, `agent_call_log`, and `drift_log`.
   - All 8 winning models versioned and promoted in the Hopsworks Model Registry.
6. **Production CI/CD Pipelines:**
   - Hourly pipeline (`hourly_pipeline.yml`) runs ingestion and LangGraph agent workflow.
   - Daily training pipeline (`training_pipeline.yml`) retrains models and promotes new champions.

---

## 📊 Model Benchmark Results (Validation Set vs Naive Baseline)

| Target Metric | Champion Family | Val RMSE | Naive Val RMSE | Improvement (%) |
| :--- | :--- | :--- | :--- | :--- |
| **`aqi`** | **Deep Learning (PyTorch LSTM)** | **17.7583** | 18.3003 | **+2.96%** |
| **`temp_c`** | **Classical ML (Ridge)** | **1.8000** | 1.9143 | **+5.97%** |
| **`precip_mm`** | **Statistical (ARIMA)** | **0.3440** | 0.4898 | **+29.77%** |
| **`rain_prob`** | **Classical ML (Ridge)** | **17.1536** | 23.8800 | **+28.17%** |
| **`humidity_pct`** | **Classical ML (Ridge)** | **8.2488** | 10.6087 | **+22.24%** |
| **`uv_index`** | **Classical ML (LightGBM)** | **0.7157** | 0.8435 | **+15.15%** |
| **`wind_speed`** | **Classical ML (Ridge)** | **2.6841** | 3.4810 | **+22.89%** |
| **`wind_dir`** | **Classical ML (LightGBM)** | **123.1110** | 159.6295 | **+22.88%** |

---

## 📁 Repository Structure

```
.
├── .github/
│   └── workflows/
│       ├── hourly_pipeline.yml     # Hourly scheduled ingestion & LangGraph agent cycle
│       └── training_pipeline.yml   # Daily scheduled & on-demand retraining pipeline
├── data/
│   ├── eda_artifacts/              # Distribution plots, diurnal cycles, correlation heatmaps & SHAP plots
│   └── latest_agent_state.json     # Cached state of most recent agent run
├── notebooks/
│   └── 01_eda.ipynb                # Executable Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration, hazard thresholds & credentials
│   ├── features.py                 # Multi-source live ingestion, fusion & feature engineering
│   ├── backfill.py                 # Historical data extraction (8 months) & target creation
│   ├── train.py                    # Multi-family model training, SHAP explainability & registry promotion
│   ├── drift.py                    # KS test data drift & forecast error degradation monitoring
│   ├── agents.py                   # 4-node LangGraph workflow & Hopsworks audit loggers
│   └── run_hourly.py               # Hourly runner passing in-memory features to agent graph
├── app.py                          # Streamlit broadcast weather dashboard
├── requirements.txt                # Pinned production dependencies
├── REPORT.md                       # Comprehensive technical engineering report
└── README.md
```

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shehzad3684/agentweather.git
   cd agentweather
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `.env` file:**
   ```dotenv
   HOPSWORKS_API_KEY=your_hopsworks_key
   HOPSWORKS_PROJECT_NAME=agent_weather
   OPENWEATHER_API_KEY=your_openweather_key
   AQICN_API_KEY=your_aqicn_token
   GROQ_API_KEY=your_groq_key
   GH_PAT=your_github_token
   GITHUB_REPOSITORY=Shehzad3684/agentweather
   CITY_NAME=Islamabad
   CITY_LAT=33.6844
   CITY_LON=73.0479
   ```

5. **Run the platform:**
   - **Backfill historical data:** `python src/backfill.py`
   - **Train models & export SHAP:** `python src/train.py`
   - **Execute live agent cycle:** `python src/run_hourly.py`
   - **Launch Streamlit dashboard:** `streamlit run app.py`

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
