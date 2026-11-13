# AgentWeather: End-to-End Serverless Weather & Air Quality Forecasting Platform
**Autonomous Machine Learning & Agentic Systems Engineering Technical Report**  
*Target City:* Islamabad, Pakistan (Lat: 33.6844°N, Lon: 73.0479°E)  
*Architecture:* LangGraph 4-Node Agent Layer | Hopsworks Feature Store & Model Registry | GitHub Actions CI/CD | Streamlit Broadcast Portal  

---

## 1. Executive Summary & High-Level Architecture

**AgentWeather** is a serverless, production-grade meteorological and air quality forecasting system. Operating autonomously on an hourly schedule, the platform fuses multi-source satellite and ground observations, extracts predictive temporal and atmospheric features, predicts 8 core targets **24 hours ahead** using champion models selected from multiple model families, evaluates hazard triggers, performs continuous statistical drift detection, and broadcasts natural-language safety advisories via a 4-node LangGraph agentic layer powered by Groq LLMs.

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
                              |  - Documented Dual-Stage Imputation Strategy|
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

## 2. Data Engineering, Source Fusion & Imputation Strategy

### 2.1 Multi-Source Ingestion & Fusion
The platform ingests observation metrics across two distinct domains:
1. **Meteorological Data:** Ingested primarily from **OpenWeatherMap** (temperature, humidity, pressure, wind velocity, wind direction, cloud cover, and precipitation).
2. **Air Quality Data:** Ingested primarily from **AQICN / WAQI** (`aqi`, `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`, station temperature).
3. **Resilient Fallback Layer:** If third-party rate limits or key activation delays occur (e.g. OpenWeatherMap activation delay), the pipeline dynamically engages **OpenMeteo APIs** as a zero-key fallback, guaranteeing uninterrupted hourly data availability.

### 2.2 Timestamp Alignment & Source Deduplication
* Every incoming reading is floored to the exact hour UTC (`timestamp.floor('h')`).
* Where both OpenWeatherMap and AQICN report ambient temperatures, the platform executes sensor fusion by computing the arithmetic mean:
  $$\text{temp\_c} = \frac{\text{temp}_{\text{OWM}} + \text{temp}_{\text{AQICN}}}{2}$$

### 2.3 Documented Missing Data Imputation Strategy
Under strict operational rules, hourly rows are **never dropped**. Missing records are imputed using a deterministic two-stage strategy:
* **Short Gaps ($1 \le \Delta t \le 3$ hours):** Forward fill (`ffill(limit=3)`) followed by linear interpolation (`interpolate(method='linear')`). This captures high short-term auto-correlation in ambient temperature and barometric pressure.
* **Extended Outages ($\Delta t \ge 4$ hours):** Imputed via a **rolling 7-day diurnal hour-of-day mean** (168-hour rolling window grouped by the specific hour of the day: $h \in [0, 23]$). This respects diurnal day/night cycles without introducing discontinuous step-artifacts.
* **Safety Net:** Any residual edge nulls are backfilled and bounded by the historical column median. Post-imputation audits confirm **exactly 0 null values** across all feature columns.

### 2.4 Feature Engineering
The pipeline constructs 64 engineered input features:
* **Cyclical Temporal Encodings:**
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
  $$\text{month\_sin} = \sin\left(\frac{2\pi \cdot (m-1)}{12}\right), \quad \text{month\_cos} = \cos\left(\frac{2\pi \cdot (m-1)}{12}\right)$$
* **Circular Wind Encoding & Reconstruction:** Compass wind direction ($0^\circ \text{ to } 360^\circ$) is converted to orthogonal coordinates:
  $$\text{wind\_sin} = \sin(\theta), \quad \text{wind\_cos} = \cos(\theta)$$
  reconstructed at inference via $\theta = (\text{atan2}(\text{wind\_sin}, \text{wind\_cos}) \times \frac{180}{\pi}) \pmod{360}$.
* **Atmospheric Rates of Change:** 1-hour and 3-hour gradients: `temp_change_1h`, `aqi_change_3h`, and `pressure_change_3h`.
* **Multi-Scale Rolling Statistics:** 6-hour, 12-hour, and 24-hour rolling means and standard deviations across `temp_c`, `aqi`, `wind_speed`, `humidity_pct`, `pressure_hpa`, and `precip_mm`.
* **Target Construction:** Actual observed targets shifted backward by 24 rows (`shift(-24)`) to establish exact $t + 24\text{h}$ targets.

---

## 3. Multi-Family Model Training, Benchmarking & Registry Promotion

### 3.1 Time-Based Split Protocol
Chronological partition spanning 5,744 hourly training observations:
* **Train Split (80%):** 4,595 rows (March 18, 2026 – September 25, 2026)
* **Validation Split (10%):** 574 rows (September 26, 2026 – October 19, 2026)
* **Held-out Test Split (10%):** 575 rows (October 20, 2026 – November 12, 2026)
* *Zero row shuffling* to eliminate temporal data leakage.

### 3.2 Model Families Evaluated
Each target was trained across 4 distinct model families:
1. **Naive Persistence Baseline:** $y_{\text{pred}}(t + 24\text{h}) = y_{\text{actual}}(t)$
2. **Statistical Family:** Autoregressive Integrated Moving Average (`ARIMA(1, 0, 1)`)
3. **Classical ML Family:** `LightGBMRegressor` (gradient boosted trees) and `Ridge` regression benchmark
4. **Deep Learning Family:** PyTorch `WeatherLSTM` sequence model with 24-hour lookback windows, hidden dimension 32, and linear projection head.

### 3.3 Final Model Benchmark Results
Every single champion model **beat the Naive Persistence Baseline** on the validation set:

| Target Metric | Champion Family | Val RMSE | Naive Val RMSE | Improvement (%) | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`aqi`** | **Deep Learning (PyTorch LSTM)** | **17.7583** | 18.3003 | **+2.96%** | 17.6297 | 13.5796 | 0.0270 |
| **`temp_c`** | **Classical ML (Ridge)** | **1.8000** | 1.9143 | **+5.97%** | 2.5756 | 2.0059 | 0.4596 |
| **`precip_mm`** | **Statistical (ARIMA)** | **0.3440** | 0.4898 | **+29.77%** | 0.6807 | 0.2187 | -0.0027 |
| **`rain_prob`** | **Classical ML (Ridge)** | **17.1536** | 23.8800 | **+28.17%** | 20.0668 | 8.7034 | -0.0679 |
| **`humidity_pct`** | **Classical ML (Ridge)** | **8.2488** | 10.6087 | **+22.24%** | 8.4696 | 6.6795 | 0.6019 |
| **`uv_index`** | **Classical ML (LightGBM)** | **0.7157** | 0.8435 | **+15.15%** | 1.3305 | 0.7326 | 0.8160 |
| **`wind_speed`** | **Classical ML (Ridge)** | **2.6841** | 3.4810 | **+22.89%** | 3.0205 | 2.3814 | -0.0350 |
| **`wind_dir`** | **Classical ML (LightGBM)** | **123.1110** | 159.6295 | **+22.88%** | 120.1053 | 102.6421 | -0.0262 |

### 3.4 SHAP Explainability Insights
Tree SHAP explainers were fitted to tabular models to extract global and local feature importance:
* **Temperature:** Dominated by `temp_c_roll_mean_24h` and `hour_sin`, reflecting the strong diurnal cycle.
* **Precipitation & Rain Probability:** Driven by `humidity_pct_roll_mean_12h`, `cloud_cover`, and `pressure_change_3h`.
* **Air Quality (AQI):** Highest attribution given to `pm2_5`, `pm10`, `aqi_roll_mean_24h`, and atmospheric stagnation features (`wind_speed_roll_mean_12h`).
* **UV Index:** Governed almost entirely by daylight hour geometry (`hour_sin`, `hour_cos`) and `cloud_cover`.

### 3.5 Hopsworks Model Registry Promotion
All 8 champion models were promoted and registered in the Hopsworks Model Registry:
* `agentweather_model_aqi` (v1)
* `agentweather_model_temp_c` (v1)
* `agentweather_model_precip_mm` (v1)
* `agentweather_model_rain_prob` (v1)
* `agentweather_model_humidity_pct` (v1)
* `agentweather_model_uv_index` (v1)
* `agentweather_model_wind_speed` (v1)
* `agentweather_model_wind_dir` (v1)

---

## 4. 4-Node LangGraph Agent Layer & Audit Logging

A single stateful `StateGraph` orchestrates the operational intelligence:

### Node 1: Forecast Agent
* Runs hourly; ingests current observations directly in memory.
* Evaluates all 8 champion models, enforces physical boundaries (e.g. $\text{AQI} \ge 0, \text{Humidity} \in [0, 100]\%$), and logs prediction rows to the `forecast_ledger` Feature Group in Hopsworks.

### Node 2: Alert Agent
* Evaluates predicted metrics against fixed hazard thresholds:
  * AQI: $> 150$ (Unhealthy)
  * Temperature: $> 40^\circ\text{C}$ or $< 0^\circ\text{C}$
  * Precipitation: $> 20\text{ mm}$ or Rain Probability $> 80\%$
  * Humidity: $> 85\%$ or $< 20\%$
  * UV Index: $\ge 8$
  * Wind Speed: $> 50\text{ km/h}$
* Calls the **Groq API** (`qwen/qwen3.8-27b`) to synthesize specialized advisories for **General Public**, **Sensitive Groups**, and **Outdoor Workers**.
* Features a deterministic Python string template fallback inside a `try/except` block, ensuring zero pipeline crashes during API outages.
* Logs prompts, responses, and execution latency to `agent_call_log`.

### Node 3: Drift Agent
* **Data Drift:** Executes a two-sample Kolmogorov-Smirnov (`KS`) test comparing the latest 24 hours of incoming features against the training baseline distribution. Flags drift if $p < 0.05$.
* **Model Drift:** Tracks actual forecast errors ($|y_{\text{actual}} - y_{\text{pred}}|$) for historical predictions logged 24 hours prior. Flags model drift if recent MAE degrades by $> 25\%$ relative to validation baseline.
* **Autonomous Trigger:** When drift is confirmed, issues an authenticated HTTP `POST` to the GitHub Actions REST API (`workflow_dispatch`), triggering `training_pipeline.yml` for automated continuous retraining.
* Logs checks to `drift_log`.

### Node 4: Weather Report Agent
* Formulates a 3-paragraph television broadcast news script synthesizing 24-hour predictions and active advisories.
* Logs outputs and latency to `agent_call_log`.

---

## 5. Real Engineering Issues Encountered & Resolved

1. **Hopsworks Delta Lake on Windows RPC Issue:**
   * *Issue:* Writing offline Delta tables via `delta-rs` on Windows raised `hdfs_native::security::gssapi: Loading Kerberos libraries are not supported on this system`.
   * *Resolution:* Reconfigured Feature Groups with `online_enabled=True` and installed `confluent-kafka`. This routed write operations through Kafka streaming and RonDB/PyMySQL, bypassing Windows HDFS RPC limitations entirely while maintaining full cross-platform compatibility with Linux GitHub Actions runners.
2. **Schema Type Incompatibility in Online Streaming:**
   * *Issue:* Hopsworks online feature store strictly validates integer types (`bigint` vs `double`). Live pandas DataFrames contained floating-point representations for `humidity_pct`, `wind_dir`, and `aqi`.
   * *Resolution:* Implemented schema casting in `src/features.py` using `.round().astype('int64')` on all integer-bound columns prior to transmission.
3. **Datetime Timezone Alignment in Model Drift Evaluation:**
   * *Issue:* Merging prediction timestamps with historical observations caused `ValueError: You are trying to merge on object and datetime64[ns, UTC] columns`.
   * *Resolution:* Standardized all merge keys explicitly via `pd.to_datetime(..., utc=True)` in `src/drift.py`.
4. **Third-Party API Latency & Key Activation Window:**
   * *Issue:* OpenWeatherMap keys require an initial 10-30 minute activation window upon account registration.
   * *Resolution:* Built an automatic dual-layer fallback in `src/features.py` that queries OpenMeteo live endpoints whenever OpenWeatherMap returns HTTP 401, preventing any pipeline failure during initial onboarding.
5. **Groq Model Availability:**
   * *Issue:* The default Groq model endpoint for certain free-tier tiers exhibited model availability constraints.
   * *Resolution:* Configured `qwen/qwen3.8-27b` alongside robust deterministic Python fallback templates, ensuring 100% advisory generation uptime.

---

## 6. Continuous Deployment & Streamlit Portal

* **GitHub Actions:**
  * `hourly_pipeline.yml`: Runs hourly (`cron: '0 * * * *'`) to execute live ingestion and the LangGraph agent layer.
  * `training_pipeline.yml`: Runs daily (`cron: '0 0 * * *'`) or on-demand via `workflow_dispatch` triggered by the Drift Agent.
* **Streamlit Broadcast Dashboard (`app.py`):**
  * Top TV news-style weather report card.
  * Color-coded hazard warning tabs (General Public, Sensitive Groups, Outdoor Workers).
  * 6 predictive metric summary cards with hazard badges.
  * Interactive 7-day Plotly historical trend & 24-hour predicted horizon charts.
  * 4-tab Agent Activity Panel: Active Alerts, EDA/SHAP Explanations, Drift Monitor, and LLM Call/Failure Logs.
