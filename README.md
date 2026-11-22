<div align="center">
  <a href="https://agentweather%2Dislamabad.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" height="52" alt="Streamlit Cloud Application">
  </a>
  <br>
  <strong>Public Live Deployment: Islamabad, Pakistan</strong>
</div>

# AgentWeather: Autonomous Weather and Air Quality Platform

An enterprise serverless weather forecasting and environmental intelligence platform for Islamabad, Pakistan. The architecture combines automated multi source data ingestion, dual stage diurnal gap imputation, Hopsworks Feature Store and Model Registry management, a four node LangGraph agent layer, and GitHub Actions continuous retraining.

<hr>

## Executive Summary

AgentWeather generates continuous predictions for eight distinct atmospheric and pollutant targets at a fixed horizon of twenty four hours ahead. Every target variable is forecasted by an independently selected champion model family that statistically outperforms naive persistence on chronological validation partitions.

The intelligence layer is orchestrated as a unified StateGraph in LangGraph, executing hourly inference, evaluating municipal hazard rules, generating structured multi audience safety advisories via Groq language models, monitoring statistical data and model drift, and drafting television style weather broadcasts.

<hr>

## System Architecture

The platform operates across four coordinated stages:

```
[ Stage 1: Ingestion and Fusion ]
  OpenWeatherMap (Meteorology Primary) + AQICN (Air Quality Primary)
  Dual layer fallback to OpenMeteo APIs for zero downtime reliability
  Floored to exact UTC hour with multi sensor temperature averaging
                         |
                         v
[ Stage 2: Feature Engineering and Feature Store ]
  Dual stage imputation: forward fill plus rolling seven day diurnal mean
  Cyclical sine and cosine encodings for diurnal and seasonal periodicity
  Hopsworks Feature Group: weather_features (online and offline stores)
                         |
                         v
[ Stage 3: Four Node LangGraph Agent Layer ]
  Node 1: Forecast Agent (Generates t plus 24 hour predictions)
  Node 2: Alert Agent (Hazard rule checks and Groq public advisories)
  Node 3: Drift Agent (Two sample Kolmogorov Smirnov and MAE tests)
  Node 4: Weather Report Agent (Broadcast news script synthesis)
                         |
                         v
[ Stage 4: Automated Auditing and Presentation ]
  Hopsworks Audit Feature Groups: forecast_ledger, agent_call_log, drift_log
  Streamlit Cloud live public interface with Google Antigravity styling
  Continuous retraining triggers dispatched via GitHub REST API
```

<hr>

## Target Variables and Operational Thresholds

Predictions are generated for a horizon of twenty four hours into the future. Each target is bound by physical meteorological limits and evaluated against fixed municipal safety rules:

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Target Identifier</th>
      <th>Measurement Unit</th>
      <th>Hazard Rule Condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Air Quality</td>
      <td>aqi</td>
      <td>US EPA Scale (0 to 500)</td>
      <td>Value above 150</td>
    </tr>
    <tr>
      <td>Temperature</td>
      <td>temp_c</td>
      <td>Degrees Celsius</td>
      <td>Value below 0 or above 40</td>
    </tr>
    <tr>
      <td>Precipitation</td>
      <td>precip_mm</td>
      <td>Millimeters</td>
      <td>Value above 20</td>
    </tr>
    <tr>
      <td>Rain Probability</td>
      <td>rain_prob</td>
      <td>Percentage (0 to 100)</td>
      <td>Value above 80</td>
    </tr>
    <tr>
      <td>Humidity</td>
      <td>humidity_pct</td>
      <td>Percentage (0 to 100)</td>
      <td>Value below 20 or above 85</td>
    </tr>
    <tr>
      <td>Ultraviolet Radiation</td>
      <td>uv_index</td>
      <td>Index Scale (0 to 11 plus)</td>
      <td>Value at or above 8</td>
    </tr>
    <tr>
      <td>Wind Velocity</td>
      <td>wind_speed</td>
      <td>Kilometers per hour</td>
      <td>Value above 50</td>
    </tr>
    <tr>
      <td>Wind Direction</td>
      <td>wind_dir</td>
      <td>Compass Degrees (0 to 360)</td>
      <td>Circular trigonometric encoding</td>
    </tr>
  </tbody>
</table>

<hr>

## LangGraph Four Node Agent Layer

The decision making pipeline runs as a unified LangGraph StateGraph sharing an AgentState dictionary:

* **Node 1: Forecast Agent**
  Loads the newest engineered feature vector and candidate champion model binaries. Computes predictions for all eight metrics at twenty four hours ahead, validates physical bounds, and writes entries to the forecast ledger.

* **Node 2: Alert Agent**
  Checks all predicted values against municipal hazard limits. If any condition triggers, prompts Groq language models to construct structured advisories tailored separately for the General Public, Sensitive Groups (respiratory or cardiovascular vulnerabilities), and Outdoor Workers. Includes deterministic template fallback protection to guarantee pipeline continuity during network delays.

* **Node 3: Drift Agent**
  Conducts two sample Kolmogorov Smirnov tests comparing recent twenty four hour feature distributions against historical training reference windows. Simultaneously tracks model prediction error degradation by comparing previous forecasts with newly observed ground truth. Dispatches automated GitHub Actions retraining workflows when drift exceeds significance thresholds.

* **Node 4: Weather Report Agent**
  Synthesizes twenty four hour forecasts and active hazard alerts into a three paragraph journalistic broadcast weather dispatch suitable for television or public media distribution.

<hr>

## Hopsworks Feature Store and Model Registry

All feature transformations and audit trails are persisted into Hopsworks:

* **weather_features:** Primary feature group containing sixty four engineered input features and eight future shifted targets with online storage enabled for low latency serving.
* **forecast_ledger:** Immutable audit log tracking prediction timestamp, target timestamp, predicted values, champion family, and model version.
* **agent_call_log:** Complete record of language model invocations, tracking prompt tokens, raw completions, execution latency, and success status.
* **drift_log:** Historical repository of Kolmogorov Smirnov statistics, probability values, model error degradation rates, and automated dispatch actions.
* **Model Registry:** Houses all versioned candidate models with validation RMSE, MAE, and R2 metadata. Models are promoted to production status only when beating the incumbent champion.

<hr>

## Model Benchmark Results

Models were evaluated across five candidate families using chronological eighty percent training, ten percent validation, and ten percent testing splits. The table below presents validation Root Mean Squared Error (RMSE) of winning champions compared directly against the Naive Persistence baseline:

<table>
  <thead>
    <tr>
      <th>Target Variable</th>
      <th>Winning Model Family</th>
      <th>Validation RMSE</th>
      <th>Naive Baseline RMSE</th>
      <th>Accuracy Improvement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Air Quality Index (aqi)</td>
      <td>Deep Learning (PyTorch LSTM)</td>
      <td>17.7583</td>
      <td>18.3003</td>
      <td>Plus 2.96 percent</td>
    </tr>
    <tr>
      <td>Temperature (temp_c)</td>
      <td>Classical ML (Ridge Regression)</td>
      <td>1.8000</td>
      <td>1.9143</td>
      <td>Plus 5.97 percent</td>
    </tr>
    <tr>
      <td>Precipitation (precip_mm)</td>
      <td>Statistical (ARIMA)</td>
      <td>0.3440</td>
      <td>0.4898</td>
      <td>Plus 29.77 percent</td>
    </tr>
    <tr>
      <td>Rain Probability (rain_prob)</td>
      <td>Classical ML (Ridge Regression)</td>
      <td>17.1536</td>
      <td>23.8800</td>
      <td>Plus 28.17 percent</td>
    </tr>
    <tr>
      <td>Humidity (humidity_pct)</td>
      <td>Classical ML (Ridge Regression)</td>
      <td>8.2488</td>
      <td>10.6087</td>
      <td>Plus 22.24 percent</td>
    </tr>
    <tr>
      <td>UV Index (uv_index)</td>
      <td>Classical ML (LightGBM)</td>
      <td>0.7157</td>
      <td>0.8435</td>
      <td>Plus 15.15 percent</td>
    </tr>
    <tr>
      <td>Wind Speed (wind_speed)</td>
      <td>Classical ML (Ridge Regression)</td>
      <td>2.6841</td>
      <td>3.4810</td>
      <td>Plus 22.89 percent</td>
    </tr>
    <tr>
      <td>Wind Direction (wind_dir)</td>
      <td>Classical ML (LightGBM)</td>
      <td>123.1110</td>
      <td>159.6295</td>
      <td>Plus 22.88 percent</td>
    </tr>
  </tbody>
</table>

<hr>

## Continuous Integration and Deployment Workflows

Automated serverless operations are governed by two GitHub Actions workflows:

1. **Hourly Ingestion and Agent Cycle (`hourly_pipeline.yml`):**
   Executes at the top of every hour via cron schedule. Pulls live observations from OpenWeatherMap and AQICN, synchronizes engineered features to Hopsworks, invokes the four node LangGraph agent graph, and logs all telemetry to audit stores.

2. **Daily Multi Family Retraining Pipeline (`training_pipeline.yml`):**
   Executes daily at midnight UTC and on demand via automated drift dispatch triggers. Refreshes feature store caches, retrains all model families, updates SHAP global explainability artifacts, and promotes improved models to the Hopsworks Registry.

<hr>

## Installation and Local Reproduction

Follow these steps to replicate the environment and execute the pipeline locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shehzad3684/agentweather.git
   cd agentweather
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python scripts/setup_environment.py
   ```

   On Windows:
   ```cmd
   venv\Scripts\activate
   ```

   On Linux or macOS:
   ```bash
   source venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   python scripts/install_dependencies.py
   ```

4. **Establish configuration variables in `.env`:**
   ```env
   HOPSWORKS_API_KEY=your_hopsworks_api_key
   HOPSWORKS_PROJECT_NAME=agent_weather
   OPENWEATHER_API_KEY=your_openweather_api_key
   AQICN_API_KEY=your_aqicn_api_key
   GROQ_API_KEY=your_groq_api_key
   GH_PAT=your_github_personal_access_token
   GITHUB_REPOSITORY=Shehzad3684/agentweather
   CITY_NAME=Islamabad
   CITY_LAT=33.6844
   CITY_LON=73.0479
   ```

5. **Execute platform modules:**
   * Run historical backfill and feature store ingestion:
     ```bash
     python src/backfill.py
     ```
   * Execute multi family training and export SHAP artifacts:
     ```bash
     python src/train.py
     ```
   * Execute live hourly agent cycle:
     ```bash
     python src/run_hourly.py
     ```
   * Launch public broadcast dashboard:
     ```bash
     streamlit run app.py
     ```

<hr>

## Project Directory Organization

```
agentweather/
  .github/
    workflows/
      hourly_pipeline.yml
      training_pipeline.yml
  data/
    eda_artifacts/
      eda_correlation_heatmap.png
      eda_hourly_diurnal_seasonality.png
      eda_metric_distributions.png
      eda_missing_data_profile.png
      shap_importance_aqi.png
      shap_importance_temp_c.png
      shap_summary_all_targets.json
    latest_agent_state.json
    training_results.json
  notebooks/
    01_eda.ipynb
  scripts/
    install_dependencies.py
    setup_environment.py
    run_eda.py
  src/
    __init__.py
    config.py
    features.py
    backfill.py
    train.py
    drift.py
    agents.py
    run_hourly.py
  app.py
  requirements.txt
  REPORT.md
  README.md
```

<hr>

## License

This project is licensed under the terms of the MIT License.
