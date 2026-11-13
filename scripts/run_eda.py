"""
Script to create, execute, and export notebooks/01_eda.ipynb
and generate all visual & statistical artifacts in data/eda_artifacts/
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

from src.config import DATA_DIR, EDA_ARTIFACTS_DIR, TARGET_METRICS, TARGET_COLUMNS, HAZARD_THRESHOLDS

EDA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
parquet_path = DATA_DIR / "historical_features.parquet"
if not parquet_path.exists():
    raise FileNotFoundError(f"Missing {parquet_path}. Run backfill first.")

df = pd.read_parquet(parquet_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df)} rows spanning {df['timestamp'].min()} to {df['timestamp'].max()}")

# 2. Generate Summary Statistics JSON
summary_metrics = [
    "temp_c", "humidity_pct", "pressure_hpa", "wind_speed", "wind_dir", "precip_mm", "rain_prob", "uv_index",
    "aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3"
]
stats_dict = {}
for m in summary_metrics:
    if m in df.columns:
        s = df[m]
        stats_dict[m] = {
            "count": int(s.count()),
            "mean": float(round(s.mean(), 2)),
            "std": float(round(s.std(), 2)),
            "min": float(round(s.min(), 2)),
            "25%": float(round(s.quantile(0.25), 2)),
            "50%": float(round(s.median(), 2)),
            "75%": float(round(s.quantile(0.75), 2)),
            "max": float(round(s.max(), 2)),
            "hazard_threshold": HAZARD_THRESHOLDS.get(m, "None"),
        }

with open(EDA_ARTIFACTS_DIR / "eda_summary_statistics.json", "w") as f:
    json.dump(stats_dict, f, indent=2)
print("Saved eda_summary_statistics.json")

# 3. Missing Data Profile (JSON + PNG)
missing_counts = df[[c for c in df.columns if not c.startswith("target_")]].isnull().sum()
missing_dict = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "feature_missing_count_post_imputation": int(missing_counts.sum()),
    "target_missing_count_newest_24h": int(df[TARGET_COLUMNS].isnull().sum().sum()),
    "imputation_strategy": "1-3h forward fill + linear interpolation; >=4h diurnal 7-day hourly average",
}
with open(EDA_ARTIFACTS_DIR / "eda_missing_data_profile.json", "w") as f:
    json.dump(missing_dict, f, indent=2)

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
features_to_check = summary_metrics + ["temp_change_1h", "aqi_change_3h", "temp_c_roll_mean_24h", "aqi_roll_mean_24h"]
null_vals = [df[c].isnull().sum() for c in features_to_check]
ax.bar(features_to_check, null_vals, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)
ax.set_title("Missing Data Audit Post-Imputation (Islamabad, Pakistan)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Null Count", fontsize=11)
ax.set_ylim(0, 10)
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(null_vals):
    ax.text(i, v + 0.2, f"{v}", ha='center', fontweight='bold', color="#27ae60")
plt.tight_layout()
plt.savefig(EDA_ARTIFACTS_DIR / "eda_missing_data_profile.png")
plt.close()
print("Saved eda_missing_data_profile.png")

# 4. Metric Distributions (6 Categories in 2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
palette = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#34495e"]

# 1. AQI
sns.histplot(df["aqi"], kde=True, ax=axes[0, 0], color=palette[0], bins=35)
axes[0, 0].axvline(150, color="darkred", linestyle="--", linewidth=1.5, label="Hazard (>150)")
axes[0, 0].set_title("1. Air Quality Index (AQI)", fontweight="bold")
axes[0, 0].set_xlabel("US EPA AQI")
axes[0, 0].legend()

# 2. Temperature
sns.histplot(df["temp_c"], kde=True, ax=axes[0, 1], color=palette[1], bins=35)
axes[0, 1].axvline(40, color="darkred", linestyle="--", linewidth=1.5, label="Heat Hazard (>40°C)")
axes[0, 1].axvline(0, color="blue", linestyle="--", linewidth=1.5, label="Freeze Hazard (<0°C)")
axes[0, 1].set_title("2. Temperature (°C)", fontweight="bold")
axes[0, 1].set_xlabel("Degrees Celsius")
axes[0, 1].legend()

# 3. Precipitation & Rain Prob
sns.histplot(df[df["precip_mm"] > 0]["precip_mm"], kde=True, ax=axes[0, 2], color=palette[2], bins=25)
axes[0, 2].axvline(20, color="darkred", linestyle="--", linewidth=1.5, label="Hazard (>20mm)")
axes[0, 2].set_title("3. Precipitation (mm, non-zero hours)", fontweight="bold")
axes[0, 2].set_xlabel("Rainfall (mm)")
axes[0, 2].legend()

# 4. Humidity
sns.histplot(df["humidity_pct"], kde=True, ax=axes[1, 0], color=palette[3], bins=30)
axes[1, 0].axvline(85, color="darkred", linestyle="--", linewidth=1.5, label="High Hazard (>85%)")
axes[1, 0].axvline(20, color="orange", linestyle="--", linewidth=1.5, label="Dry Hazard (<20%)")
axes[1, 0].set_title("4. Relative Humidity (%)", fontweight="bold")
axes[1, 0].set_xlabel("Percentage (%)")
axes[1, 0].legend()

# 5. UV Index
sns.histplot(df[df["uv_index"] > 0]["uv_index"], kde=True, ax=axes[1, 1], color=palette[4], bins=20)
axes[1, 1].axvline(8, color="darkred", linestyle="--", linewidth=1.5, label="Very High UV (>=8)")
axes[1, 1].set_title("5. UV Index (Daylight Hours)", fontweight="bold")
axes[1, 1].set_xlabel("UV Scale (0 to 11+)")
axes[1, 1].legend()

# 6. Wind Speed
sns.histplot(df["wind_speed"], kde=True, ax=axes[1, 2], color=palette[5], bins=30)
axes[1, 2].axvline(50, color="darkred", linestyle="--", linewidth=1.5, label="Gale Hazard (>50 km/h)")
axes[1, 2].set_title("6. Wind Speed (km/h)", fontweight="bold")
axes[1, 2].set_xlabel("Speed (km/h)")
axes[1, 2].legend()

plt.suptitle("AgentWeather Exploratory Distributions (Islamabad, Pakistan)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(EDA_ARTIFACTS_DIR / "eda_metric_distributions.png")
plt.close()
print("Saved eda_metric_distributions.png")

# 5. Diurnal / Hourly Seasonality (24-hour cycle)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)
hour_grp = df.groupby("hour")

# Temp diurnal
axes[0, 0].plot(hour_grp["temp_c"].mean().index, hour_grp["temp_c"].mean().values, marker="o", color="#e67e22", label="Mean Temp")
axes[0, 0].fill_between(
    hour_grp["temp_c"].mean().index,
    hour_grp["temp_c"].mean().values - hour_grp["temp_c"].std().values,
    hour_grp["temp_c"].mean().values + hour_grp["temp_c"].std().values,
    color="#e67e22", alpha=0.2, label="±1 Std Dev"
)
axes[0, 0].set_title("Hourly Temperature Diurnal Cycle (°C)", fontweight="bold")
axes[0, 0].set_xlabel("Hour of Day (UTC)")
axes[0, 0].set_ylabel("°C")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend()

# AQI diurnal
axes[0, 1].plot(hour_grp["aqi"].mean().index, hour_grp["aqi"].mean().values, marker="s", color="#e74c3c", label="Mean AQI")
axes[0, 1].fill_between(
    hour_grp["aqi"].mean().index,
    hour_grp["aqi"].mean().values - hour_grp["aqi"].std().values,
    hour_grp["aqi"].mean().values + hour_grp["aqi"].std().values,
    color="#e74c3c", alpha=0.2, label="±1 Std Dev"
)
axes[0, 1].set_title("Hourly AQI Cycle (US EPA)", fontweight="bold")
axes[0, 1].set_xlabel("Hour of Day (UTC)")
axes[0, 1].set_ylabel("AQI")
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend()

# Humidity diurnal
axes[1, 0].plot(hour_grp["humidity_pct"].mean().index, hour_grp["humidity_pct"].mean().values, marker="^", color="#2ecc71", label="Mean Humidity")
axes[1, 0].fill_between(
    hour_grp["humidity_pct"].mean().index,
    hour_grp["humidity_pct"].mean().values - hour_grp["humidity_pct"].std().values,
    hour_grp["humidity_pct"].mean().values + hour_grp["humidity_pct"].std().values,
    color="#2ecc71", alpha=0.2, label="±1 Std Dev"
)
axes[1, 0].set_title("Hourly Humidity Cycle (%)", fontweight="bold")
axes[1, 0].set_xlabel("Hour of Day (UTC)")
axes[1, 0].set_ylabel("%")
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend()

# Wind & UV diurnal
ax_uv = axes[1, 1]
ax_wind = ax_uv.twinx()
l1 = ax_uv.plot(hour_grp["uv_index"].mean().index, hour_grp["uv_index"].mean().values, marker="o", color="#9b59b6", label="Mean UV Index")
l2 = ax_wind.plot(hour_grp["wind_speed"].mean().index, hour_grp["wind_speed"].mean().values, marker="d", color="#34495e", linestyle="--", label="Mean Wind Speed (km/h)")
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax_uv.legend(lines, labels, loc="upper right")
ax_uv.set_title("Hourly UV Index and Wind Speed Cycle", fontweight="bold")
ax_uv.set_xlabel("Hour of Day (UTC)")
ax_uv.set_ylabel("UV Index", color="#9b59b6")
ax_wind.set_ylabel("Wind Speed (km/h)", color="#34495e")
ax_uv.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("AgentWeather Diurnal Seasonality Patterns (24h Profiles)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(EDA_ARTIFACTS_DIR / "eda_hourly_diurnal_seasonality.png")
plt.close()
print("Saved eda_hourly_diurnal_seasonality.png")

# 6. Feature Correlation Heatmap with 24h Targets
corr_cols = [
    "temp_c", "humidity_pct", "pressure_hpa", "wind_speed", "wind_sin", "wind_cos", "precip_mm", "uv_index", "aqi",
    "target_temp_c", "target_humidity_pct", "target_wind_speed", "target_aqi", "target_precip_mm"
]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Feature & 24h Target Correlation Heatmap (Islamabad)", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(EDA_ARTIFACTS_DIR / "eda_correlation_heatmap.png")
plt.close()
print("Saved eda_correlation_heatmap.png")


# 7. Build executable Jupyter Notebook: notebooks/01_eda.ipynb
nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""# AgentWeather: Exploratory Data Analysis (EDA)
**City Target:** Islamabad, Pakistan  
**Coverage:** 8 Months of Hourly Observations (~5,800 records)  
**Primary Metrics Analyzed:**
1. Air Quality Index (AQI) & Pollutants (`pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`)
2. Temperature (`temp_c`)
3. Precipitation (`precip_mm`) & Rain Probability (`rain_prob`)
4. Relative Humidity (`humidity_pct`)
5. UV Radiation (`uv_index`)
6. Wind Speed (`wind_speed`) & Circular Direction (`wind_sin`, `wind_cos`)
"""))

nb.cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)

# Load historical features
data_path = Path("../data/historical_features.parquet")
df = pd.read_parquet(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("Dataset Shape:", df.shape)
print("Date Range:", df["timestamp"].min(), "to", df["timestamp"].max())
df.head()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Missing Data & Imputation Audit
Verification that all 1-3 hour gaps were resolved via forward fill + linear interpolation,
and longer gaps via 7-day diurnal hour-of-day averaging.
"""))

nb.cells.append(nbf.v4.new_code_cell("""feature_cols = [c for c in df.columns if not c.startswith("target_")]
null_counts = df[feature_cols].isnull().sum()
print("Total nulls in feature columns:", null_counts.sum())
assert null_counts.sum() == 0, "Error: Unexpected nulls found!"
print("Imputation check PASSED: Exactly 0 missing values.")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Statistical Distributions Across 6 Target Categories
Visualizing probability densities, hazard thresholds, and skewness across all core weather and air quality variables.
"""))

nb.cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sns.histplot(df["aqi"], kde=True, ax=axes[0, 0], color="#e74c3c", bins=30)
axes[0, 0].axvline(150, color="darkred", linestyle="--", label="Hazard (>150)")
axes[0, 0].set_title("AQI Distribution")
axes[0, 0].legend()

sns.histplot(df["temp_c"], kde=True, ax=axes[0, 1], color="#e67e22", bins=30)
axes[0, 1].axvline(40, color="darkred", linestyle="--", label="Heat Hazard (>40°C)")
axes[0, 1].set_title("Temperature Distribution (°C)")
axes[0, 1].legend()

sns.histplot(df[df["precip_mm"] > 0]["precip_mm"], kde=True, ax=axes[0, 2], color="#3498db", bins=20)
axes[0, 2].set_title("Precipitation (mm, wet hours)")

sns.histplot(df["humidity_pct"], kde=True, ax=axes[1, 0], color="#2ecc71", bins=30)
axes[1, 0].axvline(85, color="darkred", linestyle="--", label="Hazard (>85%)")
axes[1, 0].set_title("Humidity Distribution (%)")
axes[1, 0].legend()

sns.histplot(df[df["uv_index"] > 0]["uv_index"], kde=True, ax=axes[1, 1], color="#9b59b6", bins=15)
axes[1, 1].axvline(8, color="darkred", linestyle="--", label="Hazard (>=8)")
axes[1, 1].set_title("UV Index Distribution")
axes[1, 1].legend()

sns.histplot(df["wind_speed"], kde=True, ax=axes[1, 2], color="#34495e", bins=30)
axes[1, 2].axvline(50, color="darkred", linestyle="--", label="Hazard (>50 km/h)")
axes[1, 2].set_title("Wind Speed Distribution (km/h)")
axes[1, 2].legend()

plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Diurnal Seasonality Patterns (24-Hour Day/Night Cycles)
Evaluating diurnal periodicity across temperature, air quality, humidity, UV radiation, and wind velocity.
"""))

nb.cells.append(nbf.v4.new_code_cell("""hour_grp = df.groupby("hour")
plt.figure(figsize=(12, 5))
plt.plot(hour_grp["temp_c"].mean().index, hour_grp["temp_c"].mean().values, marker="o", color="#e67e22", label="Mean Temp (°C)")
plt.plot(hour_grp["aqi"].mean().index, hour_grp["aqi"].mean().values, marker="s", color="#e74c3c", label="Mean AQI")
plt.title("Diurnal Cycle: Temperature & AQI vs Hour of Day (UTC)")
plt.xlabel("Hour of Day (UTC)")
plt.legend()
plt.grid(True)
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Correlation with 24-Hour Future Targets
Analysis of current meteorological features versus targets shifted 24 hours ahead (`target_<metric> = shift(-24)`).
"""))

nb.cells.append(nbf.v4.new_code_cell("""target_corrs = df[["temp_c", "humidity_pct", "pressure_hpa", "wind_speed", "aqi", "target_temp_c", "target_humidity_pct", "target_wind_speed", "target_aqi"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(target_corrs, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Present Observations vs 24h Ahead Targets")
plt.show()
"""))

notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "01_eda.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated notebook at {notebook_path}")
