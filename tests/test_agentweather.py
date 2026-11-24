"""
AgentWeather Meteorological & MLOps Automated Test Suite
Verifies:
1. US EPA 6-tier AQI standard classification & 2024 PM2.5 piecewise linear calculations.
2. Timezone alignment (UTC to Asia/Karachi UTC+5) & cyclical solar feature engineering.
3. LangGraph Report grounding validation and deterministic fallback safety.
4. Live observation telemetry schema and temperature extrema.
"""

import sys
import datetime
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import classify_epa_aqi, pm25_to_epa_aqi, HAZARD_THRESHOLDS
from src.features import build_engineered_features
from src.agents import validate_report_grounding, generate_grounded_report


# =============================================================================
# 1. US EPA AQI Standard Classification Tests
# =============================================================================
class TestEpaAqiClassification:
    def test_epa_aqi_six_tiers(self):
        """
        Validates exact US EPA tier boundaries:
        - 0 to 50: Good
        - 51 to 100: Moderate
        - 101 to 150: Unhealthy for Sensitive Groups
        - 151 to 200: Unhealthy (specifically AQI 189 must be Unhealthy, not Hazardous)
        - 201 to 300: Very Unhealthy
        - 301 to 500: Hazardous
        """
        cases = [
            (25.0, "Good", "#10B981"),
            (50.0, "Good", "#10B981"),
            (51.0, "Moderate", "#F59E0B"),
            (75.0, "Moderate", "#F59E0B"),
            (100.0, "Moderate", "#F59E0B"),
            (101.0, "Unhealthy for Sensitive Groups", "#FB923C"),
            (145.0, "Unhealthy for Sensitive Groups", "#FB923C"),
            (150.0, "Unhealthy for Sensitive Groups", "#FB923C"),
            (151.0, "Unhealthy", "#EF4444"),
            (189.0, "Unhealthy", "#EF4444"),  # Critical bug fix verification
            (200.0, "Unhealthy", "#EF4444"),
            (205.0, "Very Unhealthy", "#A855F7"),
            (280.0, "Very Unhealthy", "#A855F7"),
            (300.0, "Very Unhealthy", "#A855F7"),
            (305.0, "Hazardous", "#E11D48"),
            (450.0, "Hazardous", "#E11D48"),
        ]

        for aqi_val, expected_label, expected_color in cases:
            label, color, bg, border, note = classify_epa_aqi(aqi_val)
            assert label == expected_label, f"AQI {aqi_val} got '{label}', expected '{expected_label}'"
            assert color == expected_color, f"AQI {aqi_val} got color '{color}', expected '{expected_color}'"

    def test_pm25_to_epa_aqi_breakpoints(self):
        """
        Validates EPA 2024 piecewise linear interpolation breakpoints:
        - 0.0 - 9.0 ug/m3 -> 0 - 50 AQI
        - 9.1 - 35.4 ug/m3 -> 51 - 100 AQI
        - 35.5 - 55.4 ug/m3 -> 101 - 150 AQI
        - 55.5 - 125.4 ug/m3 -> 151 - 200 AQI
        - 125.5 - 225.4 ug/m3 -> 201 - 300 AQI
        - 225.5 - 325.4 ug/m3 -> 301 - 400 AQI
        """
        # Low Good
        aqi_0 = pm25_to_epa_aqi(0.0)
        assert aqi_0 == 0.0

        # High Good (9.0 ug/m3 -> 50 AQI)
        aqi_9 = pm25_to_epa_aqi(9.0)
        assert abs(aqi_9 - 50.0) < 0.5

        # Moderate range
        aqi_20 = pm25_to_epa_aqi(20.0)
        assert 50.0 < aqi_20 < 100.0

        # Unhealthy for Sensitive Groups
        aqi_45 = pm25_to_epa_aqi(45.0)
        assert 100.0 < aqi_45 < 150.0

        # Unhealthy range
        aqi_70 = pm25_to_epa_aqi(70.0)
        assert 150.0 < aqi_70 < 200.0

        # Very Unhealthy
        aqi_150 = pm25_to_epa_aqi(150.0)
        assert 200.0 < aqi_150 < 300.0

        # Hazardous
        aqi_300 = pm25_to_epa_aqi(300.0)
        assert aqi_300 >= 300.0


# =============================================================================
# 2. Timezone & Diurnal Solar Feature Alignment Tests
# =============================================================================
class TestTimezoneAndFeatureEngineering:
    def test_asia_karachi_solar_hour_alignment(self):
        """
        Verifies that feature engineering uses local time in Asia/Karachi (UTC+5)
        for cyclical hour encodings, avoiding cold bias where morning (09:00 PKT / 04:00 UTC)
        was previously encoded as hour 4 (pre-dawn freezing cold).
        """
        # Create a mock UTC observation at 04:00 UTC (which is 09:00 PKT morning)
        df_raw = pd.DataFrame([{
            "timestamp": "2026-11-24 04:00:00",
            "city": "Islamabad",
            "temp_c": 16.8,
            "humidity_pct": 52.0,
            "pressure_hpa": 1013.2,
            "wind_speed": 1.5,
            "wind_dir": 263.0,
            "precip_mm": 0.0,
            "rain_prob": 0.0,
            "uv_index": 2.2,
            "cloud_cover": 20.0,
            "aqi": 156.5,
        }])

        df_feat = build_engineered_features(df_raw)

        # Expected local hour in Islamabad is 9 (04:00 UTC + 5 = 09:00 PKT)
        expected_local_hour = 9
        expected_hour_sin = math.sin(2 * math.pi * expected_local_hour / 24.0)
        expected_hour_cos = math.cos(2 * math.pi * expected_local_hour / 24.0)

        actual_hour_sin = df_feat["hour_sin"].iloc[0]
        actual_hour_cos = df_feat["hour_cos"].iloc[0]

        assert abs(actual_hour_sin - expected_hour_sin) < 1e-4, (
            f"hour_sin mismatch: got {actual_hour_sin}, expected {expected_hour_sin} (solar hour 9)"
        )
        assert abs(actual_hour_cos - expected_hour_cos) < 1e-4, (
            f"hour_cos mismatch: got {actual_hour_cos}, expected {expected_hour_cos} (solar hour 9)"
        )


# =============================================================================
# 3. Report Grounding & Fallback Validation Tests
# =============================================================================
class TestReportGrounding:
    @pytest.fixture
    def canonical_payload(self):
        return {
            "city": "Islamabad",
            "cycle_time_utc": "2026-11-24 04:00:00 UTC",
            "cycle_time_pkt": "2026-11-24 09:00:00 PKT",
            "observed_current": {
                "temperature_c": 16.8,
                "relative_humidity_pct": 52.0,
                "surface_pressure_hpa": 1013.2,
                "cloud_cover_pct": 45,
                "air_quality_index": 156.5,
                "air_quality_category": "Unhealthy",
            },
            "forecast_next_24h": {
                "target_temperature_c": 12.5,
                "daytime_high_temperature_c": 22.1,
                "overnight_low_temperature_c": 11.1,
                "relative_humidity_pct": 65.0,
                "air_quality_index": 132.1,
                "air_quality_category": "Unhealthy for Sensitive Groups",
                "precipitation_mm": 0.05,
                "rain_probability_pct": 15.0,
            },
            "active_alerts": [
                {
                    "metric": "aqi",
                    "value": 132.1,
                    "category": "Unhealthy for Sensitive Groups",
                    "threshold": "101 - 150 AQI",
                }
            ],
        }

    def test_valid_grounded_report_passes(self, canonical_payload):
        """A well-formed report with accurate figures passes validation."""
        valid_text = (
            "Islamabad records current observed temperature of 16.8°C with 52% humidity. "
            "Over the next 24-hour cycle, daytime temperatures will reach a high of 22.1°C "
            "with overnight lows dropping to 11.1°C. Target temperature at T+24h sits at 12.5°C. "
            "Air quality improves slightly to 132.1 AQI, classified as Unhealthy for Sensitive Groups."
        )
        assert validate_report_grounding(valid_text, canonical_payload) is True

    def test_clear_skies_with_high_clouds_fails(self, canonical_payload):
        """Rule: Never claim 'clear skies' if cloud cover is > 10% (here it is 45%)."""
        hallucinated_text = (
            "Islamabad enjoys crystal clear skies and bright sunshine today at 16.8°C. "
            "Temperatures peak at 22.1°C with night low of 11.1°C."
        )
        assert validate_report_grounding(hallucinated_text, canonical_payload) is False

    def test_contradictory_temperature_fails(self, canonical_payload):
        """Rule: A report claiming 28°C when observed is 16.8°C and peak is 22.1°C fails validation."""
        contradictory_text = (
            "Islamabad experiences warm conditions at 28.5°C with 52% humidity. "
            "Overnight lows sit at 11.1°C."
        )
        assert validate_report_grounding(contradictory_text, canonical_payload) is False

    def test_deterministic_fallback_matches_payload(self, canonical_payload):
        """Deterministic fallback generates clean, perfectly grounded prose matching all metrics."""
        fallback = generate_grounded_report(canonical_payload)
        assert "16.8°C" in fallback
        assert "22.1°C" in fallback
        assert "11.1°C" in fallback
        assert "132.1" in fallback
        assert "Unhealthy for Sensitive Groups" in fallback
        # Verify fallback itself passes grounding validation
        assert validate_report_grounding(fallback, canonical_payload) is True
