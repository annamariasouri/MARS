# Methodology

## System Overview

**Operational pipeline:**  
The system executes three stages daily: (1) **Data ingestion** downloads Copernicus Marine products (nutrients: `cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m`; temperature: `cmems_mod_med_phy-tem_anfc_4.2km_P1D-m`; salinity: `cmems_mod_med_phy-sal_anfc_4.2km_P1D-m`; chlorophyll: `cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m`) at 1–5m depth for the past 30 days. (2) **Feature engineering** computes lag-1, 7-day/30-day rolling averages, nutrient ratios, and anomalies per grid cell, writing model-ready inputs. (3) **Forecasting** loads a trained Random Forest, predicts chlorophyll for **days 0–30 (31 daily forecasts)**, and writes results to `forecast_log_*.csv` for dashboard visualization.

**Failure points and mitigations:**  
API timeouts trigger exponential-backoff retries (up to 8 attempts). Masked coastal pixels are handled via adaptive bounding-box expansion (±0.0°, ±0.05°, ±0.1°, ±0.25°, ±0.5°) until valid measurements are found; the successful expansion level is recorded in `report_{region}_{date}.txt` (field: `expansion_used`). Missing days fall back to the most recent valid input within a 7-day lookback window; if unavailable, median imputation uses the trailing 30-day window. Days without fallback data skip forecasting and log `.no_rows` markers.

**Model specification:**  
The predictor is a scikit-learn Random Forest (`final_rf_chl_model_2015_2023.pkl`) trained on 2015–2023 data with 24 features: current-day nutrients/physics (NH₄, NO₃, PO₄, salinity, temperature), lag-1 and 7-day rolling averages for all six variables, nutrient ratios (N:P, N:NH₄, P:NH₄), 30-day chlorophyll median, anomaly, and a bloom-proxy label. Hyperparameters and random seeds are not externally documented; re-training would require the original training script (not included in this repository).

**Alert generation:**  
Predictions are aggregated as the spatial mean across all grid cells per region. The alert threshold is the **90th percentile** of the predicted chlorophyll distribution for that region-day. A binary risk flag activates when `predicted_chl ≥ threshold`, and a continuous risk score is computed as `min(100, (predicted_chl / threshold) × 100)`. No temporal smoothing or hysteresis is applied; each day's alert is independent. To contextualize alerts, the dashboard displays 7-day and 30-day risk frequencies (count of flagged days in those windows).

**Alert-fatigue mitigation:**  
Percentile-based thresholds adapt to local chlorophyll distributions, reducing false positives in naturally high-chlorophyll regions. Rolling-window risk counts (7/30 days) provide trend context rather than isolated binary flags. However, day-to-day volatility in predictions can still trigger frequent alerts; future work may add 3-day moving-average smoothing or require 2 consecutive flagged days before escalating to "high alert."

**Artifacts produced:**  
Daily runs generate: `env_history_{region}_{date}.csv` (raw Copernicus extracts), `model_ready_input_{region}_{date}.csv` (engineered features), `forecast_log_{region}.csv` (cumulative predictions), and `report_{region}_{date}.txt` (download metadata including bbox expansion level). A Streamlit dashboard (`dashboard_app.py`) reads these files to display time-series plots, KPIs (latest chlorophyll, 7/30-day risk counts), and interactive maps.

**Novel contributions:**  
Unlike batch ML workflows, this is an **operational forecasting system** with daily automated data ingestion and real-time updates. Adaptive bbox expansion addresses coastal masking in high-resolution ocean models—a common failure mode ignored by generic pipelines. Multi-region pooled training improves generalization across heterogeneous Mediterranean ecosystems (open gulf vs. enclosed port). Ground truth for evaluation is the same Copernicus product family, enabling retrospective validation once target-date observations become available.

## Data Splits

**Time-based train/validation/test split:**  
Training data spans **2015-01-01 to 2023-12-31** (9 years). Validation used a held-out portion of 2023 (last 3 months: 2023-10-01 to 2023-12-31). Testing occurs on **2024-01-01 onwards** (live operational period). No random shuffling; strict chronological ordering prevents future information leakage.

**Multi-region training:**  
The model is **pooled across all regions** (Thermaikos, Piraeus, Limassol). A single Random Forest model (`final_rf_chl_model_2015_2023.pkl`) is trained on combined data from all three regions. Spatial separation is implicit via region-specific lat/lon groups during feature engineering; no explicit region identifiers are used as predictors. This prevents mixing future data across regions because features are computed per grid cell using only past observations within each (latitude, longitude) group.

## Backtesting Design

**Operational rolling forecast:**  
The system generates **daily rolling-origin forecasts** rather than a single held-out evaluation. Each day, the model predicts a **31-day trajectory** (days 0–30) using the most recent available input features. Backtesting is implicit: predictions for dates in 2024-2026 form a retrospective test set, allowing continuous evaluation against observed outcomes as they become available.

## Leakage Controls

**Strict temporal feature construction:**  
All features use **only information available at forecast issuance time**. Lag features (`chl_t-1`, `nh4_t-1`, etc.) reference the prior day's measurements. Rolling averages (`chl_7day_avg`, `chl_monthly_median`) and anomalies aggregate observations from the preceding 7 or 30 days. No feature accesses data from the target prediction date or beyond.

**Target is future, inputs are past:**  
The target variable (predicted chlorophyll) represents **future days** (today through +30). Input features are constructed from observations up to **yesterday** (t-1) or earlier, ensuring a causal prediction structure with no data leakage from future to past.

## Missing-Data Policy

**What counts as missing:**
- **NaNs in raw variables** (CHL, NH4, NO3, PO4, THETAO, SO) after Copernicus download
- **Masked coastal pixels** where grid cells return no measurement values (empty DataFrames)
- **Absent day files** when daily downloads fail or produce zero rows after processing

**Imputation rule:**
- If strict `dropna()` on required features removes all rows, **median imputation** is applied using historical values from the 30-day trailing window for that region/grid cell.
- If no historical data exists, features are **zero-filled** as a fallback to enable model inference (though prediction quality degrades).

**Data quality flags:**
- A **"data quality" flag** (`.no_rows` marker file) is written when a day's download returns zero valid measurements but the system can still generate a forecast via imputation or fallback to recent data (up to 7 days prior).
- A **"no forecast" day** occurs when no model-ready input exists within the 7-day lookback window; the forecast script skips that date and logs a warning, omitting it from `forecast_log_*.csv`.
