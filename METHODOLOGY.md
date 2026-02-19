# Methodology

_Last updated: February 2026_

## System Overview

**Operational pipeline:**  
The system executes three stages daily: (1) **Data ingestion** downloads Copernicus Marine near-real-time products (nutrients: `cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m`; temperature: `cmems_mod_med_phy-tem_anfc_4.2km_P1D-m`; salinity: `cmems_mod_med_phy-sal_anfc_4.2km_P1D-m`; chlorophyll: `cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m`) at 1–10 m depth for the past 30 days. (2) **Feature engineering** computes lag-1, 7-day/30-day rolling averages, nutrient ratios, and anomalies per grid cell, writing model-ready inputs. (3) **Forecasting** loads the retrained Random Forest (`rf_chl_retrained.pkl`), predicts chlorophyll for **days 0–30 (31 daily forecasts)**, and writes results to `forecast_log_*.csv` for dashboard visualization.

**Failure points and mitigations:**  
API timeouts trigger exponential-backoff retries (up to 8 attempts). Masked coastal pixels are handled via adaptive bounding-box expansion (±0.0°, ±0.05°, ±0.1°, ±0.25°, ±0.5°) until valid measurements are found; the successful expansion level is recorded in `report_{region}_{date}.txt` (field: `expansion_used`). Missing days fall back to the most recent valid input within a 7-day lookback window; if unavailable, median imputation uses the trailing 30-day window. Days without fallback data skip forecasting and log `.no_rows` markers.

**Model specification (current — as of February 2026):**  
The active predictor is `data/models/rf_chl_retrained.pkl`, a scikit-learn `RandomForestRegressor` retrained via `retrain_rf_model.py`. Key details:

| Property | Value |
|---|---|
| Algorithm | scikit-learn RandomForestRegressor |
| Training period | 2012-01-01 – 2022-12-31 |
| Validation period | 2023-01-01 – 2025-12-31 |
| Training source | CMEMS multi-year reanalysis (`cmems_mod_med_bgc-nut_my_4.2km_P1D-m`, `cmems_mod_med_phy-temp_my_4.2km_P1D-m`, `cmems_mod_med_phy-sal_my_4.2km_P1D-m`, `cmems_mod_med_bgc-plankton_my_4.2km_P1D-m`) |
| Depth range | 1.02 – 10.0 m |
| Regions | Thermaikos, Piraeus, Limassol (pooled) |
| Number of features | 24 |
| Validation R² (overall) | **0.9449** (160,083 samples) |
| Validation RMSE (overall) | 0.0794 mg/m³ |

Per-region validation performance:

| Region | n | RMSE | R² |
|---|---|---|---|
| Thermaikos | 60,984 | 0.1283 | 0.9185 |
| Piraeus | 90,387 | 0.0070 | 0.9742 |
| Limassol | 8,712 | 0.0051 | 0.9085 |

**24 input features:**  
Current-day: `nh4`, `no3`, `po4`, `so`, `thetao`  
Lag-1 (t−1): `chl_t-1`, `nh4_t-1`, `no3_t-1`, `po4_t-1`, `so_t-1`, `thetao_t-1`  
7-day rolling means: `chl_7day_avg`, `nh4_7day_avg`, `no3_7day_avg`, `po4_7day_avg`, `so_7day_avg`, `thetao_7day_avg`  
Nutrient ratios: `n_p_ratio`, `n_nh4_ratio`, `p_nh4_ratio`  
Chlorophyll history: `chl_monthly_median`, `chl_anomaly`, `bloom_proxy_label`

**Model file distribution:**  
The model file (`rf_chl_retrained.pkl`) exceeds GitHub's file-size limits and is **not stored in the repository**. It must be obtained separately (see README for options) and placed at `data/models/rf_chl_retrained.pkl` before running any forecast or evaluation script.

**Alert generation:**  
Predictions are aggregated as the spatial mean across all grid cells per region. The alert threshold is the **90th percentile** of the predicted chlorophyll distribution for that region-day. A binary risk flag activates when `predicted_chl ≥ threshold`, and a continuous risk score is computed as `min(100, (predicted_chl / threshold) × 100)`. No temporal smoothing or hysteresis is applied; each day's alert is independent. To contextualize alerts, the dashboard displays 7-day and 30-day risk frequencies (count of flagged days in those windows).

**Bloom thresholds (binary detection):**  
Thermaikos: 1.0 mg/m³ · Piraeus: 0.5 mg/m³ · Limassol: 0.3 mg/m³

**Alert-fatigue mitigation:**  
Percentile-based thresholds adapt to local chlorophyll distributions, reducing false positives in naturally high-chlorophyll regions. Rolling-window risk counts (7/30 days) provide trend context rather than isolated binary flags. Future work may add 3-day moving-average smoothing or require 2 consecutive flagged days before escalating to "high alert."

**Artifacts produced:**  
Daily runs generate: `env_history_{region}_{date}.csv` (raw Copernicus extracts), `model_ready_input_{region}_{date}.csv` (engineered features), `forecast_log_{region}.csv` (cumulative predictions), and `report_{region}_{date}.txt` (download metadata including bbox expansion level). A Streamlit dashboard (`dashboard_app.py`) reads these files to display time-series plots, KPIs (latest chlorophyll, 7/30-day risk counts), and interactive maps.

**Novel contributions:**  
Unlike batch ML workflows, this is an **operational forecasting system** with daily automated data ingestion and real-time updates. Adaptive bbox expansion addresses coastal masking in high-resolution ocean models—a common failure mode ignored by generic pipelines. Multi-region pooled training improves generalization across heterogeneous Mediterranean ecosystems (open gulf vs. enclosed port). The retrained model extends the training window back to 2012 (vs. 2015 previously), adding an extra decade of inter-annual variability. Ground truth for evaluation is the same Copernicus product family, enabling retrospective validation once target-date observations become available.

## Data Splits

**Time-based train/validation/test split:**  
Training data spans **2012-01-01 to 2022-12-31** (11 years, CMEMS multi-year reanalysis). Validation covers **2023-01-01 to 2025-12-31** (held-out, also from the multi-year reanalysis product which extends to ~January 2026). Operational testing occurs on **live near-real-time data from 2024 onwards**. No random shuffling; strict chronological ordering prevents future information leakage.

**Multi-region training:**  
A single Random Forest (`rf_chl_retrained.pkl`) is trained on pooled data from all three regions (Thermaikos, Piraeus, Limassol). Spatial separation is implicit: features are computed per `(latitude, longitude)` group using only that cell's historical observations, so no cross-region leakage occurs. No explicit region identifier is used as a predictor.

## Backtesting Design

**Operational rolling forecast:**  
The system generates **daily rolling-origin forecasts** rather than a single held-out evaluation. Each day, the model predicts a **31-day trajectory** (days 0–30) using the most recent available input features. Backtesting is implicit: predictions for dates in 2024–2026 form a retrospective test set, allowing continuous evaluation against observed outcomes as they become available.

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
