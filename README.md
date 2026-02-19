# MARS Dashboard
# How are predictions made?

- **Data ingestion:** Daily download of Copernicus Marine Service data (nutrients, temperature, salinity, chlorophyll) for the last 30 days per region.
- **Feature engineering:** Computes lagged values, rolling averages, nutrient ratios, and anomalies for each grid cell, creating model-ready input files.
- **Prediction:** The retrained Random Forest model (`rf_chl_retrained.pkl`) predicts daily chlorophyll-a concentration for each region and day.
- **Risk scoring:** Calculates a region-specific threshold (90th percentile of predicted chlorophyll). If the predicted value exceeds this, a bloom risk flag is set. A continuous risk score is also computed.
- **Outputs:** Results are saved as CSVs and visualized in the dashboard, including time series, risk frequencies, and interactive maps.

# What do the predictions mean?

- **Chlorophyll-a (mg/m³):** Proxy for phytoplankton biomass; high values may indicate a harmful algal bloom (HAB).
- **Risk flag:** Indicates if the predicted chlorophyll exceeds the region’s adaptive threshold (potential bloom event).
- **Risk score:** Shows how close the prediction is to the threshold (as a percentage).
- **Rolling risk counts:** Show how often the region has been flagged in the last 7 and 30 days, providing context for trends.

# Model details

- **Model:** Random Forest, retrained on 2012–2022 Copernicus reanalysis data, validated on 2023–2025.
- **Regions:** Thermaikos, Piraeus, Limassol (pooled training).
- **Features:** 24 variables including nutrients, temperature, salinity, lags, rolling means, and anomalies.
- **Performance:** High accuracy (overall R² ≈ 0.94, RMSE ≈ 0.08 mg/m³).

A small Streamlit dashboard and supporting scripts for daily marine forecasts and environmental data processing (Copernicus subsets, feature engineering, and a trained random forest model for chlorophyll prediction).

This repository includes scripts that download Copernicus NetCDF files, process them into CSVs used by the model, run daily forecasts, and a Streamlit app to visualise results.

## Files of interest

- `github/scripts/dashboard_app.py` — Streamlit app (UI).
- `github/scripts/download_copernicus_day_test.py` — downloads Copernicus data, merges NetCDFs, does feature engineering, and saves CSVs.
- `github/scripts/run_daily_forecast.py` — loads a trained model and creates daily forecasts (writes `forecast_log_*.csv`).
- `github/scripts/retrain_rf_model.py` — full pipeline to retrain the Random Forest on 2012–2022 CMEMS reanalysis data.
- `github/scripts/download_model.py` — downloads the pre-trained model (~3 GB) from Google Drive to `data/models/`.
- `*.csv` — example CSV files in the repository (e.g., `env_history_*.csv`, `model_ready_input_*.csv`, `forecast_log_*.csv`).
- **`METHODOLOGY.md`** — detailed documentation of the model, data splits, backtesting design, leakage controls, and missing-data policy.

## Where CSVs are saved (important)

By default the scripts in this repo are configured to write and read CSVs from the project root (the folder containing this `README.md`). That path is:

```
C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE
```

If you want to change where CSVs are stored, update the `output_dir` variable in `download_copernicus_day_test.py` and the `csv_input`/`output_path` variables in `run_daily_forecast.py`. If you'd like, I can centralise that path into a single `config.py` for you.

## Model file

The trained Random Forest (`data/models/rf_chl_retrained.pkl`, ~3 GB) is too large for Git and is hosted on Google Drive. Download it before running any forecast or evaluation script:

```powershell
pip install gdown
python download_model.py
```

This will save the file to `data/models/rf_chl_retrained.pkl` automatically. If the file already exists it is skipped.

## Prerequisites

- Python 3.10+ (a virtual environment is recommended)
- Internet access (for Copernicus downloads and model download)

## Install dependencies

I included a `requirements.txt` with the typical packages used by the scripts. Create a virtual environment and install:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Streamlit app

From the project root (where `dashboard_app.py` is located):

```powershell
# activate env first (see above), then:
streamlit run "C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE\dashboard_app.py"
```

Streamlit will open in your browser and the app will read CSVs from the project folder. Make sure the CSV files such as `env_history_{region}_{date}.csv` and `model_ready_input_{region}_{date}.csv` exist in the folder before running the app.

## Run the downloader and forecast scripts

Downloader (produces env_history and model-ready CSVs):

```powershell
python "C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE\download_copernicus_day_test.py"
```

Forecast runner (reads `model_ready_input_*` files and writes `forecast_log_*`):

```powershell
python "C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE\run_daily_forecast.py"
```

## Troubleshooting

- If Streamlit fails to find CSVs, confirm they exist in the project root and that file permissions are correct.
- If NetCDF downloads fail, ensure `copernicusmarine` (or your chosen client) is authenticated and internet access is available.
- If you get errors importing packages, verify that you installed `requirements.txt` into the same Python interpreter used to run Streamlit.

## Next steps (recommended)

- Create a `config.py` with a single `OUTPUT_DIR` constant and update scripts to import it. I can add this for you.
- Add a GitHub Actions workflow to run a basic linter/test pipeline.

---

If you want the `config.py` and a consolidated change across scripts (so the CSV path is defined in one place), tell me and I'll implement it.