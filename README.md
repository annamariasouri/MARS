# MARS — Marine Autonomous Risk System

Daily red-tide risk forecasts for the Eastern Mediterranean (Thermaikos, Piraeus, Limassol), using Copernicus Marine data and a retrained Random Forest model.

## Dashboard (primary UI)

The **MARSdesign** prototype lives in `MARSdesign/`; the integrated UI used by Streamlit is in `web/`. **Streamlit** hosts it online and locally (same look, live `data/` CSVs).

### Run locally

```powershell
pip install -r requirements.txt
.\dev.ps1
```

Opens **http://localhost:8765/** — the new design with your project data.

### Put the app online (Streamlit Cloud) — recommended

1. Push this repo to **GitHub** (include `web/`, `data/`, `streamlit_app.py`).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **Create app**.
3. Pick your repo, branch `main`, main file: **`streamlit_app.py`**.
4. Deploy. Your public URL will look like: `https://your-app-name.streamlit.app`

That URL shows the **new design**, not the old Streamlit layout and not a folder list. Data comes from the CSVs in your repo (updated daily by GitHub Actions).

GitHub Pages is optional (workflow **Deploy MARS Dashboard**, manual run only). **Streamlit Cloud** is the main way to go online.

## How predictions are made

- **Data ingestion:** Daily Copernicus download (nutrients, temperature, salinity, chlorophyll) per region.
- **Feature engineering:** Lags, rolling means, ratios, anomalies → model-ready CSVs.
- **Prediction:** `rf_chl_retrained.pkl` predicts daily chlorophyll per grid cell.
- **Risk scoring:** Adaptive threshold (90th percentile of predicted CHL); bloom flag and risk score when above threshold.
- **Outputs:** `data/forecasts/forecast_log_*.csv`, env history, evaluation CSVs → exported to `web/data.js` for the dashboard.

## Key paths

| Path | Role |
|------|------|
| `MARSdesign/` | Design prototype (reference screenshots, handoff) |
| `web/` | **Dashboard UI** synced from MARSdesign (`index.html`, `app.jsx`, …) |
| `data/forecasts/` | Daily forecast logs |
| `data/copernicus/env_history/` | Environmental time series |
| `data/evaluation/` | Accuracy & bloom metrics |
| `.github/scripts/export_dashboard_data.py` | CSVs → `web/data.js` |
| `.github/scripts/run_daily_forecast.py` | Daily forecast job |
| `.github/scripts/download_copernicus_day_test.py` | Copernicus ingest |
| `.github/scripts/legacy/dashboard_app_streamlit.py` | Old UI (archived, do not use) |

See **`METHODOLOGY.md`** for model design, splits, and validation.

## Model file

`data/models/rf_chl_retrained.pkl` (~3 GB) is on Google Drive:

```powershell
pip install gdown
python .github/scripts/download_model.py
```

## Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline scripts

```powershell
python .github/scripts/download_copernicus_day_test.py
python .github/scripts/run_daily_forecast.py
python .github/scripts/evaluate_forecasts.py --region thermaikos
python .github/scripts/export_dashboard_data.py
```

Daily automation: `.github/workflows/mars_daily.yml` (updates CSVs + `web/data.js`).

## Troubleshooting

- **Blank dashboard:** Run `python .github/scripts/export_dashboard_data.py`, then `.\dev.ps1`.
- **Missing forecasts:** Check `data/forecasts/forecast_log_{region}.csv` exists.
- **Copernicus errors:** Set `COPERNICUS_USERNAME` / `COPERNICUS_PASSWORD` secrets for CI.
