import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import pickle
import argparse

# Add scripts directory to path for execution_logger
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from execution_logger import log_forecast_start, log_forecast_success, log_forecast_error
    LOGGING_ENABLED = True
except Exception as e:
    LOGGING_ENABLED = False
    # Fallback no-ops if logger unavailable
    def log_forecast_start(*args, **kwargs): pass
    def log_forecast_success(*args, **kwargs): pass
    def log_forecast_error(*args, **kwargs): pass

# === Parse command-line arguments
parser = argparse.ArgumentParser(description='Run daily forecast for a specific date')
parser.add_argument('--date', help='Target date YYYY-MM-DD (defaults to yesterday)', default=None)
args = parser.parse_args()

# === Dynamic date (matches model_ready_input file name)
if args.date:
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
        yesterday = target_date
    except ValueError:
        raise ValueError('Invalid --date format, expected YYYY-MM-DD')
else:
    yesterday = datetime.today() - pd.Timedelta(days=1)

target_date_str = yesterday.strftime("%Y-%m-%d")

# === DATA layout
DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
MODEL_DIR = os.path.join(DATA_DIR, "models")
MODEL_READY_DIR = os.path.join(DATA_DIR, "model_ready")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
for d in (DATA_DIR, MODEL_DIR, MODEL_READY_DIR, FORECAST_DIR):
    os.makedirs(d, exist_ok=True)

# prefer model in data/models; fallback to root filename
model_path = os.path.join(MODEL_DIR, "final_rf_chl_model_2015_2023.pkl")
if not os.path.exists(model_path):
    if os.path.exists("new_updated_model.pkl"):
        model_path = "new_updated_model.pkl"

# Check model file size
if os.path.exists(model_path):
    print(f"Model file '{model_path}' size: {os.path.getsize(model_path)} bytes")
else:
    print(f"Model file '{model_path}' not found!")

# Try loading with joblib, fallback to pickle for diagnostics
try:
    import joblib
    model = joblib.load(model_path)
    print("Model loaded successfully with joblib.")
except Exception as e:
    print(f"Joblib failed to load model: {e}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully with pickle.")
    except Exception as e2:
        print(f"Pickle failed to load model: {e2}")
        raise

# === Regions (can be overridden with MARS_REGIONS env var, comma-separated)
env_regions = os.environ.get("MARS_REGIONS", None)
if env_regions:
    REGIONS = [r.strip() for r in env_regions.split(',') if r.strip()]
else:
    REGIONS = ["thermaikos", "peiraeus", "limassol"]

# === Features used by the model
features = [
    'nh4','no3','po4','so','thetao',
    'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
    'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
    'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
    'chl_monthly_median','chl_anomaly','bloom_proxy_label'
]

# Log forecast start
log_forecast_start(target_date_str, REGIONS)

# === Loop through each region
for region in REGIONS:
    # Forecast for today and next 30 days (31 forecasts total)
    output_path = os.path.join(FORECAST_DIR, f"forecast_log_{region}.csv")
    forecast_dates = [yesterday + pd.Timedelta(days=i) for i in range(0, 31)]

    all_results = []
    for forecast_date in forecast_dates:
        forecast_date_str = forecast_date.strftime("%Y-%m-%d")
        csv_input = os.path.join(MODEL_READY_DIR, f"model_ready_input_{region}_{forecast_date_str}.csv")
        used_input_date = None
        df_input = pd.DataFrame()
        # Try the exact-date input first
        if os.path.exists(csv_input):
            try:
                df_input = pd.read_csv(csv_input)
                if df_input.shape[0] > 0:
                    used_input_date = forecast_date_str
            except Exception:
                df_input = pd.DataFrame()

        # If exact-date input missing or empty, search backwards up to 7 days for most recent non-empty input
        if df_input.shape[0] == 0:
            lookback_days = 7
            for i in range(1, lookback_days + 1):
                alt_date = (forecast_date - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                alt_path = os.path.join(MODEL_READY_DIR, f"model_ready_input_{region}_{alt_date}.csv")
                if os.path.exists(alt_path):
                    try:
                        alt_df = pd.read_csv(alt_path)
                        if alt_df.shape[0] > 0:
                            df_input = alt_df.copy()
                            used_input_date = alt_date
                            print(f"Using fallback model input for {region}: {alt_path} (for target date {forecast_date_str})")
                            break
                    except Exception:
                        continue

        if df_input.shape[0] == 0:
            if used_input_date is None:
                print(f"No model input found for {region} on {forecast_date_str} and no fallback available. Skipping prediction.")
            else:
                print(f"Fallback input for {region} on {forecast_date_str} was empty. Skipping prediction.")
            continue
        predicted_chl = model.predict(df_input)
        predicted_chl_mean = np.mean(predicted_chl)
        threshold = np.percentile(predicted_chl, 90)
        risk_pct = float(np.mean(predicted_chl >= threshold)) * 100
        risk_flag = int(predicted_chl_mean >= threshold)

        # Threshold-relative risk (main risk score)
        if threshold > 0:
            risk_score = min(100, max(0, (predicted_chl_mean / threshold) * 100))
        else:
            risk_score = 0

        result = {
            "date": forecast_date_str,
            "predicted_chl": round(predicted_chl_mean, 3),
            "bloom_risk_flag": risk_flag,
            "threshold_used": round(threshold, 3),
            "risk_pct": round(risk_pct, 1),
            "risk_score": round(risk_score, 1),
            "num_grid_points": len(predicted_chl)
        }
        all_results.append(result)

    if all_results:
        df_results = pd.DataFrame(all_results)
        if os.path.exists(output_path):
            df_results.to_csv(output_path, mode='a', index=False, header=False)
        else:
            df_results.to_csv(output_path, index=False)
        print(f"✅ Forecast complete for {region}:")
        print(df_results)
        
        # Log success
        log_forecast_success(target_date_str, region, len(all_results))
    else:
        print(f"⚠️  No forecasts generated for {region}")
        log_forecast_error(target_date_str, region, "No forecasts generated")
