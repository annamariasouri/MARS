import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

# === Dynamic date (matches model_ready_input file name)
yesterday = datetime.today() - pd.Timedelta(days=1)
target_date_str = yesterday.strftime("%Y-%m-%d")

# === Model path (use CODE folder where model file exists)
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

# === Regions
REGIONS = ["thermaikos", "peiraeus", "limassol"]

# === Features used by the model
features = [
    'nh4','no3','po4','so','thetao',
    'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
    'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
    'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
    'chl_monthly_median','chl_anomaly','bloom_proxy_label'
]

# === Loop through each region
for region in REGIONS:
    # Forecast for today, next 7 days, and next month
    output_path = os.path.join(os.getcwd(), f"forecast_log_{region}.csv")
    forecast_dates = [yesterday + pd.Timedelta(days=i) for i in range(0, 8)]
    forecast_dates.append(yesterday + pd.Timedelta(days=30))

    all_results = []
    for forecast_date in forecast_dates:
        forecast_date_str = forecast_date.strftime("%Y-%m-%d")
        csv_input = os.path.join(os.getcwd(), f"model_ready_input_{region}_{forecast_date_str}.csv")
        if not os.path.exists(csv_input):
            print(f"Input file not found for {region} on {forecast_date_str}: {csv_input}")
            continue
        df_input = pd.read_csv(csv_input)
        if df_input.shape[0] == 0:
            print(f"Input file for {region} on {forecast_date_str} is empty. Skipping prediction.")
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
