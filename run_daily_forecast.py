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
    # Read model-ready CSVs and write forecast logs to the CODE folder
    csv_input = fr"C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE\model_ready_input_{region}_{target_date_str}.csv"
    output_path = fr"C:\Users\annam\OneDrive - University of Nicosia\Desktop\DASHBOARD CODE\forecast_log_{region}.csv"

    if not os.path.exists(csv_input):
        print(f"⚠️ Skipping {region} — no input file found for {target_date_str}")
        continue

    df = pd.read_csv(csv_input)

    # === Check all features
    missing = set(model.feature_names_in_) - set(df.columns)
    if missing:
        print(f"⚠️ Skipping {region} — missing required features: {missing}")
        continue

    df_input = df[model.feature_names_in_]

    # === Skip region if no valid rows
    if df_input.empty:
        print(f"⚠️ Skipping {region} — no valid rows for prediction on {target_date_str}")
        continue

    # === Predict CHL
    predicted_chl = model.predict(df_input)
    predicted_chl_mean = np.mean(predicted_chl)

    # === Dynamic threshold (per region/day)
    threshold = np.percentile(predicted_chl, 90)
    # Continuous risk: percent of grid points above threshold
    risk_pct = float(np.mean(predicted_chl >= threshold)) * 100
    risk_flag = int(predicted_chl_mean >= threshold)

    # === Save forecast row (add risk_pct)
    result = pd.DataFrame([{
        "date": target_date_str,
        "predicted_chl": round(predicted_chl_mean, 3),
        "bloom_risk_flag": risk_flag,
        "threshold_used": round(threshold, 3),
        "risk_pct": round(risk_pct, 1),
        "num_grid_points": len(predicted_chl)
    }])

    if os.path.exists(output_path):
        result.to_csv(output_path, mode='a', index=False, header=False)
    else:
        result.to_csv(output_path, index=False)

    print(f"✅ Forecast complete for {region}:")
    print(result)
