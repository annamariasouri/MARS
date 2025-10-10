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
    # Read model-ready CSVs and write forecast logs to the workspace root
    csv_input = os.path.join(os.getcwd(), f"model_ready_input_{region}_{target_date_str}.csv")
    output_path = os.path.join(os.getcwd(), f"forecast_log_{region}.csv")

    # Default placeholder row (all NaNs)
    placeholder = pd.DataFrame([{
        "date": target_date_str,
        "predicted_chl": np.nan,
        "bloom_risk_flag": np.nan,
        "threshold_used": np.nan,
        "risk_pct": np.nan,
        "risk_score": np.nan,
        "num_grid_points": 0
    }])

    if not os.path.exists(csv_input):
        print(f"⚠️ No input file for {region} on {target_date_str}. Writing placeholder row.")
        placeholder.to_csv(output_path, mode='a' if os.path.exists(output_path) else 'w', index=False, header=not os.path.exists(output_path))
        continue

    df = pd.read_csv(csv_input)

    missing = set(model.feature_names_in_) - set(df.columns)
    if missing:
        print(f"⚠️ Missing required features for {region}: {missing}. Writing placeholder row.")
        placeholder.to_csv(output_path, mode='a' if os.path.exists(output_path) else 'w', index=False, header=not os.path.exists(output_path))
        continue

    df_input = df[model.feature_names_in_]
    if df_input.empty:
        print(f"⚠️ No valid rows for prediction for {region} on {target_date_str}. Writing placeholder row.")
        placeholder.to_csv(output_path, mode='a' if os.path.exists(output_path) else 'w', index=False, header=not os.path.exists(output_path))
        continue

    # Predict CHL
    predicted_chl = model.predict(df_input)
    predicted_chl_mean = np.mean(predicted_chl)
    threshold = np.percentile(predicted_chl, 90)
    risk_pct = float(np.mean(predicted_chl >= threshold)) * 100
    risk_flag = int(predicted_chl_mean >= threshold)
    min_chl = 0.0
    max_chl = 2.5
    risk_score = (predicted_chl_mean - min_chl) / (max_chl - min_chl)
    risk_score_pct = max(0, min(1, risk_score)) * 100

    result = pd.DataFrame([{
        "date": target_date_str,
        "predicted_chl": round(predicted_chl_mean, 3),
        "bloom_risk_flag": risk_flag,
        "threshold_used": round(threshold, 3),
        "risk_pct": round(risk_pct, 1),
        "risk_score": round(risk_score_pct, 1),
        "num_grid_points": len(predicted_chl)
    }])

    if os.path.exists(output_path):
        result.to_csv(output_path, mode='a', index=False, header=False)
    else:
        result.to_csv(output_path, index=False)

    print(f"✅ Forecast complete for {region}:")
    print(result)
