import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURE_COLS = [
    "nh4","no3","po4","so","thetao",
    "chl_t-1","nh4_t-1","no3_t-1","po4_t-1","so_t-1","thetao_t-1",
    "chl_7day_avg","nh4_7day_avg","no3_7day_avg","po4_7day_avg","so_7day_avg","thetao_7day_avg",
    "n_p_ratio","n_nh4_ratio","p_nh4_ratio",
    "chl_monthly_median","chl_anomaly","bloom_proxy_label",
]

model = joblib.load("data/models/rf_chl_retrained.pkl")

val_path = "data/retrain/val_pooled.csv"
if not os.path.exists(val_path):
    print("val_pooled.csv not found – run retrain_rf_model.py first.")
else:
    val = pd.read_csv(val_path)
    print(f"Loaded val_pooled.csv  ({len(val):,} rows)\n")
    header = f"{'Region':<15} {'N':>8}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}"
    print(header)
    print("-" * len(header))
    for region in ["thermaikos", "peiraeus", "limassol", "OVERALL"]:
        sel = val if region == "OVERALL" else val[val["region"] == region]
        sel = sel.dropna(subset=FEATURE_COLS + ["chl"])
        if sel.empty:
            print(f"{region:<15} {'0':>8}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
            continue
        y_pred = model.predict(sel[FEATURE_COLS].values)
        y_true = sel["chl"].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"{region:<15} {len(sel):>8,}  {rmse:>8.4f}  {mae:>8.4f}  {r2:>8.4f}")
