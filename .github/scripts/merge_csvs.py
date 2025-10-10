import pandas as pd
import glob
import os
import numpy as np

# Merge all forecast logs for each region
regions = ["thermaikos", "peiraeus", "limassol"]
for region in regions:
    pattern = f"forecast_log_{region}.csv"
    files = glob.glob(pattern)
    if not files:
        continue
    # Always expect 7 columns for forecast logs
    expected_columns = [
        "date",
        "predicted_chl",
        "bloom_risk_flag",
        "threshold_used",
        "risk_pct",
        "risk_score",
        "num_grid_points"
    ]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='warn')
            # Fill missing columns with NaN
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = np.nan
            # Only keep expected columns
            df = df[expected_columns]
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
    if dfs:
        out_path = f"forecast_log_{region}.csv"
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception as e:
            print(f"⚠️ Could not delete old {out_path}: {e}")
        merged = pd.concat(dfs, ignore_index=True)
        # Drop duplicates by date, keep the last (latest) entry for each date
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        merged = merged.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        merged = merged.sort_values('date').reset_index(drop=True)
        merged.to_csv(out_path, index=False)

# Merge all environmental history files for each region
for region in regions:
    pattern = f"env_history_{region}_*.csv"
    files = glob.glob(pattern)
    if not files:
        continue
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='warn')
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
    if dfs:
        merged = pd.concat(dfs).drop_duplicates()
        merged.to_csv(f"env_history_{region}.csv", index=False)
