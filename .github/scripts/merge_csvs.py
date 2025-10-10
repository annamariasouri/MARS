import pandas as pd
import glob
import os

# Merge all forecast logs for each region
regions = ["thermaikos", "peiraeus", "limassol"]
for region in regions:
    pattern = f"forecast_log_{region}.csv"
    files = glob.glob(pattern)
    if not files:
        continue
    # Read and concatenate, handle bad lines, and mixed columns
    dfs = []
    all_columns = set()
    for f in files:
        try:
            df = pd.read_csv(f, on_bad_lines='warn')
            all_columns.update(df.columns)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
    if dfs:
        # Remove old forecast log before writing merged result
        out_path = f"forecast_log_{region}.csv"
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception as e:
            print(f"⚠️ Could not delete old {out_path}: {e}")
        # Reindex all DataFrames to have the union of all columns
        all_columns = list(all_columns)
        dfs = [d.reindex(columns=all_columns) for d in dfs]
        merged = pd.concat(dfs, ignore_index=True)
        # Drop duplicates by date, keep the last (latest) entry for each date
        if 'date' in merged.columns:
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
