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
    # Read and concatenate, drop duplicates
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs).drop_duplicates()
    merged.to_csv(f"forecast_log_{region}.csv", index=False)

# Merge all environmental history files for each region
for region in regions:
    pattern = f"env_history_{region}_*.csv"
    files = glob.glob(pattern)
    if not files:
        continue
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs).drop_duplicates()
    merged.to_csv(f"env_history_{region}.csv", index=False)
