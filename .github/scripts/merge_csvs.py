import pandas as pd
import glob
import os
import numpy as np

# Merge all forecast logs for each region
regions = ["thermaikos", "peiraeus", "limassol"]

## Only merge environmental history files for each region
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
