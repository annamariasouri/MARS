import os
import pandas as pd
from glob import glob

# Paths
EVAL_DIR = os.path.join("data", "evaluation")
ENV_HISTORY_DIR = os.path.join("data", "copernicus", "env_history")

# Regions
regions = ["thermaikos", "peiraeus", "limassol"]

for region in regions:
    acc_path = os.path.join(EVAL_DIR, f"accuracy_{region}.csv")
    if not os.path.exists(acc_path):
        print(f"No accuracy file for {region}")
        continue
    acc_df = pd.read_csv(acc_path)
    # Only update rows with missing observed_chl
    for idx, row in acc_df[acc_df['observed_chl'].isna()].iterrows():
        date = str(row['target_date'])[:10]  # YYYY-MM-DD
        # Find env_history file for this region and date
        env_file = os.path.join(ENV_HISTORY_DIR, f"env_history_{region}_{date}.csv")
        if os.path.exists(env_file):
            env_df = pd.read_csv(env_file)
            # Try to get mean or first value of CHL column
            chl_cols = [c for c in env_df.columns if 'chl' in c.lower()]
            if chl_cols:
                observed_val = env_df[chl_cols[0]].mean()
                acc_df.at[idx, 'observed_chl'] = observed_val
                print(f"Filled {region} {date} with {observed_val}")
            else:
                print(f"No CHL column in {env_file}")
        else:
            print(f"No env_history file for {region} {date}")
    # Save back
    acc_df.to_csv(acc_path, index=False)
print("Done.")
