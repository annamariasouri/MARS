import pandas as pd
import os

# Table 3-like summary from retrain/validation_metrics_retrained.csv
print("\nTable 3-like summary from retrain/validation_metrics_retrained.csv:")
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data", "retrain", "validation_metrics_retrained.csv")
summary = pd.read_csv(csv_path)
summary = summary.rename(columns={"region": "Region", "n": "N days", "rmse": "RMSE", "r2": "R2"})
cols = ["Region", "N days", "RMSE", "R2"]
print(summary[cols].to_csv(index=False))
results = {}
for region in regions:
    df = pd.read_csv(f'data/evaluation/accuracy_{region}.csv')
    df = df.dropna(subset=['predicted_chl', 'observed_chl'])
    n_days = len(df)
    mae = mean_absolute_error(df['observed_chl'], df['predicted_chl'])
    rmse = mean_squared_error(df['observed_chl'], df['predicted_chl']) ** 0.5
    r2 = r2_score(df['observed_chl'], df['predicted_chl'])
    results[region] = {'n_days': n_days, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Optionally, compute overall metrics by concatenating all regions
df_all = pd.concat([
    pd.read_csv(f'data/evaluation/accuracy_{region}.csv').dropna(subset=['predicted_chl', 'observed_chl'])
    for region in regions
])
n_days = len(df_all)
mae = mean_absolute_error(df_all['observed_chl'], df_all['predicted_chl'])
rmse = mean_squared_error(df_all['observed_chl'], df_all['predicted_chl']) ** 0.5
r2 = r2_score(df_all['observed_chl'], df_all['predicted_chl'])
results['overall'] = {'n_days': n_days, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

print(results)

# Table 3-like summary from retrain/validation_metrics_retrained.csv
print("\nTable 3-like summary from retrain/validation_metrics_retrained.csv:")
summary = pd.read_csv("data/retrain/validation_metrics_retrained.csv")
summary = summary.rename(columns={"region": "Region", "n": "N days", "rmse": "RMSE", "r2": "R2"})
cols = ["Region", "N days", "RMSE", "R2"]
print(summary[cols].to_csv(index=False))