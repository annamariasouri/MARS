"""Move existing artifact files into the DATA_DIR subfolders.

This script is idempotent and will create the following layout under DATA_DIR:
 - env_history/        -> env_history_*.csv and env_history_{region}.csv
 - model_ready/        -> model_ready_input_*.csv
 - forecasts/          -> forecast_log_*.csv
 - models/             -> final model .pkl files
 - raw_nc/             -> any .nc files
 - download_reports/   -> small markers and reports

Run: python scripts/move_existing_to_data.py
"""
import os
import shutil
import glob
from pathlib import Path

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
ENV_DIR = os.path.join(DATA_DIR, "env_history")
MODEL_READY_DIR = os.path.join(DATA_DIR, "model_ready")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
MODELS_DIR = os.path.join(DATA_DIR, "models")
RAW_DIR = os.path.join(DATA_DIR, "raw_nc")
REPORT_DIR = os.path.join(DATA_DIR, "download_reports")
for d in (DATA_DIR, ENV_DIR, MODEL_READY_DIR, FORECAST_DIR, MODELS_DIR, RAW_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

root = os.getcwd()
print("Moving files from:", root, "to DATA_DIR:", DATA_DIR)

moved = []
# Patterns and destinations
patterns = [
    ("env_history_*.csv", ENV_DIR),
    ("model_ready_input_*.csv", MODEL_READY_DIR),
    ("forecast_log_*.csv", FORECAST_DIR),
    ("final_rf_*.pkl", MODELS_DIR),
    ("new_updated_model.pkl", MODELS_DIR),
    ("*.nc", RAW_DIR),
    ("env_history_*.csv.bak.*", ENV_DIR),
    ("download_reports/*", REPORT_DIR),
]

for pattern, dest in patterns:
    for p in glob.glob(os.path.join(root, pattern)):
        try:
            fname = os.path.basename(p)
            target = os.path.join(dest, fname)
            # if target exists, append a suffix to avoid overwrite
            if os.path.exists(target):
                i = 1
                base, ext = os.path.splitext(fname)
                while os.path.exists(os.path.join(dest, f"{base}.dup{i}{ext}")):
                    i += 1
                target = os.path.join(dest, f"{base}.dup{i}{ext}")
            shutil.move(p, target)
            moved.append((p, target))
            print(f"Moved: {p} -> {target}")
        except Exception as e:
            print(f"Failed to move {p}: {e}")

# Report summary
print("\nSummary: moved", len(moved), "files")
for s, t in moved:
    print(s, "->", t)

print('\nDone.')
