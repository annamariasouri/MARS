"""Evaluate forecasts against observations and write per-row evaluation to data/eval/accuracy_{region}.csv

Usage:
  python scripts/evaluate_forecasts.py --region limassol [--start YYYY-MM-DD] [--end YYYY-MM-DD]

This script is idempotent: it will not re-write existing evaluated rows (keyed by forecast_date+target_date+lat+lon).

Matching strategy (first pass): round lat/lon to 3 decimals and join on target_date + lat_r + lon_r.
If forecast rows lack spatial coords, evaluation is done at region-average level.
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
ROOT = os.environ.get("MARS_DATA_DIR", "data")
EVAL_DIR = os.path.join(ROOT, "eval")
ENV_DIR = os.path.join(ROOT, "env_history")
FORECAST_DIR = os.path.join(ROOT, "forecasts")

os.makedirs(EVAL_DIR, exist_ok=True)


def find_forecast_file(region: str):
    candidates = [
        os.path.join(FORECAST_DIR, f"forecast_log_{region}.csv"),
        os.path.join(FORECAST_DIR, f"forecast_{region}.csv"),
        os.path.join(ROOT, f"forecast_log_{region}.csv"),
        os.path.join(ROOT, f"forecast_{region}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_env_histories(region: str, start: str | None, end: str | None) -> pd.DataFrame:
    pattern = os.path.join(ENV_DIR, f"env_history_{region}_*.csv")
    files = glob.glob(pattern)
    # also allow undated file
    undated = os.path.join(ENV_DIR, f"env_history_{region}.csv")
    if os.path.exists(undated):
        files.append(undated)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in sorted(files):
        try:
            d = pd.read_csv(f)
            d_cols = {c.lower(): c for c in d.columns}
            # normalise column names -> lowercase keys
            d = d.rename(columns={v: k for k, v in d_cols.items()})
            # prefer 'time' or 'date' for timestamp
            if 'time' in d.columns:
                d['time'] = pd.to_datetime(d['time'], errors='coerce')
            elif 'date' in d.columns:
                d['time'] = pd.to_datetime(d['date'], errors='coerce')
            else:
                # try first datetime-like
                for c in d.columns:
                    parsed = pd.to_datetime(d[c], errors='coerce')
                    if parsed.notna().sum() > 0:
                        d['time'] = parsed
                        break
            # standardise column names
            if 'latitude' in d.columns:
                d = d.rename(columns={'latitude': 'lat'})
            if 'longitude' in d.columns:
                d = d.rename(columns={'longitude': 'lon'})
            for chl_name in ('chl', 'chlorophyll', 'chl_mg_m3'):
                if chl_name in d.columns:
                    d = d.rename(columns={chl_name: 'observed_chl'})
                    break
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.dropna(subset=['time'])
    out['target_date'] = pd.to_datetime(out['time']).dt.normalize()
    if start:
        out = out[out['target_date'] >= pd.to_datetime(start)]
    if end:
        out = out[out['target_date'] <= pd.to_datetime(end)]
    # keep only relevant columns
    keep = [c for c in ['target_date', 'lat', 'lon', 'observed_chl'] if c in out.columns]
    return out[keep].copy()


def evaluate(region: str, start: str | None, end: str | None, tolerance: float = 0.2):
    fpath = find_forecast_file(region)
    if not fpath:
        print(f"No forecast file found for region '{region}'. Looked in {FORECAST_DIR} and {ROOT}.")
        return 1
    print(f"Loading forecast: {fpath}")
    f = pd.read_csv(fpath)
    # normalise lower-case cols
    f.columns = [c.strip().lower() for c in f.columns]
    # try to find target_date column
    date_col = None
    for c in ('target_date', 'date', 'day', 'ds', 'timestamp'):
        if c in f.columns:
            date_col = c
            break
    if date_col is None:
        print("No date column found in forecast file. Aborting.")
        return 1
    f['target_date'] = pd.to_datetime(f[date_col], errors='coerce').dt.normalize()
    obs = load_env_histories(region, start, end)
    if obs.empty:
        print(f"No env_history observations found for region '{region}'. Create env_history files first.")
        return 1
    # prepare forecasts: check if lat/lon present
    has_spatial = 'lat' in f.columns and 'lon' in f.columns
    if has_spatial:
        # round coords to 3 decimals for matching
        f['lat_r'] = f['lat'].round(3)
        f['lon_r'] = f['lon'].round(3)
        obs['lat_r'] = obs['lat'].round(3)
        obs['lon_r'] = obs['lon'].round(3)
        merged = pd.merge(f, obs, left_on=['target_date', 'lat_r', 'lon_r'], right_on=['target_date', 'lat_r', 'lon_r'], how='left')
    else:
        # region-average mode
        daily_obs = obs.groupby('target_date', as_index=False)['observed_chl'].mean()
        merged = pd.merge(f, daily_obs, on='target_date', how='left')
    # compute evaluation columns
    merged['predicted_chl'] = merged['predicted_chl'] if 'predicted_chl' in merged.columns else (merged['chl_pred'] if 'chl_pred' in merged.columns else np.nan)
    merged['observed_chl'] = merged['observed_chl'] if 'observed_chl' in merged.columns else np.nan
    merged['err'] = merged['predicted_chl'] - merged['observed_chl']
    merged['abs_err'] = merged['err'].abs()
    # bloom flags (if threshold available)
    if 'threshold_used' in merged.columns:
        merged['bloom_pred'] = merged['predicted_chl'] >= merged['threshold_used']
        merged['bloom_obs'] = merged['observed_chl'] >= merged['threshold_used']
    else:
        # fallback: use tolerance-based correctness
        merged['bloom_pred'] = False
        merged['bloom_obs'] = merged['abs_err'] <= tolerance
    # add key for idempotency
    def make_key(row):
        lat = row.get('lat', None)
        lon = row.get('lon', None)
        return f"{row.get('forecast_date', '')}|{row.get('target_date')}|{lat}|{lon}"
    merged['eval_key'] = merged.apply(lambda r: make_key(r), axis=1)
    out_path = os.path.join(EVAL_DIR, f"accuracy_{region}.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, dtype=str)
        existing_keys = set(existing.get('eval_key', []))
    else:
        existing_keys = set()
    # select new rows
    new = merged[~merged['eval_key'].isin(existing_keys)].copy()
    if new.empty:
        print("No new rows to evaluate.")
        return 0
    # keep canonical columns
    keep_cols = ['forecast_date', 'target_date', 'lat', 'lon', 'predicted_chl', 'observed_chl', 'err', 'abs_err', 'bloom_pred', 'bloom_obs', 'eval_key']
    for c in keep_cols:
        if c not in new.columns:
            new[c] = np.nan
    new = new[keep_cols]
    # append
    if os.path.exists(out_path):
        try:
            new.to_csv(out_path, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Failed to append evaluations: {e}")
            return 1
    else:
        try:
            new.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Failed to write evaluations: {e}")
            return 1
    print(f"Appended {len(new)} evaluation rows to {out_path}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', required=True)
    parser.add_argument('--start', required=False)
    parser.add_argument('--end', required=False)
    parser.add_argument('--tolerance', type=float, default=0.2)
    args = parser.parse_args()
    rc = evaluate(args.region, args.start, args.end, args.tolerance)
    raise SystemExit(rc)
