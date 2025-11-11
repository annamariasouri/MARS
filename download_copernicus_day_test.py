import copernicusmarine
import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import random
import argparse
import binascii

# === Parameters ===
parser = argparse.ArgumentParser(description='Download Copernicus data for a given date (default: yesterday)')
parser.add_argument('--date', help='Target date YYYY-MM-DD (defaults to yesterday)', default=None)
args = parser.parse_args()
if args.date:
    try:
        tgt = datetime.strptime(args.date, '%Y-%m-%d')
    except Exception:
        raise ValueError('Invalid --date format, expected YYYY-MM-DD')
else:
    tgt = datetime.today() - timedelta(days=1)

target_date = tgt.strftime("%Y-%m-%dT00:00:00")
target_date_str = tgt.strftime("%Y-%m-%d")

# Debug date information
print(f"Debug: Using date {target_date_str} for downloads and processing")

# Default regions and their bounding boxes (lat_min, lat_max, lon_min, lon_max)
DEFAULT_REGIONS = {
    "thermaikos": (40.2, 40.7, 22.5, 23.0),
    "peiraeus": (37.9, 38.1, 23.5, 23.8),
    "limassol": (34.6, 34.8, 33.0, 33.2)
}

# Allow overriding which regions to process via environment variable MARS_REGIONS
# Example: export MARS_REGIONS=limassol,thermaikos
env_regions = os.environ.get("MARS_REGIONS", None)
if env_regions:
    selected = [r.strip() for r in env_regions.split(',') if r.strip()]
    REGIONS = {k: v for k, v in DEFAULT_REGIONS.items() if k in selected}
else:
    REGIONS = DEFAULT_REGIONS

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
ENV_DIR = os.path.join(DATA_DIR, "env_history")
MODEL_READY_DIR = os.path.join(DATA_DIR, "model_ready")
RAW_DIR = os.path.join(DATA_DIR, "raw_nc")
REPORT_DIR = os.path.join(DATA_DIR, "download_reports")
for d in (DATA_DIR, ENV_DIR, MODEL_READY_DIR, RAW_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# === Authenticate ===
username = os.getenv('COPERNICUS_USERNAME', '').strip()
password = os.getenv('COPERNICUS_PASSWORD', '').strip()

if not username or not password:
    raise ValueError("Copernicus credentials not found in environment variables. Please set COPERNICUS_USERNAME and COPERNICUS_PASSWORD")
copernicusmarine.login(username=username, password=password, check_credentials_valid=True, force_overwrite=True)

# === Dataset list ===
datasets = [
    ("cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m", ["nh4", "no3", "po4"]),
    ("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m", ["thetao"]),
    ("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m", ["so"]),
    ("cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m", ["chl"])
]

# Requested depth window (user-facing)
REQUESTED_MIN_DEPTH = 1.0
REQUESTED_MAX_DEPTH = 5.0

# Known dataset coordinate bounds (used to clamp requests and avoid warnings)
# These are conservative defaults observed for the CMEMS MED products; they
# prevent the library from warning when the requested min/max slightly fall
# outside exact grid coordinates. If datasets change, adjust these values.
DEPTH_COORD_MIN = 1.0182366371154785
DEPTH_COORD_MAX = 5754.0439453125

# === Process each region ===
for region, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
    print(f"\n🌍 Processing region: {region.upper()}")

    # store raw .nc per-region under RAW_DIR; do NOT chdir
    region_dir = os.path.join(RAW_DIR, f"{region}_downloads_{target_date_str}")
    region_dir = os.path.normpath(region_dir)
    os.makedirs(region_dir, exist_ok=True)

    print("📥 Starting downloads (using read_dataframe -> CSV)...")
    # Try progressively larger bounding boxes if the initial bbox returns no measurement values.
    # This helps for near-shore ports where the immediate bbox may be masked or have no valid grid cells.
    bbox_expansions = [0.0, 0.05, 0.1, 0.25, 0.5]
    found_non_empty = False
    df = pd.DataFrame()
    # dataset_csv_files will capture the CSVs for the successful expansion (if any)
    dataset_csv_files = []
    for expand in bbox_expansions:
        cur_lat_min = lat_min - expand
        cur_lat_max = lat_max + expand
        cur_lon_min = lon_min - expand
        cur_lon_max = lon_max + expand
        print(f"→ Attempting bbox expansion ±{expand}°: lat {cur_lat_min}/{cur_lat_max}, lon {cur_lon_min}/{cur_lon_max}")
        df_list = []
        local_csv_files = []
        for dataset_id, vars in datasets:
            print(f"→ Fetching {vars} from {dataset_id} as DataFrame")
            # Retry read_dataframe up to N times to avoid transient errors on the runner
            max_attempts = 8
            attempt = 0
            success_df = None
            while attempt < max_attempts and success_df is None:
                attempt += 1
                try:
                    # call read_dataframe to get a pandas DataFrame directly (avoids saving .nc files)
                    # clamp requested depths to known dataset bounds to avoid warnings
                    min_depth = max(REQUESTED_MIN_DEPTH, DEPTH_COORD_MIN)
                    max_depth = min(REQUESTED_MAX_DEPTH, DEPTH_COORD_MAX)
                    if (min_depth != REQUESTED_MIN_DEPTH) or (max_depth != REQUESTED_MAX_DEPTH):
                        print(f"ℹ️ Clamping requested depth range {REQUESTED_MIN_DEPTH}-{REQUESTED_MAX_DEPTH} to {min_depth}-{max_depth} to match dataset coords")

                    df_tmp = copernicusmarine.read_dataframe(
                        dataset_id=dataset_id,
                        variables=vars,
                        minimum_longitude=cur_lon_min,
                        maximum_longitude=cur_lon_max,
                        minimum_latitude=cur_lat_min,
                        maximum_latitude=cur_lat_max,
                        start_datetime=(tgt - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00"),
                        end_datetime=target_date,
                        minimum_depth=min_depth,
                        maximum_depth=max_depth,
                        username=username,
                        password=password,
                    )
                    # read_dataframe may return None or an empty dataframe on failure
                    if df_tmp is None or (hasattr(df_tmp, 'empty') and df_tmp.empty):
                        print(f"⚠️ Received empty DataFrame for {dataset_id} (attempt {attempt})")
                        try:
                            with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as efh:
                                efh.write(f"attempt={attempt}\nerror=empty_dataframe\n---\n")
                        except Exception:
                            pass
                        if attempt < max_attempts:
                            backoff = (2 ** attempt) + random.uniform(0, 2)
                            print(f"→ Retrying read_dataframe after {backoff:.1f}s backoff...")
                            time.sleep(backoff)
                        continue
                    # normalize to DataFrame with coords as columns
                    try:
                        df_tmp = df_tmp.reset_index()
                    except Exception:
                        pass
                    # save a CSV copy of the raw dataset for records
                    safe_vars = "-".join(vars)
                    csv_name = f"{dataset_id}_{safe_vars}_{target_date_str}.csv"
                    csv_path = os.path.join(region_dir, csv_name)
                    try:
                        df_tmp.to_csv(csv_path, index=False)
                        local_csv_files.append(csv_path)
                    except Exception:
                        # if saving fails, continue but still use the in-memory df
                        pass
                    print(f"✅ DataFrame fetched for {dataset_id}, rows={len(df_tmp)}")
                    success_df = df_tmp
                    df_list.append(df_tmp)
                except Exception as e:
                    err = str(e)
                    print(f"⚠️ read_dataframe failed for {vars} in {region} (attempt {attempt}): {err}")
                    try:
                        with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as efh:
                            efh.write(f"attempt={attempt}\nexception={err}\n---\n")
                    except Exception:
                        pass
                    if attempt < max_attempts:
                        backoff = (2 ** attempt) + random.uniform(0, 2)
                        print(f"→ Retrying read_dataframe after {backoff:.1f}s backoff...")
                        time.sleep(backoff)
            if success_df is None:
                print(f"❌ Failed to obtain a valid DataFrame for {vars} in {region} after {max_attempts} attempts")
                try:
                    with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as fh:
                        fh.write(f"failed_to_download_or_open timestamp={datetime.utcnow().isoformat()} attempts={max_attempts}\n")
                except Exception:
                    pass

        # after attempting all datasets for this bbox expansion, try to merge and check if we have any non-empty measurement rows
        if df_list:
            tmp = df_list[0]
            for d in df_list[1:]:
                merge_keys = ['time', 'depth', 'latitude', 'longitude']
                tmp = tmp.merge(d, on=merge_keys, how='outer')
            # normalize and check if any measurement columns have non-null values
            tmp_cols = [c.lower() for c in tmp.columns]
            measurement_like = any(('chl' in c or 'no3' in c or 'nh4' in c or 'po4' in c or 'theta' in c or 'sal' in c) for c in tmp_cols)
            # simple non-empty check: any non-null in common variable names
            non_null_count = tmp.dropna(how='all', subset=[c for c in tmp.columns if any(x in c.lower() for x in ['chl','no3','nh4','po4','theta','sal'])]).shape[0]
            if measurement_like and non_null_count > 0:
                df = tmp.copy()
                found_non_empty = True
                dataset_csv_files = local_csv_files
                print(f"✅ Found non-empty measurements with bbox expansion ±{expand}°")
                break
            else:
                print(f"⚠️ No measurement values found for bbox ±{expand}°; trying next expansion")
        else:
            print(f"⚠️ No dataframes returned for bbox ±{expand}°; trying next expansion")
        # cleanup before next expansion
        df_list = []
        local_csv_files = []
    # end bbox expansion loop
    if not found_non_empty:
        # if we never found any non-empty measurement rows, fall back to last attempt (may be empty)
        print(f"❌ No non-empty measurements found for {region} within expansion radius; proceeding with empty env_history/model_ready")

    # write a simple per-region download report (CSV copies count)
    try:
        with open(os.path.join(REPORT_DIR, f"report_{region}_{target_date_str}.txt"), "w") as rfh:
            rfh.write(f"region={region}\n")
            rfh.write(f"found_non_empty={found_non_empty}\n")
            rfh.write(f"files:\n")
            for p in dataset_csv_files:
                rfh.write(p + "\n")
    except Exception:
        pass

    # At this point `df` is either the merged DataFrame from a successful bbox expansion
    # or an empty DataFrame if no measurements were found.

    # Save environmental history (even if empty)
    env_cols = ["time", "chl", "nh4", "no3", "po4", "thetao", "so"]
    # Normalize column names from datasets to our canonical env names so aggregation works.
    # Some CMEMS products use slightly different names (e.g. 'chlorophyll', 'chl_mg_m3', 'pft_chl', 'salinity', 'temperature').
    if not df.empty:
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if 'chl' in cl or 'chlorophyll' in cl or ('pft' in cl and 'chl' in cl) or 'chla' in cl:
                col_map[c] = 'chl'
            elif 'nh4' in cl or 'ammonium' in cl:
                col_map[c] = 'nh4'
            elif 'no3' in cl or 'nitrate' in cl:
                col_map[c] = 'no3'
            elif 'po4' in cl or 'phosphate' in cl:
                col_map[c] = 'po4'
            elif 'theta' in cl or 'temp' in cl or 'temperature' in cl or 'sst' in cl:
                col_map[c] = 'thetao'
            elif cl in ('so',) or 'sal' in cl or 'salinity' in cl:
                col_map[c] = 'so'
        if col_map:
            try:
                df = df.rename(columns=col_map)
            except Exception:
                pass

    if not df.empty:
        # Only keep env columns that actually exist after renaming
        present = [c for c in env_cols if c in df.columns]
        if present:
            df_env = df[present].copy()
            # if time still present, ensure grouping works
            if 'time' in df_env.columns:
                df_env = df_env.groupby('time').mean().reset_index()
        else:
            df_env = pd.DataFrame(columns=env_cols)
    else:
        df_env = pd.DataFrame(columns=env_cols)
    history_filename = f"env_history_{region}_{target_date_str}.csv"
    # Only write per-day env_history if we have data; otherwise skip to avoid creating empty files
    if not df_env.empty:
        df_env.to_csv(os.path.join(ENV_DIR, history_filename), index=False)
        print(f"📤 Saved environmental history: {os.path.join(ENV_DIR, history_filename)}")
        # Clean up any previous failure markers for this region/date because we have valid data now
        try:
            import glob
            # remove the simple .no_rows marker
            no_rows_path = os.path.join(REPORT_DIR, f"{region}_{target_date_str}.no_rows")
            if os.path.exists(no_rows_path):
                os.remove(no_rows_path)
            # remove any per-dataset .error files for this region/date (they may be from earlier failed runs)
            for p in glob.glob(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_*.error")):
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass
    else:
        print(f"⚠️ No valid environmental rows for {region} on {target_date_str}; not writing {history_filename}")
        # write a marker so merge can detect a failed date later
        try:
            with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}.no_rows"), "w") as fh:
                fh.write("no_rows_after_processing\n")
        except Exception:
            pass

    # Feature engineering (robust to empty df)
    if not df.empty:
        group = df.groupby(['latitude', 'longitude'])
        def add_lag_and_rolling(df, var):
            df[f"{var}_t-1"] = group[var].shift(1)
            df[f"{var}_7day_avg"] = group[var].rolling(window=7, min_periods=1).mean().reset_index(drop=True)
        for var in ["chl", "nh4", "no3", "po4", "so", "thetao"]:
            add_lag_and_rolling(df, var)
        df["chl_monthly_median"] = group["chl"].rolling(window=30, min_periods=1).median().reset_index(drop=True)
        df["chl_anomaly"] = df["chl"] - df["chl_monthly_median"]
        df["n_p_ratio"] = np.where(df["po4"] != 0, df["no3"] / df["po4"], np.nan)
        df["n_nh4_ratio"] = np.where(df["nh4"] != 0, df["no3"] / df["nh4"], np.nan)
        df["p_nh4_ratio"] = np.where(df["nh4"] != 0, df["po4"] / df["nh4"], np.nan)
        df["bloom_proxy_label"] = 0
        df_latest = df[df["time"] == df["time"].max()]
    else:
        df_latest = pd.DataFrame()

    model_features = [
        'nh4','no3','po4','so','thetao',
        'chl_t-1','nh4_t-1','no3_t-1','po4_t-1','so_t-1','thetao_t-1',
        'chl_7day_avg','nh4_7day_avg','no3_7day_avg','po4_7day_avg','so_7day_avg','thetao_7day_avg',
        'n_p_ratio','n_nh4_ratio','p_nh4_ratio',
        'chl_monthly_median','chl_anomaly','bloom_proxy_label'
    ]
    # Always write a model input file, even if empty
    if not df_latest.empty:
        df_ready = df_latest.dropna(subset=model_features)[model_features]
        # If strict dropna removed all rows, fall back to a simple median-imputation
        # using the available historical values in `df` so we can still produce a forecast.
        if df_ready.empty:
            try:
                med = df[model_features].median().fillna(0)
                df_ready = df_latest[model_features].fillna(med)
                print("ℹ️ model_ready fields had missing values; applied median imputation to produce model input rows")
            except Exception:
                # If anything goes wrong, keep df_ready empty (we'll still write an empty file)
                df_ready = pd.DataFrame(columns=model_features)
    else:
        df_ready = pd.DataFrame(columns=model_features)
    filename = f"model_ready_input_{region}_{target_date_str}.csv"
    final_path = os.path.join(MODEL_READY_DIR, filename)
    df_ready.to_csv(final_path, index=False)
    print(f"✅ Saved model input: {final_path}")
    print(df_ready.head())
