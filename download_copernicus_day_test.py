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

# === Process each region ===
for region, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
    print(f"\n🌍 Processing region: {region.upper()}")

    # store raw .nc per-region under RAW_DIR; do NOT chdir
    region_dir = os.path.join(RAW_DIR, f"{region}_downloads_{target_date_str}")
    region_dir = os.path.normpath(region_dir)
    os.makedirs(region_dir, exist_ok=True)

    print("📥 Starting downloads...")
    nc_files = []
    for dataset_id, vars in datasets:
        print(f"→ Downloading {vars} from {dataset_id}")
        # Retry download/open up to N times to avoid transient NetCDF/HDF errors on the runner
        max_attempts = 8
        attempt = 0
        success_path = None
        while attempt < max_attempts and success_path is None:
            attempt += 1
            try:
                response = copernicusmarine.subset(
                    dataset_id=dataset_id,
                    variables=vars,
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    start_datetime=(tgt - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00"),
                    end_datetime=target_date,
                    minimum_depth=1.0,
                    maximum_depth=5.0,
                    username=username,
                    password=password,
                    output_directory=region_dir
                )
                output_dir_attr = getattr(response, 'output_directory', None)
                filename_attr = getattr(response, 'filename', None)
                file_path = None
                if output_dir_attr and filename_attr:
                    file_path = os.path.join(region_dir, filename_attr)
                    print(f"DEBUG: expected file_path in region_dir = {file_path}")
                if file_path and isinstance(file_path, str) and os.path.exists(file_path):
                    # quick magic-byte check to detect HTML/error pages or corrupted downloads
                    def looks_like_netcdf(path):
                        try:
                            with open(path, 'rb') as fh:
                                head = fh.read(16)
                            # HDF5 files start with b'\x89HDF\r\n\x1a\n'
                            if head.startswith(b"\x89HDF\r\n\x1a\n"):
                                return True
                            # classic NetCDF may start with 'CDF' in first bytes
                            if head[0:3] == b"CDF":
                                return True
                            return False
                        except Exception:
                            return False

                    if not looks_like_netcdf(file_path):
                        # file doesn't look like a NetCDF/HDF5 file; capture a short preview and remove
                        preview = ""
                        try:
                            with open(file_path, 'rb') as fh:
                                preview = fh.read(512)
                                # limit size in the report
                                preview = binascii.hexlify(preview[:256]).decode('ascii')
                        except Exception:
                            preview = '<unreadable>'
                        try:
                            with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as efh:
                                efh.write(f"attempt={attempt}\nerror=invalid_magic\nfile={file_path}\nsize={os.path.getsize(file_path)}\npreview_hex={preview}\n---\n")
                        except Exception:
                            pass
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                        if attempt < max_attempts:
                            backoff = (2 ** attempt) + random.uniform(0, 2)
                            print(f"→ Retrying download after {backoff:.1f}s backoff due to invalid file magic...")
                            time.sleep(backoff)
                        continue
                    # validate the file can be opened
                    try:
                        ds_test = xr.open_dataset(file_path)
                        ds_test.close()
                        print(f"✅ File downloaded and opened: {file_path}")
                        success_path = file_path
                        nc_files.append(file_path)
                    except Exception as eopen:
                        # record detailed error
                        err_text = str(eopen)
                        size_info = None
                        try:
                            size_info = os.path.getsize(file_path)
                        except Exception:
                            pass
                        print(f"⚠️ Downloaded file could not be opened (attempt {attempt}): {err_text}; size={size_info}")
                        # append details to a report file for triage
                        try:
                            with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as efh:
                                efh.write(f"attempt={attempt}\nerror={err_text}\nfile={file_path}\nsize={size_info}\n---\n")
                        except Exception:
                            pass
                        # remove possibly corrupted file so retry gets a fresh download
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                        if attempt < max_attempts:
                            backoff = (2 ** attempt) + random.uniform(0, 2)
                            print(f"→ Retrying download after {backoff:.1f}s backoff...")
                            time.sleep(backoff)
                else:
                    print(f"⚠️ File not found in region directory after download (attempt {attempt}): {file_path}")
                    if attempt < max_attempts:
                        backoff = (2 ** attempt) + random.uniform(0, 2)
                        print(f"→ Retrying download after {backoff:.1f}s backoff...")
                        time.sleep(backoff)
            except Exception as e:
                err = str(e)
                print(f"⚠️ Download failed for {vars} in {region} (attempt {attempt}): {err}")
                try:
                    with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as efh:
                        efh.write(f"attempt={attempt}\nexception={err}\n---\n")
                except Exception:
                    pass
                if attempt < max_attempts:
                    backoff = (2 ** attempt) + random.uniform(0, 2)
                    print(f"→ Retrying download after {backoff:.1f}s backoff...")
                    time.sleep(backoff)
        if success_path is None:
            print(f"❌ Failed to obtain a valid .nc for {vars} in {region} after {max_attempts} attempts")
            # append a marker for debugging (use append so any earlier exception details are preserved)
            try:
                with open(os.path.join(REPORT_DIR, f"{region}_{target_date_str}_{dataset_id}.error"), "a") as fh:
                    fh.write(f"failed_to_download_or_open timestamp={datetime.utcnow().isoformat()} attempts={max_attempts}\n")
            except Exception:
                pass

    # Try to open all available .nc files, even if some are missing
    dfs = []
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            dfs.append(ds.to_dataframe().reset_index())
        except Exception as e:
            print(f"⚠️ Failed to open {f}: {e}")
    # write a simple per-region download report
    try:
        with open(os.path.join(REPORT_DIR, f"report_{region}_{target_date_str}.txt"), "w") as rfh:
            rfh.write(f"region={region}\n")
            rfh.write(f"nc_files={len(nc_files)}\n")
            rfh.write("files:\n")
            for p in nc_files:
                rfh.write(p + "\n")
    except Exception:
        pass

    # Merge all available datasets
    if dfs:
        df = dfs[0]
        for d in dfs[1:]:
            merge_keys = ['time', 'depth', 'latitude', 'longitude']
            df = df.merge(d, on=merge_keys, how='outer')
        df = df.sort_values(by=["latitude", "longitude", "time"]).reset_index(drop=True)
    else:
        print(f"⚠️ No valid .nc files for {region} after retries; skipping env history write.")
        df = pd.DataFrame()

    # Save environmental history (even if empty)
    env_cols = ["time", "chl", "nh4", "no3", "po4", "thetao", "so"]
    if not df.empty:
        df_env = df[env_cols].copy()
        df_env = df_env.groupby("time").mean().reset_index()
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
    else:
        df_ready = pd.DataFrame(columns=model_features)
    filename = f"model_ready_input_{region}_{target_date_str}.csv"
    final_path = os.path.join(MODEL_READY_DIR, filename)
    df_ready.to_csv(final_path, index=False)
    print(f"✅ Saved model input: {final_path}")
    print(df_ready.head())
