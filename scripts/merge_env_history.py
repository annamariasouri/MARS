"""Merge dated env_history CSV files into single env_history_{region}.csv files.

Usage:
    python scripts/merge_env_history.py

This script will:
- Look for files matching env_history_{region}_*.csv and env_history_{region}.csv
- Read and validate files (ensure a parseable TIME column)
- Concatenate, de-duplicate by TIME and sort
- Backup existing env_history_{region}.csv and write the merged file

Designed to be safe and idempotent.
"""
from __future__ import annotations
import os
import glob
import pandas as pd
from datetime import datetime

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
ENV_DIR = os.path.join(DATA_DIR, "env_history")
os.makedirs(ENV_DIR, exist_ok=True)
env_regions = os.environ.get("MARS_REGIONS", None)
if env_regions:
    REGIONS = [r.strip() for r in env_regions.split(',') if r.strip()]
else:
    REGIONS = ["thermaikos", "peiraeus", "limassol"]

# common time column names to check
TIME_CANDIDATES = ["time", "date", "datetime", "ts", "TIME"]


def _read_and_normalize(path: str) -> pd.DataFrame:
    """Read a CSV and normalize to a dataframe with a TIME column (datetime dtype).
    Returns empty DataFrame on failure or if no parseable TIME found.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # find time-like column
    tcol = None
    for c in df.columns:
        if c.lower() in TIME_CANDIDATES:
            tcol = c
            break
    if tcol is None:
        # try to parse any column
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    df.insert(0, "TIME", parsed)
                    tcol = "TIME"
                    break
            except Exception:
                continue
    else:
        df = df.rename(columns={tcol: "TIME"})
        try:
            df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        except Exception:
            pass

    if "TIME" not in df.columns or df["TIME"].dropna().empty:
        return pd.DataFrame()

    # remove rows without TIME
    df = df.dropna(subset=["TIME"]).copy()

    # uppercase common vars
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("chl", "chlorophyll", "chl_mg_m3"):
            rename_map[c] = "CHL"
        elif cl in ("thetao", "theta", "sst", "t", "temp", "temperature"):
            rename_map[c] = "THETAO"
        elif cl in ("so", "sal", "salinity"):
            rename_map[c] = "SO"
        elif cl in ("nh4",):
            rename_map[c] = "NH4"
        elif cl in ("no3",):
            rename_map[c] = "NO3"
        elif cl in ("po4",):
            rename_map[c] = "PO4"
    if rename_map:
        df = df.rename(columns=rename_map)

    # keep TIME as first column
    cols = [c for c in df.columns if c != "TIME"]
    df = df[["TIME"] + cols]
    return df


def merge_region(region: str) -> str | None:
    """Merge files for a region and write env_history_{region}.csv in DATA_DIR.
    Returns path to written file or None on failure/no valid files.
    """
    pattern = os.path.join(ENV_DIR, f"env_history_{region}_*.csv")
    dated = glob.glob(pattern)
    undated = os.path.join(ENV_DIR, f"env_history_{region}.csv")

    frames = []
    # read undated first (so its rows can be preferred)
    if os.path.exists(undated):
        dfu = _read_and_normalize(undated)
        if not dfu.empty:
            frames.append(dfu)

    # read dated files
    for p in sorted(dated):
        if os.path.abspath(p) == os.path.abspath(undated):
            continue
        dfd = _read_and_normalize(p)
        if not dfd.empty:
            frames.append(dfd)

    if not frames:
        return None

    # concat and dedupe by TIME (keep last occurrence)
    big = pd.concat(frames, ignore_index=True)
    # round TIME to date if times are 00:00:00 etc - keep as datetime
    big = big.drop_duplicates(subset=["TIME"], keep="last")
    big = big.sort_values("TIME").reset_index(drop=True)

    # backup existing undated file
    if os.path.exists(undated):
        bak = undated + ".bak." + datetime.now().strftime("%Y%m%dT%H%M%S")
        try:
            os.replace(undated, bak)
        except Exception:
            pass

    # write merged file with 'time' lowercase header for compatibility
    out = big.copy()
    out = out.rename(columns={"TIME": "time"})
    out.to_csv(undated, index=False)
    return undated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge dated env_history CSVs into master files.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Optional start date (YYYY-MM-DD). Only dated files on/after this date will be considered.")
    parser.add_argument("--force", action="store_true",
                        help="If set, existing master env_history_{region}.csv will be removed before merging.")
    args = parser.parse_args()

    # helper to decide whether a dated file is >= start-date
    def _dated_file_in_range(path: str, start_date: str | None) -> bool:
        if start_date is None:
            return True
        # attempt to extract YYYY-MM-DD from filename
        basename = os.path.basename(path)
        # look for a date-like segment
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", basename)
        if not m:
            return True
        try:
            file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            return file_date >= start
        except Exception:
            return True

    updated = {}
    for r in REGIONS:
        # if force is set, remove existing undated master so we start fresh
        undated = os.path.join(ENV_DIR, f"env_history_{r}.csv")
        if args.force and os.path.exists(undated):
            try:
                os.remove(undated)
                print(f"Removed existing master file for {r}: {undated}")
            except Exception as e:
                print(f"Could not remove {undated}: {e}")

        # collect dated files filtered by start-date
        pattern = os.path.join(ENV_DIR, f"env_history_{r}_*.csv")
        dated_files = sorted(glob.glob(pattern))
        if args.start_date:
            dated_files = [p for p in dated_files if _dated_file_in_range(p, args.start_date)]

        # temporarily override globbing in merge_region by injecting the filtered list
        # We'll implement merging here to honor the filtered set.
        frames = []
        # read existing undated first (if any)
        if os.path.exists(undated):
            dfu = _read_and_normalize(undated)
            if not dfu.empty:
                frames.append(dfu)

        for p in dated_files:
            if os.path.abspath(p) == os.path.abspath(undated):
                continue
            dfd = _read_and_normalize(p)
            if not dfd.empty:
                frames.append(dfd)

        if not frames:
            updated[r] = None
            print(f"No valid env_history files found for {r} after applying start-date filter.")
            continue

        big = pd.concat(frames, ignore_index=True)
        big = big.drop_duplicates(subset=["TIME"], keep="last")
        big = big.sort_values("TIME").reset_index(drop=True)

        # backup existing undated file if present
        if os.path.exists(undated):
            bak = undated + ".bak." + datetime.now().strftime("%Y%m%dT%H%M%S")
            try:
                os.replace(undated, bak)
            except Exception:
                pass

        out = big.copy()
        out = out.rename(columns={"TIME": "time"})
        out.to_csv(undated, index=False)
        updated[r] = undated

    for r, p in updated.items():
        if p:
            print(f"Merged for {r}: {p}")
        else:
            print(f"No valid env_history files found for {r}.")
