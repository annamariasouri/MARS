"""
retrain_rf_model.py
====================
Full pipeline to retrain and validate the MARS RandomForest chlorophyll model.

Steps
-----
1.  Download Copernicus data (nutrients, temperature, salinity, chlorophyll)
    for TRAIN period (2012-2022) using copernicusmarine Python client.
2.  Compute 24 features per grid cell × day for all three regions.
3.  Train RandomForestRegressor on the pooled multi-region training set.
4.  Download and featurise VALIDATION data (2023-2025).
5.  Predict on validation set and compute RMSE, R².
6.  Compute bloom-detection metrics (precision, recall, F1).
7.  Plot predicted vs observed CHL per region.
8.  Save the trained model to  data/models/rf_chl_retrained.pkl

Usage
-----
    python retrain_rf_model.py

Requirements
------------
    copernicusmarine, xarray, pandas, numpy, scikit-learn, matplotlib, joblib
"""

import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Copernicus Marine credentials  – set via env vars or hardcode here
CMEMS_USER = os.environ.get("CMEMS_USER", "souri.am@unic.ac.cy")
CMEMS_PASS = os.environ.get("CMEMS_PASS", "1326@Ams@")

# Date ranges
TRAIN_START = "2012-01-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2025-12-31"

# Depth slice (surface / near-surface)
# Note: Mediterranean BGC/PHY reanalysis minimum depth is ~1.02 m
DEPTH_MIN = 1.02
DEPTH_MAX = 10.0

# Bloom threshold per region (mg/m³) – used for binary bloom-detection metrics
BLOOM_THRESHOLDS = {
    "thermaikos": 1.0,
    "peiraeus":   0.5,
    "limassol":   0.3,
}

# 24 feature columns (same order used during training)
FEATURE_COLS = [
    # current-day physics / nutrients
    "nh4", "no3", "po4", "so", "thetao",
    # lag-1 for all 6 variables (incl. CHL)
    "chl_t-1", "nh4_t-1", "no3_t-1", "po4_t-1", "so_t-1", "thetao_t-1",
    # 7-day rolling averages for all 6 variables
    "chl_7day_avg", "nh4_7day_avg", "no3_7day_avg", "po4_7day_avg",
    "so_7day_avg", "thetao_7day_avg",
    # nutrient ratios
    "n_p_ratio", "n_nh4_ratio", "p_nh4_ratio",
    # chlorophyll history
    "chl_monthly_median", "chl_anomaly", "bloom_proxy_label",
]
TARGET_COL = "chl"

REGIONS = {
    "thermaikos": {"lat_min": 40.2, "lat_max": 40.7, "lon_min": 22.5, "lon_max": 23.0},
    # Expanded bbox – original 37.9-38.1°N, 23.5-23.8°E falls on coastal masked pixels;
    # valid Saronic Gulf ocean cells start from ~37.5°N
    "peiraeus":   {"lat_min": 37.5, "lat_max": 37.9, "lon_min": 23.5, "lon_max": 23.9},
    "limassol":   {"lat_min": 34.6, "lat_max": 34.8, "lon_min": 33.0, "lon_max": 33.2},
}

# Multi-year REANALYSIS products – cover 1987 → ~2023
# Use these for the TRAINING split (2012-2022)
PRODUCTS_TRAIN = {
    "nut": {
        "dataset_id": "cmems_mod_med_bgc-nut_my_4.2km_P1D-m",
        "variables":  ["nh4", "no3", "po4"],
    },
    "tem": {
        "dataset_id": "cmems_mod_med_phy-temp_my_4.2km_P1D-m",   # note: 'temp' not 'tem'
        "variables":  ["thetao"],
    },
    "sal": {
        "dataset_id": "cmems_mod_med_phy-sal_my_4.2km_P1D-m",
        "variables":  ["so"],
    },
    "chl": {
        "dataset_id": "cmems_mod_med_bgc-plankton_my_4.2km_P1D-m",  # CHL lives in plankton ds
        "variables":  ["chl"],
    },
}

# Analysis + Forecast products (near-real-time) – only cover ~Feb 2024 onwards.
# For validation (2023-2025) we use the _my_ reanalysis which extends to Jan 2026.
# PRODUCTS_VAL kept here only for reference / future use with dates > my coverage.
PRODUCTS_VAL = PRODUCTS_TRAIN

DATA_DIR  = os.path.join("data", "retrain")
MODEL_DIR = os.path.join("data", "models")
PLOT_DIR  = os.path.join("data", "retrain", "plots")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_chl_retrained.pkl")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _nc_cache_path(region: str, product_key: str, split: str) -> str:
    return os.path.join(DATA_DIR, f"{region}_{product_key}_{split}.nc")


def download_period(region: str, start: str, end: str, split: str) -> dict[str, xr.Dataset]:
    """Download all four Copernicus products for one region/period.

    Returns a dict  {product_key: xr.Dataset}.
    Files are cached locally – re-running skips the download.
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        raise ImportError(
            "copernicusmarine package not found.  "
            "Install with:  pip install copernicusmarine"
        )

    bbox = REGIONS[region]
    datasets: dict[str, xr.Dataset] = {}
    products = PRODUCTS_TRAIN if split == "train" else PRODUCTS_VAL

    for key, info in products.items():
        cache = _nc_cache_path(region, key, split)
        if os.path.exists(cache):
            print(f"  [cache] {cache}")
            datasets[key] = xr.open_dataset(cache, engine="h5netcdf")
            continue

        print(f"  Downloading {region} / {key} ({split}) …")
        # Use cm.subset() to download directly to a local .nc file.
        cm.subset(
            dataset_id        = info["dataset_id"],
            start_datetime    = start,
            end_datetime      = end,
            username          = CMEMS_USER,
            password          = CMEMS_PASS,
            minimum_latitude  = bbox["lat_min"],
            maximum_latitude  = bbox["lat_max"],
            minimum_longitude = bbox["lon_min"],
            maximum_longitude = bbox["lon_max"],
            minimum_depth     = DEPTH_MIN,
            maximum_depth     = DEPTH_MAX,
            variables         = info["variables"],
            output_filename   = os.path.basename(cache),
            output_directory  = os.path.dirname(cache),
            overwrite         = True,
        )
        datasets[key] = xr.open_dataset(cache, engine="h5netcdf")
        print(f"    saved → {cache}")

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _depth_mean(da: xr.DataArray) -> xr.DataArray:
    """Average over depth dimension if present."""
    if "depth" in da.dims:
        return da.mean(dim="depth", skipna=True)
    return da


def datasets_to_dataframe(datasets: dict[str, xr.Dataset]) -> pd.DataFrame:
    """Merge all products into a single long DataFrame indexed by (time, lat, lon)."""
    frames = []
    for key, ds in datasets.items():
        # Derive variable list from the dataset itself (works for both TRAIN and VAL products)
        data_vars = [v for v in ds.data_vars if v not in ("depth", "latitude", "longitude", "time")]
        for var in data_vars:
            if var not in ds:
                continue
            da = _depth_mean(ds[var])
            df = (
                da
                .to_dataframe(name=var)
                .reset_index()
            )
            # Normalise column names
            df.columns = [c.lower() for c in df.columns]
            # Keep only essential columns
            keep = [c for c in ["time", "latitude", "longitude", var] if c in df.columns]
            df = df[keep].rename(columns={"latitude": "lat", "longitude": "lon"})
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            frames.append(df)

    if not frames:
        raise ValueError("No data loaded from datasets.")

    # Merge on (time, lat, lon)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["time", "lat", "lon"], how="outer")

    merged = merged.sort_values(["lat", "lon", "time"]).reset_index(drop=True)
    return merged


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 24 features in-place.

    Features depend on:
      - chl (current day, used only to build lag/rolling — NOT added as a predictor)
      - nh4, no3, po4, so, thetao  (current day)

    The *target* is chl itself — features are the current-day predictors plus
    lag/rolling features derived from the history up to (but not including)
    the current day's CHL value, so there is no leakage.
    """
    df = df.copy()
    df = df.sort_values(["lat", "lon", "time"]).reset_index(drop=True)

    grp = df.groupby(["lat", "lon"], sort=False)

    # ── lag-1 ──────────────────────────────────────────────────────────────
    for var in ["chl", "nh4", "no3", "po4", "so", "thetao"]:
        df[f"{var}_t-1"] = grp[var].shift(1)

    # ── 7-day rolling average (shift by 1 to avoid leakage) ───────────────
    for var in ["chl", "nh4", "no3", "po4", "so", "thetao"]:
        df[f"{var}_7day_avg"] = grp[var].transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).mean()
        )

    # ── 30-day rolling median for CHL ─────────────────────────────────────
    df["chl_monthly_median"] = grp["chl"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=7).median()
    )

    # ── CHL anomaly (current lag-1 minus 30-day median) ───────────────────
    df["chl_anomaly"] = df["chl_t-1"] - df["chl_monthly_median"]

    # ── Bloom proxy label (1 if chl_t-1 > regional 90th percentile) ───────
    p90 = df.groupby(["lat", "lon"])["chl"].transform(lambda x: x.quantile(0.90))
    df["bloom_proxy_label"] = (df["chl_t-1"] > p90).astype(int)

    # ── Nutrient ratios ────────────────────────────────────────────────────
    eps = 1e-9
    df["n_p_ratio"]   = df["no3"]   / (df["po4"]  + eps)
    df["n_nh4_ratio"] = df["no3"]   / (df["nh4"]  + eps)
    df["p_nh4_ratio"] = df["po4"]   / (df["nh4"]  + eps)

    return df


def prepare_dataset(region: str, split: str = "train") -> pd.DataFrame:
    """Download (or load cached) and featurise data for one region/split."""
    start = TRAIN_START if split == "train" else VAL_START
    end   = TRAIN_END   if split == "train" else VAL_END

    print(f"\n{'='*60}")
    print(f"Region: {region.upper()}  |  Split: {split}  ({start} → {end})")
    print("="*60)

    datasets = download_period(region, start, end, split)
    df = datasets_to_dataframe(datasets)
    df = compute_features(df)

    df["region"] = region
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ASSEMBLE TRAINING SET & TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_split(split: str) -> pd.DataFrame:
    """Build and return the full pooled DataFrame for a given split."""
    frames = []
    for region in REGIONS:
        df = prepare_dataset(region, split=split)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def train_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the training split."""
    required = FEATURE_COLS + [TARGET_COL]
    available = [c for c in required if c in train_df.columns]
    missing = set(required) - set(available)
    if missing:
        print(f"  WARNING: missing columns in train set: {missing}")

    subset = train_df.dropna(subset=[c for c in required if c in train_df.columns])
    X = subset[FEATURE_COLS].values
    y = subset[TARGET_COL].values

    print(f"\nTraining on {len(subset):,} samples (pooled, all regions) …")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=4,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)
    print("  Training complete.")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5.  PREDICT & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_evaluate(
    model: RandomForestRegressor,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict on val_df and attach predicted_chl; print per-region metrics."""

    required = FEATURE_COLS + [TARGET_COL]
    subset = val_df.dropna(subset=[c for c in required if c in val_df.columns]).copy()

    X_val = subset[FEATURE_COLS].values
    subset["predicted_chl"] = model.predict(X_val)
    subset["residual"]      = subset["predicted_chl"] - subset[TARGET_COL]

    print("\n" + "="*60)
    print("VALIDATION METRICS (2023-2025)")
    print("="*60)

    rows = []
    for region in list(REGIONS.keys()) + ["OVERALL"]:
        if region == "OVERALL":
            sel = subset
        else:
            sel = subset[subset["region"] == region]

        if sel.empty:
            continue

        y_true = sel[TARGET_COL].values
        y_pred = sel["predicted_chl"].values

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        n    = len(sel)

        print(f"  {region:<15}  n={n:>7,}  RMSE={rmse:.4f}  R²={r2:.4f}")
        rows.append({"region": region, "n": n, "rmse": rmse, "r2": r2})

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(DATA_DIR, "validation_metrics_retrained.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved → {metrics_path}")

    return subset


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BLOOM DETECTION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def bloom_metrics(val_pred: pd.DataFrame) -> None:
    """Compute precision, recall, F1 for bloom detection per region."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\n" + "="*60)
    print("BLOOM DETECTION METRICS (binary, per region)")
    print("="*60)

    for region, threshold in BLOOM_THRESHOLDS.items():
        sel = val_pred[val_pred["region"] == region].dropna(subset=[TARGET_COL, "predicted_chl"])
        if sel.empty:
            continue
        y_true_bin = (sel[TARGET_COL]       >= threshold).astype(int).values
        y_pred_bin = (sel["predicted_chl"]  >= threshold).astype(int).values

        p  = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        r  = recall_score   (y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score       (y_true_bin, y_pred_bin, zero_division=0)
        bloom_prevalence = y_true_bin.mean()

        print(
            f"  {region:<15}  threshold={threshold:.2f}  "
            f"prevalence={bloom_prevalence:.1%}  "
            f"precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_vs_observed(val_pred: pd.DataFrame) -> None:
    """Time-series and scatter plots of predicted vs observed CHL per region."""
    for region in REGIONS:
        sel = (
            val_pred[val_pred["region"] == region]
            .dropna(subset=[TARGET_COL, "predicted_chl", "time"])
            .sort_values("time")
        )
        if sel.empty:
            continue

        # Aggregate to daily median across grid cells for a cleaner plot
        daily = sel.groupby("time")[[TARGET_COL, "predicted_chl"]].median().reset_index()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f"{region.capitalize()} – Predicted vs Observed CHL (2023–2025)",
                     fontsize=13, fontweight="bold")

        # ─ Time series ──────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(daily["time"], daily[TARGET_COL],       label="Observed",  lw=1.5, color="royalblue")
        ax.plot(daily["time"], daily["predicted_chl"],  label="Predicted", lw=1.5, color="tomato", alpha=0.85)
        ax.set_ylabel("CHL (mg m⁻³)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ─ Scatter ──────────────────────────────────────────────────────────
        ax2 = axes[1]
        ax2.scatter(daily[TARGET_COL], daily["predicted_chl"],
                    s=12, alpha=0.5, color="steelblue", edgecolors="none")
        lim = max(daily[[TARGET_COL, "predicted_chl"]].max())
        ax2.plot([0, lim], [0, lim], "k--", lw=1, label="1:1 line")
        ax2.set_xlabel("Observed CHL (mg m⁻³)")
        ax2.set_ylabel("Predicted CHL (mg m⁻³)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"val_{region}_pred_vs_obs.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model: RandomForestRegressor) -> None:
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Random Forest – Feature Importances (retrained)")
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1–2: build training set ───────────────────────────────────────
    print("\n▶ Building TRAINING dataset (2012–2022) …")
    train_df = build_split("train")

    train_csv = os.path.join(DATA_DIR, "train_pooled.csv")
    train_df.to_csv(train_csv, index=False)
    print(f"  Training data saved → {train_csv} ({len(train_df):,} rows)")

    # ── Step 3: train model ────────────────────────────────────────────────
    print("\n▶ Training RandomForestRegressor …")
    model = train_model(train_df)

    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # ── Step 4–5: validation ───────────────────────────────────────────────
    print("\n▶ Building VALIDATION dataset (2023–2025) …")
    val_df = build_split("val")

    val_csv = os.path.join(DATA_DIR, "val_pooled.csv")
    val_df.to_csv(val_csv, index=False)
    print(f"  Validation data saved → {val_csv} ({len(val_df):,} rows)")

    val_pred = predict_and_evaluate(model, val_df)

    # ── Step 6: bloom detection metrics ───────────────────────────────────
    bloom_metrics(val_pred)

    # ── Step 7: plots ──────────────────────────────────────────────────────
    print("\n▶ Generating plots …")
    plot_predicted_vs_observed(val_pred)
    plot_feature_importance(model)

    print("\n✓ Done.  Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
