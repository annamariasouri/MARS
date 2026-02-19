"""
compare_baselines.py
====================
Compares the Random Forest CHL model against two baselines:
  1. Persistence  – predicts CHL_t = CHL_{t-1} (per region)
  2. Linear Regression – trained on the same feature set as the RF

Metrics computed per region: RMSE, MAE, R²

Input  : a pooled validation CSV (e.g. data/evaluation/validation_2024_2025.csv)
Output : a printed summary table + optional CSV saved to data/evaluation/

Usage
-----
    python compare_baselines.py
    python compare_baselines.py --csv data/evaluation/validation_2024_2025.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration – edit these if your column names differ
# ---------------------------------------------------------------------------

# Path to the pooled validation CSV
DEFAULT_CSV = os.path.join("data", "evaluation", "validation_2024_2025.csv")

# Column that holds the date/time
DATE_COL = "time"

# Column that identifies the region
REGION_COL = "region"

# Column with ground-truth chlorophyll-a
OBS_COL = "observed_chl"

# Column with the RF model predictions (pre-computed in the CSV)
RF_COL = "predicted_chl"

# Feature columns – must match exactly what the RF model was trained on
FEATURE_COLS = [
    # current-day physics / nutrients
    "nh4", "no3", "po4", "so", "thetao",
    # lag-1 for all 6 variables (including CHL)
    "chl_t-1", "nh4_t-1", "no3_t-1", "po4_t-1", "so_t-1", "thetao_t-1",
    # 7-day rolling averages for all 6 variables
    "chl_7day_avg", "nh4_7day_avg", "no3_7day_avg", "po4_7day_avg",
    "so_7day_avg", "thetao_7day_avg",
    # nutrient ratios
    "n_p_ratio", "n_nh4_ratio", "p_nh4_ratio",
    # chlorophyll history features
    "chl_monthly_median", "chl_anomaly", "bloom_proxy_label",
]

# LR-safe feature columns: CHL-derived features are removed to prevent data
# leakage. chl_t-1, chl_7day_avg, chl_monthly_median, chl_anomaly, and
# bloom_proxy_label are all direct functions of the target variable and give
# LR an unfair advantage (near-perfect in-sample fit).
LR_FEATURE_COLS = [
    # current-day physics / nutrients only
    "nh4", "no3", "po4", "so", "thetao",
    # lag-1 for non-CHL variables
    "nh4_t-1", "no3_t-1", "po4_t-1", "so_t-1", "thetao_t-1",
    # 7-day rolling averages for non-CHL variables
    "nh4_7day_avg", "no3_7day_avg", "po4_7day_avg",
    "so_7day_avg", "thetao_7day_avg",
    # nutrient ratios
    "n_p_ratio", "n_nh4_ratio", "p_nh4_ratio",
]

# Number of TimeSeriesSplit folds for the LR cross-validation.
# If a region has too few samples to support this many folds, it is
# reduced automatically so each test fold has at least 1 sample.
CV_N_SPLITS = 5

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, R² for a pair of arrays, ignoring NaNs."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "N": int(mask.sum())}

    yt, yp = y_true[mask], y_pred[mask]
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":  float(mean_absolute_error(yt, yp)),
        "R2":   float(r2_score(yt, yp)),
        "N":    int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(csv_path: str) -> None:
    # ------------------------------------------------------------------
    # 1. Load and basic validation
    # ------------------------------------------------------------------
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    print(f"  Shape: {df.shape}  |  Regions: {sorted(df[REGION_COL].unique())}\n")

    # Verify required columns exist
    required_cols = [DATE_COL, REGION_COL, OBS_COL, RF_COL] + FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Sort chronologically within each region (needed for the persistence lag)
    df = df.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)

    # Drop rows where the observation itself is NaN – nothing to evaluate on
    df = df.dropna(subset=[OBS_COL]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Persistence baseline: CHL_t = CHL_{t-1}  (per region)
    #    We shift the observation column by 1 within each region group.
    # ------------------------------------------------------------------
    df["CHL_persistence"] = df.groupby(REGION_COL)[OBS_COL].shift(1)

    # ------------------------------------------------------------------
    # 3. Linear Regression baseline  (honest: TimeSeriesSplit CV)
    #
    #    Two deliberate design choices vs. plain in-sample evaluation:
    #
    #    a) CHL-derived features excluded (LR_FEATURE_COLS).
    #       chl_t-1, chl_7day_avg, chl_monthly_median, chl_anomaly and
    #       bloom_proxy_label are direct linear functions of the target;
    #       keeping them lets LR trivially reconstruct CHL with RMSE≈0.
    #
    #    b) TimeSeriesSplit cross-validation per region.
    #       The model is never evaluated on data it was trained on, giving
    #       honest out-of-fold predictions that are comparable to RF.
    # ------------------------------------------------------------------
    df["CHL_linear"] = np.nan  # initialise column

    for region, grp in df.groupby(REGION_COL):
        # Work on the chronologically-sorted subset with complete features
        idx_all = grp.index
        clean   = grp[LR_FEATURE_COLS + [OBS_COL]].dropna()
        idx_clean = clean.index

        n = len(clean)
        if n < 4:
            # Too few samples for any meaningful split – skip
            print(f"  [LR] {region}: only {n} complete rows – skipping CV.")
            continue

        # Adapt number of folds so every fold has >= 1 test sample
        n_splits = min(CV_N_SPLITS, n - 1)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        oof_preds = np.full(n, np.nan)
        for train_idx, test_idx in tscv.split(clean):
            X_train = clean.iloc[train_idx][LR_FEATURE_COLS].values
            y_train = clean.iloc[train_idx][OBS_COL].values
            X_test  = clean.iloc[test_idx][LR_FEATURE_COLS].values

            lr = LinearRegression()
            lr.fit(X_train, y_train)
            oof_preds[test_idx] = lr.predict(X_test)

        # Write OOF predictions back into the main dataframe
        df.loc[idx_clean, "CHL_linear"] = oof_preds

    # ------------------------------------------------------------------
    # 4. Compute metrics per region for RF, Persistence, Linear Regression
    # ------------------------------------------------------------------
    models = {
        "RandomForest":       RF_COL,
        "Persistence":        "CHL_persistence",
        "LinearRegression":   "CHL_linear",
    }

    rows = []
    for region, grp in df.groupby(REGION_COL):
        y_true = grp[OBS_COL].values
        for model_name, pred_col in models.items():
            y_pred = grp[pred_col].values if pred_col in grp.columns else np.full_like(y_true, np.nan, dtype=float)
            m = compute_metrics(y_true.astype(float), y_pred.astype(float))
            rows.append({
                "Region":    region,
                "Model":     model_name,
                "RMSE":      round(m["RMSE"], 6),
                "MAE":       round(m["MAE"],  6),
                "R²":        round(m["R2"],   4),
                "N_samples": m["N"],
            })

    results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Also compute overall (all regions pooled)
    # ------------------------------------------------------------------
    overall_rows = []
    y_true_all = df[OBS_COL].values.astype(float)
    for model_name, pred_col in models.items():
        y_pred_all = df[pred_col].values.astype(float) if pred_col in df.columns else np.full_like(y_true_all, np.nan)
        m = compute_metrics(y_true_all, y_pred_all)
        overall_rows.append({
            "Region":    "OVERALL",
            "Model":     model_name,
            "RMSE":      round(m["RMSE"], 6),
            "MAE":       round(m["MAE"],  6),
            "R²":        round(m["R2"],   4),
            "N_samples": m["N"],
        })

    results = pd.concat([results, pd.DataFrame(overall_rows)], ignore_index=True)

    # ------------------------------------------------------------------
    # 6. Sort and print
    # ------------------------------------------------------------------
    # Custom sort: regions alphabetically, OVERALL last; models in logical order
    model_order  = {"RandomForest": 0, "Persistence": 1, "LinearRegression": 2}
    region_order = {r: i for i, r in enumerate(sorted(df[REGION_COL].unique()))}
    region_order["OVERALL"] = len(region_order)

    results["_r"] = results["Region"].map(region_order)
    results["_m"] = results["Model"].map(model_order)
    results = results.sort_values(["_r", "_m"]).drop(columns=["_r", "_m"]).reset_index(drop=True)

    # Pretty-print
    pd.set_option("display.max_columns",  None)
    pd.set_option("display.width",        120)
    pd.set_option("display.float_format", "{:.6f}".format)

    print("=" * 72)
    print("  Model Comparison: RF vs Persistence vs LR (CV, no CHL leakage)")
    print("=" * 72)
    print(results.to_string(index=False))
    print("=" * 72)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    out_dir  = os.path.join("data", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_comparison.csv")
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RF vs baseline models")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to pooled validation CSV (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()
    main(args.csv)
