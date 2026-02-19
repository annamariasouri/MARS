"""
diagnose_lr.py
==============
Diagnostic script to investigate why Linear Regression produces suspiciously
perfect metrics (RMSE≈0, R²≈1) on the CHL validation dataset.

Checks performed
----------------
1.  Unique CHL_obs values per region  – detects near-constant targets
2.  Constant / near-constant features per region  – detects zero-variance predictors
3.  Feature-target correlations  – identifies which features are almost identical to CHL
4.  Sample count per region  – detects n_samples << n_features (over-determination)
5.  Per-region LR fit: RMSE, MAE, R², max absolute error
6.  Scatter plots: observed vs predicted per region (saved to disk)

Based on typical patterns this will reveal one or more of:
  - Very few unique training samples per region  →  rank deficiency
  - Lag / rolling features that are linear functions of the target  →  data leakage
  - n_features >= n_samples  →  LR interpolates perfectly
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works everywhere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration – edit these to match your CSV column names
# ---------------------------------------------------------------------------
CSV_PATH   = os.path.join("data", "evaluation", "validation_2024_2025.csv")
DATE_COL   = "time"            # date/time column
REGION_COL = "region"          # region identifier column
OBS_COL    = "observed_chl"    # ground-truth CHL column
RF_COL     = "predicted_chl"   # pre-computed RF predictions

FEATURE_COLS = [
    "nh4", "no3", "po4", "so", "thetao",
    "chl_t-1", "nh4_t-1", "no3_t-1", "po4_t-1", "so_t-1", "thetao_t-1",
    "chl_7day_avg", "nh4_7day_avg", "no3_7day_avg", "po4_7day_avg",
    "so_7day_avg", "thetao_7day_avg",
    "n_p_ratio", "n_nh4_ratio", "p_nh4_ratio",
    "chl_monthly_median", "chl_anomaly", "bloom_proxy_label",
]

PLOT_DIR = os.path.join("data", "evaluation", "lr_diagnostics")
os.makedirs(PLOT_DIR, exist_ok=True)

DIVIDER  = "=" * 72
DIVIDER2 = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    """Compute RMSE/MAE/R² on non-NaN pairs.  Returns a dict."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    n = mask.sum()
    if n < 2:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MaxAbsErr=np.nan, N=n)
    yt, yp = y_true[mask], y_pred[mask]
    return dict(
        RMSE      = float(np.sqrt(mean_squared_error(yt, yp))),
        MAE       = float(mean_absolute_error(yt, yp)),
        R2        = float(r2_score(yt, yp)),
        MaxAbsErr = float(np.max(np.abs(yt - yp))),
        N         = int(n),
    )


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(DIVIDER)
print("  DIAGNOSTIC: Linear Regression on CHL Validation Data")
print(DIVIDER)

df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL])
df = df.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)
df = df.dropna(subset=[OBS_COL]).reset_index(drop=True)

regions = sorted(df[REGION_COL].unique())
print(f"\nDataset shape : {df.shape}")
print(f"Regions found : {regions}")
print(f"Date range    : {df[DATE_COL].min().date()}  →  {df[DATE_COL].max().date()}")


# ---------------------------------------------------------------------------
# 2. CHECK 1 – unique observed CHL values per region
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 1 – Unique CHL_obs values per region")
print(DIVIDER2)

for region, grp in df.groupby(REGION_COL):
    n_total  = len(grp)
    n_unique = grp[OBS_COL].nunique()
    ratio    = n_unique / n_total
    flag     = " ← VERY FEW UNIQUE VALUES" if n_unique <= 5 else ""
    print(f"  {region:<14}  total={n_total:>4}   unique_CHL={n_unique:>4}  "
          f"({ratio:.1%} distinct){flag}")


# ---------------------------------------------------------------------------
# 3. CHECK 2 – constant or near-constant features per region
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 2 – Constant / near-constant features per region  (std ≈ 0)")
print(DIVIDER2)

CONST_THRESHOLD = 1e-8   # standard deviation below this → effectively constant

constant_found = False
for region, grp in df.groupby(REGION_COL):
    stds = grp[FEATURE_COLS].std()
    bad  = stds[stds < CONST_THRESHOLD].index.tolist()
    if bad:
        print(f"  {region}: constant features → {bad}")
        constant_found = True

if not constant_found:
    print("  No perfectly constant features found.")


# ---------------------------------------------------------------------------
# 4. CHECK 3 – correlation of each feature with CHL_obs (per region)
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 3 – Feature–target Pearson correlations (|r| ≥ 0.95 flagged)")
print(DIVIDER2)

HIGH_CORR = 0.95    # threshold for flagging potential data leakage

for region, grp in df.groupby(REGION_COL):
    clean = grp[FEATURE_COLS + [OBS_COL]].dropna()
    if len(clean) < 3:
        print(f"  {region}: too few rows to compute correlations")
        continue
    corrs = clean[FEATURE_COLS].corrwith(clean[OBS_COL]).sort_values(
        key=abs, ascending=False
    )
    flagged = corrs[corrs.abs() >= HIGH_CORR]
    print(f"\n  {region}  (top 5 by |r| with {OBS_COL})")
    for feat, r in corrs.head(5).items():
        marker = "  ← LEAKAGE RISK (|r|≥0.95)" if abs(r) >= HIGH_CORR else ""
        print(f"    {feat:<25}  r = {r:+.4f}{marker}")
    if len(flagged) == 0:
        print("    → No feature exceeds |r|=0.95 threshold.")


# ---------------------------------------------------------------------------
# 5. CHECK 4 – n_samples vs n_features per region
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 4 – Sample count vs feature count per region")
print(DIVIDER2)

n_features = len(FEATURE_COLS)
print(f"  N features = {n_features}")
print()

for region, grp in df.groupby(REGION_COL):
    clean = grp[FEATURE_COLS + [OBS_COL]].dropna()
    n   = len(clean)
    ratio = n / n_features
    flag = ""
    if n <= n_features:
        flag = "  ← n_samples ≤ n_features  →  EXACT LR FIT GUARANTEED"
    elif ratio < 3:
        flag = "  ← very low n/p ratio  →  near-perfect fit likely"
    print(f"  {region:<14}  n={n:>4}  p={n_features}  n/p={ratio:.1f}{flag}")


# ---------------------------------------------------------------------------
# 6. CHECK 5 – Per-region LR fit with RMSE, MAE, R², max absolute error
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 5 – Per-region Linear Regression metrics")
print(DIVIDER2)

lr_summary = []

for region, grp in df.groupby(REGION_COL):
    clean = grp[FEATURE_COLS + [OBS_COL]].dropna().copy()
    n = len(clean)

    if n < 2:
        print(f"  {region}: insufficient rows after dropping NaN.")
        continue

    lr = LinearRegression()
    lr.fit(clean[FEATURE_COLS], clean[OBS_COL])
    y_pred = lr.predict(clean[FEATURE_COLS])
    y_true = clean[OBS_COL].values

    m = metrics(y_true.astype(float), y_pred.astype(float))
    lr_summary.append({"Region": region, **m})

    print(f"\n  {region}  (n={m['N']})")
    print(f"    RMSE        = {m['RMSE']:.8f}")
    print(f"    MAE         = {m['MAE']:.8f}")
    print(f"    R²          = {m['R2']:.8f}")
    print(f"    Max |error| = {m['MaxAbsErr']:.8f}")

    # Interpretation
    if m["R2"] >= 0.9999:
        if n <= n_features:
            print("    ⚠  PERFECT FIT: n_samples ≤ n_features → LR interpolates exactly.")
        else:
            print("    ⚠  PERFECT FIT: features are (near-)linear combination of target.")
            print("       Likely cause: lag/rolling CHL features encode the target directly.")
    elif m["R2"] >= 0.99:
        print("    ⚠  Near-perfect fit: likely data leakage in lag/rolling features.")


# ---------------------------------------------------------------------------
# 7. CHECK 6 – Scatter plots: observed vs predicted per region
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER2}")
print("CHECK 6 – Scatter plots (saved to disk)")
print(DIVIDER2)

fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5))
if len(regions) == 1:
    axes = [axes]

for ax, region in zip(axes, regions):
    grp = df[df[REGION_COL] == region].copy()
    clean = grp[FEATURE_COLS + [OBS_COL]].dropna()
    if len(clean) < 2:
        ax.set_title(f"{region}\n(insufficient data)")
        continue

    lr = LinearRegression().fit(clean[FEATURE_COLS], clean[OBS_COL])
    y_pred = lr.predict(clean[FEATURE_COLS])
    y_true = clean[OBS_COL].values

    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05

    ax.scatter(y_true, y_pred, alpha=0.6, s=20, color="steelblue", label="LR pred")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

    # Annotate with R²
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"R² = {r2:.6f}", transform=ax.transAxes,
            fontsize=9, color="darkred")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed CHL", fontsize=10)
    ax.set_ylabel("LR Predicted CHL", fontsize=10)
    ax.set_title(f"LR: {region.capitalize()}\n(n={len(y_true)}, p={n_features})", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", "box")

plt.suptitle("Linear Regression: Observed vs Predicted CHL\n"
             "(perfect diagonal = suspicious / data leakage)", fontsize=12, y=1.02)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "lr_obs_vs_pred.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Scatter plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# 8. Summary diagnosis
# ---------------------------------------------------------------------------
print(f"\n{DIVIDER}")
print("  DIAGNOSIS SUMMARY")
print(DIVIDER)
print("""
Most likely causes of perfect LR metrics, in order of probability:

1. n_samples ≤ n_features  (Check 4)
   → Linear regression has ≥ n degrees of freedom and can interpolate
     every point exactly.  With ~60 rows and 23 features this is a
     near-certain explanation.

2. Data leakage via lag/rolling CHL features  (Check 3)
   → 'chl_t-1', 'chl_7day_avg', 'chl_monthly_median', 'chl_anomaly'
     are all derived from CHL itself.  A linear model can trivially
     reconstruct CHL from these without any real predictive power.

3. Near-constant CHL values in a region  (Check 1)
   → Even a constant prediction achieves R²≈1 on a near-constant target.

Recommendation:
  • Use cross-validation (e.g. LeaveOneOut or TimeSeriesSplit) to get
    honest LR metrics instead of in-sample evaluation.
  • Consider removing direct CHL-derived features when comparing against
    the RF model to ensure a fair baseline.
""")

print(DIVIDER)
