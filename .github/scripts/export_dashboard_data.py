"""
Export MARS CSV data to web/data.js for the static React dashboard.

Run from repo root:
  python .github/scripts/export_dashboard_data.py
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from glob import glob

import pandas as pd

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.abspath(os.path.join(REPO_ROOT, DATA_DIR))
WEB_DIR = os.path.join(REPO_ROOT, "web")
OUT_PATH = os.path.join(WEB_DIR, "data.js")

REGIONS = ("thermaikos", "peiraeus", "limassol")
REGION_TITLES = {
    "thermaikos": "Thessaloniki",
    "peiraeus": "Piraeus",
    "limassol": "Limassol",
}


def _json_num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    if df.empty:
        return []
    out = []
    for row in df.itertuples(index=False):
        rec = {}
        for c, v in zip(cols, row):
            rec[c] = _json_num(v)
        out.append(rec)
    return out


def load_forecast(region: str) -> pd.DataFrame:
    for name in (f"forecast_log_{region}.csv", f"forecast_{region}.csv"):
        path = os.path.join(DATA_DIR, "forecasts", name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in ("date", "day", "ds", "timestamp"):
            if c in df.columns:
                df["date"] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
                break
        df = df.rename(
            columns={
                "threshold_used": "threshold",
                "bloom_risk_flag": "flag",
            }
        )
        keep = ["date", "predicted_chl", "threshold", "risk_score", "risk_pct", "flag"]
        for col in keep:
            if col not in df.columns:
                df[col] = None
        df = df[keep].dropna(subset=["date"]).sort_values("date")
        return df.reset_index(drop=True)
    return pd.DataFrame()


def load_env(region: str) -> pd.DataFrame:
    env_dir = os.path.join(DATA_DIR, "copernicus", "env_history")
    pattern = os.path.join(env_dir, f"env_history_{region}_*.csv")
    files = sorted(glob(pattern))
    undated = os.path.join(env_dir, f"env_history_{region}.csv")
    if os.path.exists(undated):
        files.append(undated)

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        cols = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=cols)
        tcol = next((c for c in df.columns if c in ("time", "date", "datetime", "ts")), None)
        if not tcol:
            continue
        df["date"] = pd.to_datetime(df[tcol], errors="coerce").dt.strftime("%Y-%m-%d")
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ("chl", "chlorophyll"):
                rename[c] = "chl"
            elif cl in ("thetao", "theta", "temp", "temperature"):
                rename[c] = "thetao"
            elif cl == "so":
                rename[c] = "so"
            elif cl == "nh4":
                rename[c] = "nh4"
            elif cl == "no3":
                rename[c] = "no3"
            elif cl == "po4":
                rename[c] = "po4"
        df = df.rename(columns=rename)
        keep = ["date", "chl", "nh4", "no3", "po4", "thetao", "so"]
        for col in keep:
            if col not in df.columns:
                df[col] = None
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return merged.sort_values("date").reset_index(drop=True)


def _env_chl_daily(region: str) -> pd.DataFrame:
    """Daily median CHL from Copernicus env-history (same source as Environmental Trends)."""
    env = load_env(region)
    if env.empty or "chl" not in env.columns:
        return pd.DataFrame()
    daily = (
        env.dropna(subset=["date", "chl"])
        .groupby("date", as_index=False)["chl"]
        .median()
        .rename(columns={"date": "target_date", "chl": "env_chl"})
    )
    return daily


def load_accuracy(region: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "evaluation", f"accuracy_{region}.csv")
    df = pd.DataFrame()
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "target_date" in df.columns:
            df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
        # One row per target day (median if multiple grid points)
        if "target_date" in df.columns and len(df) > 0:
            agg = {
                c: "median"
                for c in ("predicted_chl", "observed_chl", "err", "abs_err")
                if c in df.columns
            }
            if agg:
                df = df.groupby("target_date", as_index=False).agg(agg)

    if df.empty:
        fc = load_forecast(region)
        if fc.empty:
            return pd.DataFrame()
        df = fc.rename(columns={"date": "target_date"})[["target_date", "predicted_chl"]].copy()
        df["observed_chl"] = None
        df["err"] = None
        df["abs_err"] = None

    keep = ["target_date", "predicted_chl", "observed_chl", "err", "abs_err"]
    for col in keep:
        if col not in df.columns:
            df[col] = None

    return enrich_accuracy_with_env(region, df[keep])


def enrich_accuracy_with_env(region: str, acc: pd.DataFrame) -> pd.DataFrame:
    """Fill missing observed_chl from env-history CHL so accuracy charts are complete."""
    env_daily = _env_chl_daily(region)
    if env_daily.empty:
        return acc.sort_values("target_date").reset_index(drop=True)

    out = acc.merge(env_daily, on="target_date", how="left")
    missing = out["observed_chl"].isna() & out["env_chl"].notna()
    out.loc[missing, "observed_chl"] = out.loc[missing, "env_chl"]

    has_both = out["predicted_chl"].notna() & out["observed_chl"].notna()
    out.loc[has_both, "err"] = out.loc[has_both, "predicted_chl"] - out.loc[has_both, "observed_chl"]
    out.loc[has_both, "abs_err"] = out.loc[has_both, "err"].abs()

    return (
        out.drop(columns=["env_chl"])
        .sort_values("target_date")
        .reset_index(drop=True)
    )


def load_validation_table() -> list[dict]:
    path = os.path.join(DATA_DIR, "retrain", "validation_metrics_retrained.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        region_key = str(r.get("region", "")).strip().lower()
        if region_key == "overall":
            continue
        rows.append(
            {
                "region": REGION_TITLES.get(region_key, region_key.title()),
                "n": int(r["n"]) if pd.notna(r.get("n")) else None,
                "rmse": float(r["rmse"]) if pd.notna(r.get("rmse")) else None,
                "mae": None,
                "r2": float(r["r2"]) if pd.notna(r.get("r2")) else None,
            }
        )
    overall = df[df["region"].astype(str).str.upper() == "OVERALL"]
    if not overall.empty:
        r = overall.iloc[0]
        rows.append(
            {
                "region": "Overall",
                "n": int(r["n"]),
                "rmse": float(r["rmse"]),
                "mae": None,
                "r2": float(r["r2"]),
                "is_total": True,
            }
        )
    return rows


def load_bloom_table() -> list[dict]:
    path = os.path.join(DATA_DIR, "evaluation", "bloom_metrics_2024_2025.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        key = str(r.get("region", "")).strip().lower()
        rows.append(
            {
                "region": REGION_TITLES.get(key, key.title()),
                "threshold": float(r["threshold_pred"])
                if pd.notna(r.get("threshold_pred"))
                else None,
                "prevalence": None,
                "precision": float(r["precision"]) if pd.notna(r.get("precision")) else 0,
                "recall": float(r["recall"]) if pd.notna(r.get("recall")) else 0,
                "f1": float(r["f1"]) if pd.notna(r.get("f1")) else 0,
            }
        )
    return rows


def build_payload() -> dict:
    mars_data = {}
    all_forecast_dates = []

    for region in REGIONS:
        fc = load_forecast(region)
        env = load_env(region)
        acc = load_accuracy(region)
        mars_data[region] = {
            "forecast": _records(
                fc,
                ["date", "predicted_chl", "threshold", "risk_score", "risk_pct", "flag"],
            ),
            "env": _records(env, ["date", "chl", "nh4", "no3", "po4", "thetao", "so"]),
            "accuracy": _records(
                acc, ["target_date", "predicted_chl", "observed_chl", "err", "abs_err"]
            ),
        }
        if not fc.empty:
            all_forecast_dates.extend(fc["date"].tolist())

    coverage_start = min(all_forecast_dates) if all_forecast_dates else None
    coverage_end = max(all_forecast_dates) if all_forecast_dates else None
    forecast_days = (
        len(set(all_forecast_dates)) if all_forecast_dates else 0
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_dir": DATA_DIR,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "forecast_days": forecast_days,
        "port_count": len(REGIONS),
        "basin_count": len(REGIONS),
        "validation": load_validation_table(),
        "bloom_metrics": load_bloom_table(),
    }
    return {"MARS_DATA": mars_data, "MARS_META": meta}


def main() -> None:
    os.makedirs(WEB_DIR, exist_ok=True)
    payload = build_payload()
    body = json.dumps(payload["MARS_DATA"], indent=1, allow_nan=False)
    meta = json.dumps(payload["MARS_META"], indent=1, allow_nan=False)
    content = (
        "// Auto-generated by .github/scripts/export_dashboard_data.py — do not edit by hand.\n"
        f"window.MARS_DATA = {body};\n"
        f"window.MARS_META = {meta};\n"
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {os.path.relpath(OUT_PATH, REPO_ROOT)}")
    print(
        f"  Coverage: {payload['MARS_META'].get('coverage_start')} -> "
        f"{payload['MARS_META'].get('coverage_end')} "
        f"({payload['MARS_META'].get('forecast_days')} forecast days)"
    )


if __name__ == "__main__":
    main()
