import os
import re
from glob import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

# === CONFIG ===
st.set_page_config(page_title="MARS – Marine Autonomous Risk System", page_icon="surprise-favicon-32x32.png", layout="wide")

# Clear any cached data to avoid stale reads when switching regions
try:
    st.cache_data.clear()
except Exception:
    pass

# --- Theme ---
PRIMARY_DARK = "#062B4F"
PRIMARY = "#0B4F6C"
PRIMARY_GRAD_1 = "#0072BC"
PRIMARY_GRAD_2 = "#00B4D8"
ACCENT = "#34D1BF"
AMBER = "#FFB703"
RED = "#D00000"
GREEN = "#2A9D8F"
MUTED = "#6B7A90"
PLOTLY_TEMPLATE = "plotly_white"

REGIONS = {
    "thermaikos": {"title": "Thermaikos (Greece)", "bbox": (40.2, 40.7, 22.5, 23.0), "color": "#2E86DE"},
    "peiraeus": {"title": "Piraeus (Greece)", "bbox": (37.9, 38.1, 23.5, 23.8), "color": "#E17055"},
    "limassol": {"title": "Limassol (Cyprus)", "bbox": (34.6, 34.8, 33.0, 33.2), "color": "#00B894"},
}

ENV_VARS = [
    ("CHL", "Chlorophyll-a (mg/m³)"),
    ("NH4", "Ammonium NH₄ (µmol/L)"),
    ("NO3", "Nitrate NO₃ (µmol/L)"),
    ("PO4", "Phosphate PO₄ (µmol/L)"),
    ("THETAO", "Temperature θ (°C)"),
    ("SO", "Salinity (PSU)"),
]

# === THEME CSS (escaped braces) ===
st.markdown(
    f"""
    <style>
      :root {{
        --grad1: {PRIMARY_GRAD_1};
        --grad2: {PRIMARY_GRAD_2};
      }}
      body {{ background: linear-gradient(180deg,#00111f 0%,#001b33 40%,#001f3f 100%) fixed; color:#E0F2FF; }}
      .marine-hero {{
        background: linear-gradient(90deg, var(--grad1), var(--grad2));
        color: white; padding: 18px 22px; border-radius: 16px; box-shadow: 0 10px 28px rgba(0,0,0,.25);
      }}
      .kpi {{
        background: #ffffff; border: 1px solid rgba(0,0,0,.06); border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 6px 20px rgba(13, 51, 89, .20); text-align:center;
      }}
      .kpi .label {{ color: #6B7A90; font-size: 13px; letter-spacing: .2px; font-weight:600; }}
      .kpi .value {{ font-size: 26px; font-weight: 800; color: #000; }}
      .badge {{ display:inline-block; padding:6px 12px; border-radius:999px; font-weight:700; font-size:14px;}}
      .badge.low {{ background: rgba(42,157,143,.15); color:#005f4b; border:1px solid rgba(42,157,143,.35); }}
      .badge.med {{ background: rgba(255,183,3,.15); color:#7c5700; border:1px solid rgba(255,183,3,.4); }}
      .badge.high {{ background: rgba(208,0,0,.15); color:#750000; border:1px solid rgba(208,0,0,.35); }}
      .section-title {{ color: #fff; font-weight:800; }}
    /* map-specific title: black text */
    .map-section-title {{ color: #000; font-weight:800; }}
      .soft-card {{ background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:14px;box-shadow:0 8px 24px rgba(0,0,0,.25);}}
      iframe, .folium-map {{ border-radius: 16px; box-shadow: 0 0 40px rgba(0,0,0,0.6); }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### 🌊 MARS – Marine Autonomous Risk System")
    st.write("Part of **Annamaria Souri**’s PhD research • Powered by **Copernicus Marine**")

# Data dir (prefer a top-level `data/` folder, overridable via MARS_DATA_DIR)
DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")

# --- Hero Header ---
st.markdown(
    """
    <div class="marine-hero">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-size:28px;">🛰️</div>
        <div>
          <div style="font-size:22px;font-weight:800;letter-spacing:.3px;">MARS Dashboard</div>
          <div style="opacity:.9">Real‑Time Bloom Forecasts for the Eastern Mediterranean</div>
        </div>
        <div style="margin-left:auto;opacity:.9;">Updated daily</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# === HELPERS ===

def list_files():
    out = []
    try:
        # top-level files
        out.extend(sorted(os.listdir(DATA_DIR)))
    except Exception:
        pass
    # also include copernicus/env_history and forecasts subfolders if present
    for sub in ("copernicus/env_history", "forecasts"):
        p = os.path.join(DATA_DIR, sub)
        try:
            if os.path.exists(p):
                out.extend([os.path.join(sub, f) for f in sorted(os.listdir(p))])
        except Exception:
            continue
    return out


def latest_env_file(region: str) -> str | None:
    # Find both dated and undated env_history files. Search data/copernicus/env_history/.
    search_dirs = [os.path.join(DATA_DIR, "copernicus", "env_history")]
    dated = []
    undated_candidates = []
    for d in search_dirs:
        dated.extend(glob(os.path.join(d, f"env_history_{region}_*.csv")))
        undated_candidates.append(os.path.join(d, f"env_history_{region}.csv"))

    undated = None
    for u in undated_candidates:
        if os.path.exists(u):
            undated = u
            break

    # Helper: quick validity check for a CSV file (has >0 rows and a time-like column)
    def is_valid_env_file(path: str) -> bool:
        try:
            if not os.path.exists(path) or os.path.getsize(path) < 40:
                return False
            sample = pd.read_csv(path, nrows=5)
            if sample.empty:
                return False
            # look for any datetime-like column
            for c in sample.columns:
                try:
                    parsed = pd.to_datetime(sample[c], errors="coerce")
                    if parsed.notna().sum() >= 1:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    # Prefer the undated full-history file if it exists and looks valid
    if undated and is_valid_env_file(undated):
        return undated

    # Otherwise consider dated files; filter valid ones
    valid_dated = [p for p in dated if is_valid_env_file(p)]
    if valid_dated:
        # pick most recently modified valid dated file
        valid_dated.sort(key=lambda p: os.path.getmtime(p))
        return valid_dated[-1]

    # Fallback: if undated exists (even if not ideal) return it, else return newest dated file if any
    if undated:
        return undated
    if dated:
        dated.sort(key=lambda p: os.path.getmtime(p))
        return dated[-1]

    return None


def load_forecast(region: str) -> pd.DataFrame:
    # Look in data/forecasts/ for forecast files
    search_dirs = [os.path.join(DATA_DIR, "forecasts")]
    for name in [f"forecast_log_{region}.csv", f"forecast_{region}.csv"]:
        for d in search_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    st.warning(f"Could not read forecast file for {region}: {e}")
                    return pd.DataFrame()
                df.columns = [c.strip().lower() for c in df.columns]
                # parse date
                for c in ["date", "day", "ds", "timestamp"]:
                    if c in df.columns:
                        df["date"] = pd.to_datetime(df[c], errors="coerce")
                        break
                # normalize possible alternative names
                df = df.rename(columns={
                    "bloom_flag": "bloom_risk_flag",
                    "risk_flag": "bloom_risk_flag",
                    "chl_pred": "predicted_chl",
                    "threshold": "threshold_used",
                })
                # If all values are NaN or empty, treat as missing
                if df.empty or df.isna().all(axis=None):
                    st.info(f"No forecast data available for {region}.")
                    return pd.DataFrame()
                return df.sort_values("date").reset_index(drop=True)
    st.info(f"No forecast file found for {region}.")
    return pd.DataFrame()


def load_env(region: str) -> pd.DataFrame:
    f = latest_env_file(region)
    if not f:
        return pd.DataFrame()
    df = pd.read_csv(f)
    # normalize headings – keep original too for safety
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # unify time column
    time_candidates = [c for c in df.columns if c.lower() in ("time", "date", "datetime", "ts")]
    if time_candidates:
        tcol = time_candidates[0]
        df = df.rename(columns={tcol: "TIME"})
    else:
        # attempt to find any datetime-like column
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > max(1, len(df)//4):
                    df.insert(0, "TIME", parsed)
                    break
            except Exception:
                continue

    # unify variable names (case-insensitive)
    rename_map = {}
    for c in list(df.columns):
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

    return df


def plot_ts(df: pd.DataFrame, x: str, y: str, title: str, ylab: str):
    fig = px.line(df, x=x, y=y, title=title, template=PLOTLY_TEMPLATE,
                  color_discrete_sequence=[PRIMARY_GRAD_1])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=ylab)
    return fig


def summarize_region(forecast: pd.DataFrame) -> dict:
    out = {"latest_chl": None, "threshold": None,
        "rec7": None, "rec30": None, "risk7": None, "risk30": None, "risk_score": None}
    if forecast.empty:
        return out
    last = forecast.dropna(subset=["date"]).iloc[-1] if "date" in forecast.columns else forecast.iloc[-1]
    out["latest_chl"] = last.get("predicted_chl")
    out["threshold"] = last.get("threshold_used")
    # Removed bloom_flag logic
    out["risk_score"] = last.get("risk_score") if "risk_score" in forecast.columns else None

    # risk flags
    if "bloom_risk_flag" in forecast.columns:
        flags = forecast["bloom_risk_flag"].astype(str).str.lower().isin(["1","true","yes"]) 
    elif {"predicted_chl","threshold_used"}.issubset(forecast.columns):
        flags = forecast["predicted_chl"] >= forecast["threshold_used"]
    else:
        flags = pd.Series([False]*len(forecast), index=forecast.index)

    # windows
    if "date" in forecast.columns:
        fc = forecast.dropna(subset=["date"]).copy(); fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
        fc = fc.dropna(subset=["date"]).reset_index(drop=True)
        if fc.empty:
            return out
        end = fc["date"].max()
        idx7 = fc.index[fc["date"] >= end - timedelta(days=7)]
        idx30 = fc.index[fc["date"] >= end - timedelta(days=30)]
        out["risk7"] = int(flags.loc[idx7].sum()) if len(idx7) else 0
        out["risk30"] = int(flags.loc[idx30].sum()) if len(idx30) else 0
        out["rec7"] = float(last.get("recurrence_7d_prob")) if "recurrence_7d_prob" in forecast.columns and pd.notna(last.get("recurrence_7d_prob")) else (round(flags.loc[idx7].mean()*100,1) if len(idx7) else None)
        out["rec30"] = float(last.get("recurrence_30d_prob")) if "recurrence_30d_prob" in forecast.columns and pd.notna(last.get("recurrence_30d_prob")) else (round(flags.loc[idx30].mean()*100,1) if len(idx30) else None)
    else:
        out["risk7"], out["risk30"] = int(flags.tail(7).sum()), int(flags.tail(30).sum())
        out["rec7"], out["rec30"] = round(flags.tail(7).mean()*100,1), round(flags.tail(30).mean()*100,1)

    return out

# === MAP ===
all_lat, all_lon = [], []
for v in REGIONS.values():
    lat_min, lat_max, lon_min, lon_max = v["bbox"]
    all_lat.extend([lat_min, lat_max])
    all_lon.extend([lon_min, lon_max])

if all_lat and all_lon:
    # Adjust center slightly east to better show Cyprus
    center = [37.0, 28.0]  # moved east and slightly south to center between Greece and Cyprus
else:
    center = [37.0, 28.0]

def _has_forecast_for(region: str) -> bool:
    # check both root and forecasts/ folder
    for d in (DATA_DIR, os.path.join(DATA_DIR, "forecasts")):
        if os.path.exists(os.path.join(d, f"forecast_log_{region}.csv")) or os.path.exists(os.path.join(d, f"forecast_{region}.csv")):
            return True
    return False

available = [r for r in REGIONS if _has_forecast_for(r)]
if "region" not in st.session_state:
    st.session_state.region = available[0] if available else "thermaikos"

head_cols = st.columns([3,1])
with head_cols[0]:
    st.markdown("<div class='section-title map-section-title' style='margin:16px 0 6px;'>📍 Regions Map (click to select)</div>", unsafe_allow_html=True)
with head_cols[1]:
    st.selectbox("Active region", options=list(REGIONS.keys()), format_func=lambda k: REGIONS[k]["title"], key="region")


# Make map stretch full width
map_col = st.columns(1)[0]
with map_col:
    m = folium.Map(location=center, zoom_start=6, tiles="cartodbpositron")  # zoomed out to show Cyprus
    
    # Add custom CSS for marker styling
    css = """
    <style>
        .leaflet-interactive {
            cursor: pointer !important;
        }
        .custom-marker-icon {
            color: white;
            text-shadow: 0 0 3px rgba(0,0,0,0.4);
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(css))
    
    for k, v in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = v["bbox"]
        is_active = k == st.session_state.region
        # Calculate center point for the marker
        marker_lat = (lat_min + lat_max) / 2
        marker_lon = (lon_min + lon_max) / 2
        
        # Create a custom icon for each region (pin only, no label)
        icon = folium.DivIcon(
            html=f'''
                <div class="custom-marker-icon" style="text-align:center;">
                    <div style="font-size:28px;color:{v['color']};">📍</div>
                </div>
            ''',
            icon_size=(40, 40),
            icon_anchor=(20, 40),
        )
        
        # Add marker
        folium.Marker(
            location=[marker_lat, marker_lon],
            icon=icon,
            popup=v["title"],
            tooltip=v["title"]
        ).add_to(m)
        
        # Add a subtle region outline
        folium.Rectangle(
            bounds=[[lat_min, lon_min],[lat_max, lon_max]],
            color=v["color"],
            weight=1,
            fill=True,
            fill_opacity=0.1 if not is_active else 0.2,
            popup=None,
            opacity=0.5
        ).add_to(m)
    mret = st_folium(m, height=600, width="100%", key="mars_map")

if mret and mret.get("last_clicked"):
    clat = mret["last_clicked"]["lat"]
    clon = mret["last_clicked"]["lng"]
    for k, v in REGIONS.items():
        lat_min, lat_max, lon_min, lon_max = v["bbox"]
        if lat_min <= clat <= lat_max and lon_min <= clon <= lon_max:
            st.session_state.region = k
            break

region = st.session_state.region
forecast = load_forecast(region)
env = load_env(region)
region_title = REGIONS[region]["title"]
summary = summarize_region(forecast)

# === KPI helpers ===

def fmt_val(val, prec: int = 3, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return f"{float(val):.{prec}f}{suffix}"
    except Exception:
        return str(val)


def likelihood_badge(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and pd.isna(pct)):
        return "<span class='badge'>—</span>"
    if pct <= 20: cls, label = "low", "Low"
    elif pct <= 60: cls, label = "med", "Moderate"
    else: cls, label = "high", "High"
    return f"<span class='badge {cls}'>{label} ({pct:.1f}%)</span>"

# === KPI CARDS ===
k1, k2, k3 = st.columns([2,2,3])
with k1:
        st.markdown(f"""
        <div class='kpi'>
            <div class='label'>{region_title} – CHL</div>
            <div class='value'>{fmt_val(summary['latest_chl'], 3, ' mg/m³')}</div>
        </div>""", unsafe_allow_html=True)
        with st.expander("How is this predicted?", expanded=False):
            st.markdown("""
            The model uses recent environmental data (nutrients, temperature, salinity, and past chlorophyll levels) to estimate the chlorophyll concentration for each grid point in the region. It is trained on historical data to learn patterns that indicate bloom risk.
            """)
with k2:
        st.markdown(f"""
        <div class='kpi'>
            <div class='label'>Continuous Risk (Today)</div>
            <div class='value'>{likelihood_badge(summary['risk_score'])}</div>
        </div>""", unsafe_allow_html=True)
        with st.expander("What does this mean?", expanded=False):
            st.markdown("""
            The risk score shows how close the average predicted chlorophyll level is to the bloom threshold for today.  
            **100** means the average is at or above the threshold (high risk), **0** means far below (low risk).  
            Values in between indicate increasing likelihood of a bloom event.
            """)
with k3:
        st.markdown(f"""
        <div class='kpi'>
            <div class='label'>Threshold Used</div>
            <div class='value'>{fmt_val(summary['threshold'], 3)}</div>
        </div>""", unsafe_allow_html=True)
        with st.expander("What is the threshold?", expanded=False):
            st.markdown("""
            **Threshold Used:**
            This is the chlorophyll concentration value (mg/m³) used as the threshold for bloom risk. Predictions above this value indicate higher risk of a bloom event.
            
            **How is the threshold calculated?**
            The threshold is dynamic: it is recalculated for every region and day as the 90th percentile of the predicted chlorophyll values. This means it adapts to the distribution of predicted concentrations for each forecast, marking the value above which only the highest 10% of predicted concentrations fall.
            """)

k4, k5, k6 = st.columns([3,3,2])
with k4:
    st.markdown(f"""<div class='kpi'>
        <div class='label'>Likelihood (Next 7 d)</div>
        <div style='margin-top:8px;margin-bottom:4px'>{likelihood_badge(summary['rec7'])}</div>
    </div>""", unsafe_allow_html=True)
    with st.expander("How is this likelihood calculated?", expanded=False):
        st.markdown("""
        The likelihood (Next 7 d) is shown as a percentage and represents the model's estimate of how likely it is that a bloom event will occur in the next 7 days.
        
        **How is it calculated?**
        For each day in the next 7 days, the model predicts the chlorophyll concentration for the region. If the predicted value is at or above the dynamic threshold, that day is counted as a 'risk day'. The percentage is calculated as:
        
        `Likelihood (%) = (Number of risk days in next 7) / 7 × 100`
        
        For example, if 3 out of 7 days are predicted to be above the threshold, the likelihood is 43%.
        
        **Why is today's risk not always a percentage?**
        Today's risk is shown as a continuous score (0–100) based on how close the average predicted chlorophyll is to the threshold, while the next 7/30 days likelihood is based on the count of days above threshold.
        """)
with k5:
    st.markdown(f"""<div class='kpi'>
        <div class='label'>Likelihood (Next 30 d)</div>
        <div style='margin-top:8px;margin-bottom:4px'>{likelihood_badge(summary['rec30'])}</div>
    </div>""", unsafe_allow_html=True)
    with st.expander("How is this likelihood calculated?", expanded=False):
        st.markdown("""
        The likelihood (Next 30 d) is shown as a percentage and represents the model's estimate of how likely it is that a bloom event will occur in the next 30 days.
        
        **How is it calculated?**
        For each day in the next 30 days, the model predicts the chlorophyll concentration for the region. If the predicted value is at or above the dynamic threshold, that day is counted as a 'risk day'. The percentage is calculated as:
        
        `Likelihood (%) = (Number of risk days in next 30) / 30 × 100`
        
        For example, if 9 out of 30 days are predicted to be above the threshold, the likelihood is 30%.
        
        **Why is today's risk not always a percentage?**
        Today's risk is shown as a continuous score (0–100) based on how close the average predicted chlorophyll is to the threshold, while the next 7/30 days likelihood is based on the count of days above threshold.
        """)
with k6:
    r7 = summary['risk7'] if summary['risk7'] is not None else 0
    r30 = summary['risk30'] if summary['risk30'] is not None else 0
    st.markdown(f"<div class='kpi'><div class='label'>Risk Days (7/30)</div><div class='value'>{r7}/{r30}</div></div>", unsafe_allow_html=True)

# === TABS ===

tab1, tab2, tab3 = st.tabs(["Today’s Forecast", "Environmental Trends", "Accuracy"])

with tab1:
    st.markdown("<div class='section-title'>CHL Forecasts</div>", unsafe_allow_html=True)
    if not env.empty and "TIME" in env.columns and "CHL" in env.columns:
        env = env.copy(); env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce"); env = env.dropna(subset=["TIME"])
        now = env["TIME"].max()
        last7 = env[env["TIME"] >= now - timedelta(days=7)]
        last30 = env[env["TIME"] >= now - timedelta(days=30)]
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_ts(last7, "TIME", "CHL", "CHL – Last 7 days", "mg/m³"), use_container_width=True)
        with c2: st.plotly_chart(plot_ts(last30, "TIME", "CHL", "CHL – Last 30 days", "mg/m³"), use_container_width=True)
    elif not forecast.empty and {"date","predicted_chl"}.issubset(forecast.columns):
        st.info("Using predicted CHL from forecast history (env history not found).")
        st.plotly_chart(px.line(forecast.tail(30), x="date", y="predicted_chl", title="Predicted CHL (last 30 days)",
                                color_discrete_sequence=[PRIMARY_GRAD_2], template=PLOTLY_TEMPLATE), use_container_width=True)
    else:
        st.info("No environmental or forecast CHL series available yet.")

with tab2:
    st.markdown(f"<div class='section-title'>{region_title} – Environmental Trends (30 days)</div>", unsafe_allow_html=True)
    if env.empty or "TIME" not in env.columns:
        st.info("No env_history file with a TIME column found for this region yet.")
    else:
        env = env.copy(); env["TIME"] = pd.to_datetime(env["TIME"], errors="coerce")
        variables = [v for v, _ in ENV_VARS if v in env.columns]
        if not variables:
            st.info("No known environmental variables present.")
        else:
            chosen = st.multiselect("Variables to plot", variables, default=variables[:2])
            for v in chosen:
                label = dict(ENV_VARS).get(v, v)
                st.plotly_chart(plot_ts(env, "TIME", v, label, label), use_container_width=True)

    with tab3:
        st.markdown(f"<div class='section-title'>{region_title} – Forecast Accuracy</div>", unsafe_allow_html=True)
        # look for eval file in DATA_DIR/eval or data/eval
        eval_paths = [os.path.join(DATA_DIR, 'eval', f'accuracy_{region}.csv'), os.path.join(DATA_DIR, f'accuracy_{region}.csv')]
        eval_path = next((p for p in eval_paths if os.path.exists(p)), None)
        if not eval_path:
            st.info(f"No evaluation file found for {region}. Run the evaluation script: python scripts/evaluate_forecasts.py --region {region}")
        else:
            try:
                eval_df = pd.read_csv(eval_path, parse_dates=['target_date', 'forecast_date'])
            except Exception as e:
                st.warning(f"Could not read evaluation file: {e}")
                eval_df = pd.DataFrame()

            if eval_df.empty:
                st.info("No evaluated rows available for this region yet.")
            else:
                # location selector: region average or specific lat/lon
                eval_df['lat_str'] = eval_df['lat'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '')
                eval_df['lon_str'] = eval_df['lon'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '')
                eval_df['loc_key'] = eval_df.apply(lambda r: (f"{r['lat_str']},{r['lon_str']}" if r['lat_str'] and r['lon_str'] else 'region average'), axis=1)
                locs = ['region average'] + sorted([k for k in eval_df['loc_key'].unique() if k != 'region average'])
                sel_loc = st.selectbox('Location (region average or specific grid point)', options=locs)

                if sel_loc == 'region average':
                    ts = eval_df.groupby(eval_df['target_date'].dt.normalize(), as_index=False).agg({'predicted_chl':'mean','observed_chl':'mean'})
                else:
                    lat_s, lon_s = sel_loc.split(',')
                    sub = eval_df[(eval_df['lat_str']==lat_s)&(eval_df['lon_str']==lon_s)].copy()
                    ts = sub.groupby(sub['target_date'].dt.normalize(), as_index=False).agg({'predicted_chl':'mean','observed_chl':'mean'})

                # Time series overlay
                if not ts.empty:
                    fig_ts = px.line(ts, x='target_date', y=['predicted_chl','observed_chl'], labels={'value':'CHL (mg/m³)','target_date':'Date'}, template=PLOTLY_TEMPLATE)
                    fig_ts.update_layout(title='Predicted vs Observed CHL', legend_title_text='Series')
                    st.plotly_chart(fig_ts, use_container_width=True)

                # Scatter plot
                scatter_df = eval_df if sel_loc=='region average' else eval_df[(eval_df['lat_str']==lat_s)&(eval_df['lon_str']==lon_s)]
                if not scatter_df.empty:
                    scatter_df = scatter_df.dropna(subset=['predicted_chl','observed_chl'])
                    fig_sc = px.scatter(scatter_df, x='predicted_chl', y='observed_chl', trendline='ols', template=PLOTLY_TEMPLATE)
                    fig_sc.add_shape(type='line', x0=0, x1=max(scatter_df['predicted_chl'].max(), scatter_df['observed_chl'].max()), y0=0, y1=max(scatter_df['predicted_chl'].max(), scatter_df['observed_chl'].max()), line=dict(dash='dash'))
                    fig_sc.update_layout(title='Predicted vs Observed (scatter)')
                    st.plotly_chart(fig_sc, use_container_width=True)

                # Simple table: predicted | actual | accuracy percent
                show_n = st.slider('Rows to show (most recent)', min_value=5, max_value=200, value=20)
                table_df = eval_df.sort_values('target_date', ascending=False).head(show_n)
                # compute simple accuracy: if bloom flags exist use them, else percentage within tolerance
                if {'bloom_pred','bloom_obs'}.issubset(table_df.columns):
                    tp = ((table_df['bloom_pred']==True)&(table_df['bloom_obs']==True)).sum()
                    tn = ((table_df['bloom_pred']==False)&(table_df['bloom_obs']==False)).sum()
                    total = len(table_df.dropna(subset=['bloom_pred','bloom_obs']))
                    acc_pct = (tp+tn)/total*100 if total>0 else None
                    st.metric('Bloom flag accuracy (%)', f"{acc_pct:.1f}%" if acc_pct is not None else 'N/A')
                else:
                    # tolerance-based
                    tol = st.number_input('Tolerance for "correct" (mg/m³)', value=0.2, step=0.05)
                    ok = (table_df['abs_err'] <= tol).sum()
                    total = len(table_df.dropna(subset=['abs_err']))
                    acc_pct = ok/total*100 if total>0 else None
                    st.metric('Accuracy within tolerance (%)', f"{acc_pct:.1f}%" if acc_pct is not None else 'N/A')

                # show table
                display_cols = ['target_date','lat','lon','predicted_chl','observed_chl','err','abs_err']
                display = table_df[[c for c in display_cols if c in table_df.columns]].copy()
                display['target_date'] = pd.to_datetime(display['target_date']).dt.date
                st.dataframe(display)


# --- Diagnostics ---
st.sidebar.markdown("""
MARS – Marine Autonomous Risk System forecasts harmful algal bloom (red tide) risk in the Eastern Mediterranean using daily Copernicus Marine data and a trained machine‑learning model.

Regions: Thermaikos (GR), Piraeus (GR), Limassol (CY).
Variables: NH₄, NO₃, PO₄, θ (temperature), SO (salinity), CHL.
""")
with st.expander("🔍 Diagnostics"):
    st.write("Working directory:", DATA_DIR)
    st.write("Files:", list_files())
    st.write("Forecast columns:", list(forecast.columns))
    # show which env_history file was used (if any)
    env_file = latest_env_file(region)
    st.write("Env file used:", env_file)
    st.write("Env columns:", list(env.columns))

st.markdown(
    f"<hr style='margin-top:2em;border:0;height:2px;background:linear-gradient(90deg,{PRIMARY_GRAD_1},{PRIMARY_GRAD_2});opacity:.8;'>"
    f"<div style='text-align:center;color:{MUTED};font-size:12px;'>© {datetime.now().year} MARS • Research prototype</div>",
    unsafe_allow_html=True,
)