"""
MARS dashboard — Streamlit host for the Claude design (web/).

Deploy online: https://share.streamlit.io → this file as main script.
Run locally:  streamlit run streamlit_app.py --server.port 8765
             or: .\\dev.ps1
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / ".github" / "scripts"
WEB = ROOT / "web"
sys.path.insert(0, str(SCRIPTS))

from export_dashboard_data import build_payload  # noqa: E402
from mars_dashboard_html import build_html  # noqa: E402

st.set_page_config(
    page_title="MARS — Marine Autonomous Risk System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { display: none; }
      .block-container {
        padding: 0 !important;
        max-width: 100% !important;
      }
      iframe { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def _dashboard_html() -> str:
    return build_html(WEB, build_payload())


try:
    html = _dashboard_html()
except Exception as exc:
    st.error("Could not load MARS dashboard data from the repository CSVs.")
    st.exception(exc)
    st.stop()

# st.html() does not accept scrolling= (causes TypeError on Streamlit Cloud).
components.html(html, height=900, scrolling=True)
