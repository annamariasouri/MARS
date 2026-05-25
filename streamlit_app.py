"""
MARS dashboard — Streamlit host for the MARSdesign UI (web/).

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
      /* Full-bleed dashboard — hide Streamlit chrome */
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { display: none; }
      footer { visibility: hidden; }
      #MainMenu { visibility: hidden; }
      .stApp {
        background: #f6f8fa;
      }
      .block-container {
        padding: 0 !important;
        max-width: 100% !important;
      }
      section[data-testid="stMain"] > div {
        padding-top: 0 !important;
      }
      iframe {
        border: none !important;
        width: 100% !important;
        min-height: 100vh !important;
        height: 100vh !important;
      }
      div[data-testid="stHtml"] {
        width: 100%;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def _dashboard_html(_v: int = 5) -> str:
    """_v bumps cache when dashboard data wiring changes."""
    return build_html(WEB, build_payload())


try:
    html = _dashboard_html()
except Exception as exc:
    st.error("Could not load MARS dashboard data from the repository CSVs.")
    st.exception(exc)
    st.stop()

components.html(html, height=900, scrolling=True)
