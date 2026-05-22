"""Build a self-contained HTML page for the MARS dashboard (Streamlit or static)."""
from __future__ import annotations

import json
import re
from pathlib import Path

WEB_JSX = (
    "tweaks-panel.jsx",
    "charts.jsx",
    "map.jsx",
    "logo.jsx",
    "app.jsx",
)


def build_html(web_dir: Path, payload: dict) -> str:
    web_dir = Path(web_dir)
    index = (web_dir / "index.html").read_text(encoding="utf-8")
    css = (web_dir / "styles.css").read_text(encoding="utf-8")

    index = index.replace(
        '<link rel="stylesheet" href="styles.css" />',
        f"<style>\n{css}\n</style>",
    )
    index = re.sub(r'<script src="data\.js"></script>\s*', "", index)
    index = re.sub(
        r'<script type="text/babel" src="[^"]+"></script>\s*',
        "",
        index,
    )

    data_block = (
        "<script>\n"
        f"window.MARS_DATA = {json.dumps(payload['MARS_DATA'], allow_nan=False)};\n"
        f"window.MARS_META = {json.dumps(payload['MARS_META'], allow_nan=False)};\n"
        "</script>\n"
    )
    jsx_blocks = ""
    for name in WEB_JSX:
        code = (web_dir / name).read_text(encoding="utf-8")
        jsx_blocks += f'<script type="text/babel">\n{code}\n</script>\n'

    injection = data_block + jsx_blocks
    if "</body>" in index:
        index = index.replace("</body>", f"{injection}</body>")
    else:
        index += injection

    return index
