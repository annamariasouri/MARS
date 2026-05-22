"""
Serve MARS at http://localhost:8765/ only.

Do not run other http.server commands. Use:  .\\dev.ps1
"""
from __future__ import annotations

import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DIR = os.path.join(REPO_ROOT, "web")
PORT = 8765


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def list_directory(self, path):
        self.send_response(302)
        self.send_header("Location", "/")
        self.end_headers()
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    if not os.path.isfile(os.path.join(WEB_DIR, "index.html")):
        print(f"ERROR: Missing {WEB_DIR}/index.html", file=sys.stderr)
        sys.exit(1)
    try:
        server = ThreadingHTTPServer(("", PORT), DashboardHandler)
    except OSError as e:
        print(f"ERROR: Port {PORT} is in use. Close other terminals and run .\\dev.ps1 again.", file=sys.stderr)
        sys.exit(1)
    print(f"MARS dashboard: http://localhost:{PORT}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
