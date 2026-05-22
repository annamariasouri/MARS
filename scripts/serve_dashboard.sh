#!/usr/bin/env bash
# Run MARS locally — same as dev.ps1 on Windows. Only http://localhost:8765/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8765
cd "$ROOT"
if [[ "${SKIP_EXPORT:-}" != "1" ]]; then
  python .github/scripts/export_dashboard_data.py
fi
echo "MARS: http://localhost:${PORT}/"
fuser -k "${PORT}/tcp" 2>/dev/null || true
exec python .github/scripts/serve_dashboard.py
