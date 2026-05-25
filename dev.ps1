# Run MARS locally — same app as online (Streamlit + Claude design).
# Opens http://localhost:8765/
param([switch]$SkipExport)

$ErrorActionPreference = "Stop"
$Port = 8765
$Root = $PSScriptRoot
Set-Location $Root

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $listeners) {
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
}

$Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py", "-3" } else { "python" }

if (-not $SkipExport) {
    Write-Host "Updating dashboard data from data/ ..."
    & @Python .github/scripts/export_dashboard_data.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "MARS (Streamlit + new design): http://localhost:$Port/"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

Start-Process "http://localhost:$Port/"
& @Python -m streamlit run streamlit_app.py --server.port $Port --server.headless true
