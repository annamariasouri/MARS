# Deprecated wrapper — use dev.ps1 from the repo root instead.
Write-Warning "Use .\dev.ps1 from the project root (one localhost: http://localhost:8765/)"
& (Join-Path (Split-Path -Parent $PSScriptRoot) "dev.ps1") @args
