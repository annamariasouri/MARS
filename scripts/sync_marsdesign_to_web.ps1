# Copy MARSdesign UI assets into web/ (used by Streamlit and GitHub Pages).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root "MARSdesign"
$dst = Join-Path $root "web"
# web/app.jsx is kept separately (MARSdesign UI + live MARS_META from CSVs).
$files = @("styles.css", "map.jsx", "charts.jsx", "logo.jsx", "tweaks-panel.jsx", "favicon.png")
foreach ($f in $files) {
    Copy-Item -Path (Join-Path $src $f) -Destination (Join-Path $dst $f) -Force
    Write-Host "Synced $f"
}
Write-Host "Done. Merge design changes into web/app.jsx manually if needed."
