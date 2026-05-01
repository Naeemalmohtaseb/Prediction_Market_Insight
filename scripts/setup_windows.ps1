$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next commands:"
Write-Host "  .\scripts\run_tests.ps1"
Write-Host "  .\scripts\run_app.ps1"
Write-Host "  python scripts\smoke_polymarket.py"
Write-Host "  python scripts\smoke_forecast.py"
