$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

& ".\.venv\Scripts\Activate.ps1"
$env:PYTHONPATH = "src"
python -m compileall src app tests
python -m pytest
