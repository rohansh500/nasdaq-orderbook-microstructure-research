$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11 or newer and reopen PowerShell."
}

python --version
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m ipykernel install `
    --user `
    --name nasdaq-orderbook `
    --display-name "Python (nasdaq-orderbook)"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with: .venv\Scripts\Activate.ps1"
