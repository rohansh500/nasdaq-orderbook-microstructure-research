$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m orderbook_research.audit `
    --ticker AAPL `
    --levels 10 `
    --max-rows 100000

& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m orderbook_research.smoke_test

Write-Host ""
Write-Host "Phase 0 checks passed."
