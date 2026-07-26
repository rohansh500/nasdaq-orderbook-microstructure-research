$ErrorActionPreference = "Stop"

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running $horizon-event baseline..."

    & .\.venv\Scripts\python.exe `
        -m orderbook_research.train_baseline `
        --ticker AAPL `
        --levels 10 `
        --horizon $horizon

    if ($LASTEXITCODE -ne 0) {
        throw "$horizon-event baseline failed."
    }
}

Write-Host ""
Write-Host "All baseline experiments completed successfully."