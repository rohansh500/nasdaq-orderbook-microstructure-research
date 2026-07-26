$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}

# Remove only prior Phase A1 outputs so every horizon is generated from
# the same current source code.
Remove-Item `
    reports\tables\walk_forward_h*_folds.csv, `
    reports\tables\walk_forward_h*_metrics.json, `
    reports\tables\walk_forward_summary.csv, `
    reports\tables\walk_forward_summary_metrics.json `
    -Force `
    -ErrorAction SilentlyContinue

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running $horizon-event walk-forward validation..."

    & .\.venv\Scripts\python.exe `
        -m orderbook_research.run_walk_forward `
        --ticker AAPL `
        --levels 10 `
        --horizon $horizon `
        --folds 5 `
        --purge-events 100

    if ($LASTEXITCODE -ne 0) {
        throw "$horizon-event walk-forward experiment failed."
    }
}

Write-Host ""
Write-Host "All walk-forward experiments completed successfully."
Write-Host ""
Get-ChildItem reports\tables\walk_forward* |
    Sort-Object Name |
    Format-Table Name, Length, LastWriteTime
