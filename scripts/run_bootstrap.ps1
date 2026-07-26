$ErrorActionPreference = "Stop"

$outputDirectory = "reports/tables"

Remove-Item "$outputDirectory/bootstrap_h*_intervals.csv" `
    -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/bootstrap_h*_metrics.json" `
    -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/bootstrap_summary.csv" `
    -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/bootstrap_summary_metrics.json" `
    -Force -ErrorAction SilentlyContinue

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running $horizon-event moving-block bootstrap..."

    python -m orderbook_research.run_bootstrap `
        --ticker AAPL `
        --levels 10 `
        --horizon $horizon `
        --folds 5 `
        --purge-events 100 `
        --bootstrap-draws 1000 `
        --block-length 1000 `
        --confidence-level 0.95 `
        --random-seed 42

    if ($LASTEXITCODE -ne 0) {
        throw "$horizon-event bootstrap failed."
    }
}

Write-Host ""
Write-Host "All moving-block bootstrap experiments completed."
