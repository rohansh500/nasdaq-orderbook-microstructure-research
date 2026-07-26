$ErrorActionPreference = "Stop"

$outputDirectory = "reports/tables"

Remove-Item "$outputDirectory/diagnostics_h*_folds.csv" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_h*_autocorrelation.csv" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_h*_ljung_box.csv" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_h*_time_buckets.csv" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_h*_metrics.json" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_summary.csv" -Force -ErrorAction SilentlyContinue
Remove-Item "$outputDirectory/diagnostics_summary_metrics.json" -Force -ErrorAction SilentlyContinue

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running $horizon-event econometric diagnostics..."

    python -m orderbook_research.run_diagnostics `
        --ticker AAPL `
        --levels 10 `
        --horizon $horizon `
        --folds 5 `
        --purge-events 100 `
        --bucket-minutes 30 `
        --minimum-bucket-observations 100 `
        --output-directory $outputDirectory

    if ($LASTEXITCODE -ne 0) {
        throw "$horizon-event diagnostics failed."
    }
}

Write-Host ""
Write-Host "All A3 diagnostics completed successfully."
