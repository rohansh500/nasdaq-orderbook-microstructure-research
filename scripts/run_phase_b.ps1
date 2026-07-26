$ErrorActionPreference = "Stop"

$outputDirectory = "reports/tables"

Get-ChildItem $outputDirectory -Filter "phase_b*" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running Phase B for $horizon-event horizon..."

    python -m orderbook_research.run_regime_analysis `
        --ticker AAPL `
        --levels 10 `
        --horizon $horizon `
        --folds 5 `
        --purge-events 100 `
        --tick-size 0.01 `
        --confidence-thresholds "0.05,0.10,0.20,0.30,0.40,0.50" `
        --cost-fractions "0,0.25,0.50,0.75,1.0" `
        --output-directory $outputDirectory

    if ($LASTEXITCODE -ne 0) {
        throw "$horizon-event Phase B analysis failed."
    }
}

Write-Host ""
Write-Host "All Phase B analyses completed successfully."
