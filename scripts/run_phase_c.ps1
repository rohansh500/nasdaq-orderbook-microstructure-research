$ErrorActionPreference = "Stop"

$outputDirectory = "reports/tables"

Get-ChildItem $outputDirectory -Filter "phase_c*" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

foreach ($horizon in @(10, 50, 100)) {
    Write-Host ""
    Write-Host "Running Phase C model comparison at $horizon events..."

    $arguments = @(
        "-m", "orderbook_research.run_model_comparison",
        "--ticker", "AAPL",
        "--levels", "10",
        "--horizon", "$horizon",
        "--folds", "5",
        "--purge-events", "100",
        "--output-directory", $outputDirectory
    )

    if ($horizon -eq 50) {
        $arguments += "--run-ablation"
    }

    & python @arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Phase C failed at the $horizon-event horizon."
    }
}

Write-Host ""
Write-Host "All Phase C model comparisons completed successfully."
