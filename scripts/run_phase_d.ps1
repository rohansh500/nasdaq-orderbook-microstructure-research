param(
    [switch]$Smoke,
    [switch]$AllowRerun
)

$ErrorActionPreference = "Stop"

if ($Smoke) {
    Write-Host "Running Phase D smoke evaluation on 50,000 rows..."

    Remove-Item reports\tables\phase_d_smoke -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item reports\figures\phase_d_smoke -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item reports\phase_d_smoke.md -Force -ErrorAction SilentlyContinue
    Remove-Item models\phase_d_smoke -Recurse -Force -ErrorAction SilentlyContinue

    python -m orderbook_research.run_final_evaluation `
        --ticker AAPL `
        --levels 10 `
        --horizon 50 `
        --purge-events 100 `
        --max-rows 50000 `
        --smoke `
        --table-directory reports/tables/phase_d_smoke `
        --figure-directory reports/figures/phase_d_smoke `
        --report-path reports/phase_d_smoke.md `
        --model-directory models/phase_d_smoke

    if ($LASTEXITCODE -ne 0) {
        throw "Phase D smoke evaluation failed."
    }

    Write-Host "Phase D smoke evaluation completed."
    exit 0
}

Write-Host ""
Write-Host "FINAL FROZEN-CANDIDATE EVALUATION"
Write-Host "This command evaluates the selected 50-event LightGBM no-time candidate."
Write-Host "Do not rerun or tune against the result."
Write-Host ""

$confirmation = Read-Host "Type RUN FINAL HOLDOUT to continue"
if ($confirmation -ne "RUN FINAL HOLDOUT") {
    throw "Final evaluation cancelled."
}

$arguments = @(
    "-m", "orderbook_research.run_final_evaluation",
    "--ticker", "AAPL",
    "--levels", "10",
    "--horizon", "50",
    "--purge-events", "100",
    "--confirm-final-holdout"
)

if ($AllowRerun) {
    $arguments += "--allow-rerun"
}

& python @arguments

if ($LASTEXITCODE -ne 0) {
    throw "Phase D final evaluation failed."
}

Write-Host ""
Write-Host "Phase D final evaluation completed successfully."
Write-Host "Review reports/tables/final_holdout_metrics.json"
Write-Host "Review reports/final_research_note.md"
