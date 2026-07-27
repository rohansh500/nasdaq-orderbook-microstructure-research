param(
    [switch]$Smoke,
    [string]$InputPath,
    [string]$Symbol = "AAPL",
    [int]$Levels = 10,
    [long]$MaxMessages = 0,
    [long]$StopAfterTargetEvents = 0,
    [switch]$NonStrict
)

$ErrorActionPreference = "Stop"

if ($Smoke) {
    Write-Host "Running Phase E synthetic BinaryFILE smoke test..."

    Remove-Item reports\tables\phase_e_smoke -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path data\raw\itch -Force | Out-Null

    python -m orderbook_research.run_itch_reconstruction `
        --generate-fixture data\raw\itch\synthetic_AAPL.itch.gz `
        --symbol AAPL `
        --levels 3 `
        --output-directory reports\tables\phase_e_smoke `
        --batch-size 3 `
        --sample-rows 20

    if ($LASTEXITCODE -ne 0) {
        throw "Phase E smoke reconstruction failed."
    }

    Write-Host "Phase E smoke reconstruction completed."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    throw "Provide -InputPath to an official or licensed ITCH 5.0 BinaryFILE (.gz is supported)."
}

$arguments = @(
    "-m", "orderbook_research.run_itch_reconstruction",
    "--input-path", $InputPath,
    "--symbol", $Symbol,
    "--levels", $Levels,
    "--output-directory", "reports/tables"
)

if ($MaxMessages -gt 0) {
    $arguments += @("--max-messages", $MaxMessages)
}
if ($StopAfterTargetEvents -gt 0) {
    $arguments += @("--stop-after-target-events", $StopAfterTargetEvents)
}
if ($NonStrict) {
    $arguments += "--non-strict"
}

Write-Host "Running Phase E raw ITCH reconstruction..."
Write-Host "Input: $InputPath"
Write-Host "Symbol: $Symbol"

& python @arguments

if ($LASTEXITCODE -ne 0) {
    throw "Phase E raw ITCH reconstruction failed."
}

Write-Host "Phase E reconstruction completed successfully."
Write-Host "Review reports/tables/phase_e_itch_${Symbol}_reconstruction_metrics.json"
