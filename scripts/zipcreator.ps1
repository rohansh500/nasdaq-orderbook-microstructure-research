param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    throw "Run this script from the Git repository root."
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $projectRoot = (Get-Location).Path
    $OutputPath = Join-Path `
        (Split-Path $projectRoot -Parent) `
        "nasdaq-orderbook-microstructure-research-v1.0.0.zip"
}

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
Remove-Item $OutputPath -Force -ErrorAction SilentlyContinue

git archive `
    --format=zip `
    --output=$OutputPath `
    HEAD

Write-Host "Release archive created:"
Write-Host $OutputPath
