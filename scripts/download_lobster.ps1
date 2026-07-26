$ErrorActionPreference = "Stop"

# Hugging Face defaults are short for large-file downloads.
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:HF_HUB_ETAG_TIMEOUT = "60"

# Harmless Windows cache warning; disabling it does not change downloads.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$maximumAttempts = 4
$downloadSucceeded = $false

for ($attempt = 1; $attempt -le $maximumAttempts; $attempt++) {
    Write-Host ""
    Write-Host "LOBSTER download attempt $attempt of $maximumAttempts..."

    & .\.venv\Scripts\python.exe `
        -m orderbook_research.download_lobster `
        --ticker AAPL `
        --levels 10 `
        --output-root data/raw/lobster

    if ($LASTEXITCODE -eq 0) {
        $downloadSucceeded = $true
        break
    }

    if ($attempt -lt $maximumAttempts) {
        $delaySeconds = 5 * $attempt
        Write-Warning "Download interrupted. Retrying in $delaySeconds seconds..."
        Start-Sleep -Seconds $delaySeconds
    }
}

if (-not $downloadSucceeded) {
    throw "LOBSTER sample download failed after $maximumAttempts attempts."
}

Write-Host ""
Write-Host "LOBSTER sample download completed successfully."