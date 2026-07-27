param(
    [string]$Url = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz",
    [string]$ChecksumUrl = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz.md5sum",
    [string]$Destination = "data\raw\itch\01302019.NASDAQ_ITCH50.gz",
    [switch]$ConfirmLargeDownload
)

$ErrorActionPreference = "Stop"
$expectedBytes = 4764426091

if (-not $ConfirmLargeDownload) {
    throw (
        "This download is approximately 4.76 GB. " +
        "Run again with -ConfirmLargeDownload."
    )
}

$destinationPath = if ([System.IO.Path]::IsPathRooted($Destination)) {
    $Destination
}
else {
    Join-Path (Get-Location) $Destination
}

$destinationPath = [System.IO.Path]::GetFullPath($destinationPath)
$destinationDirectory = Split-Path $destinationPath -Parent
New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

$downloadRequired = $true
if (Test-Path $destinationPath) {
    $existingBytes = (Get-Item $destinationPath).Length

    if ($existingBytes -eq $expectedBytes) {
        Write-Host "Existing ITCH sample has the expected size. Reusing it."
        $downloadRequired = $false
    }
    else {
        throw (
            "An incomplete or unexpected file already exists at $destinationPath. " +
            "Expected $expectedBytes bytes but found $existingBytes. " +
            "Remove or rename it before downloading again."
        )
    }
}

if ($downloadRequired) {
    Write-Host "Downloading the official Nasdaq ITCH sample..."
    Write-Host $Url

    Invoke-WebRequest `
        -Uri $Url `
        -OutFile $destinationPath
}

$actualBytes = (Get-Item $destinationPath).Length
if ($actualBytes -ne $expectedBytes) {
    throw (
        "Downloaded file has an unexpected size. " +
        "Expected $expectedBytes bytes but found $actualBytes bytes."
    )
}

Write-Host "Compressed file size verified: $actualBytes bytes."

$checksumPath = "$destinationPath.md5sum"
$checksumAvailable = $false

try {
    Invoke-WebRequest `
        -Uri $ChecksumUrl `
        -OutFile $checksumPath `
        -ErrorAction Stop

    $checksumAvailable = $true
}
catch {
    Write-Warning (
        "The Nasdaq checksum endpoint is unavailable. " +
        "Continuing because the compressed file matches the expected byte size."
    )
}

if ($checksumAvailable) {
    $expected = (
        (Get-Content $checksumPath -Raw).Trim() -split "\s+"
    )[0].ToLowerInvariant()

    $actual = (
        Get-FileHash -Path $destinationPath -Algorithm MD5
    ).Hash.ToLowerInvariant()

    if ($expected -ne $actual) {
        throw "MD5 mismatch. Expected $expected but calculated $actual."
    }

    Write-Host "MD5 verification completed."
}

Write-Host "ITCH sample is ready:"
Write-Host $destinationPath
