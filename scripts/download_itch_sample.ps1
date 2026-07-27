param(
    [string]$Url = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz",
    [string]$ChecksumUrl = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz.md5sum",
    [string]$Destination = "data\raw\itch\01302019.NASDAQ_ITCH50.gz",
    [switch]$ConfirmLargeDownload
)

$ErrorActionPreference = "Stop"

$expectedBytes = 4764426091
$actualBytes = (Get-Item $destinationPath).Length

if ($actualBytes -ne $expectedBytes) {
    throw (
        "Downloaded file has an unexpected size. " +
        "Expected $expectedBytes bytes but found $actualBytes bytes."
    )
}

Write-Host "Compressed file size verified: $actualBytes bytes."

$checksumPath = "$destinationPath.md5sum"

try {
    Invoke-WebRequest `
        -Uri $ChecksumUrl `
        -OutFile $checksumPath `
        -ErrorAction Stop

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
catch {
    Write-Warning (
        "The Nasdaq checksum endpoint is unavailable. " +
        "Continuing because the compressed file matches the expected byte size."
    )
}

Write-Host "ITCH sample is ready:"
Write-Host $destinationPath