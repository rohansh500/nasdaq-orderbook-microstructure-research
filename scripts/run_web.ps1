$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $projectRoot "web")

if (-not (Test-Path "node_modules")) {
    npm install
}

npm run dev
