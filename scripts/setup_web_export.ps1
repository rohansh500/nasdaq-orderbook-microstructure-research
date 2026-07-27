param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$projectVenvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$exportVenv = Join-Path $projectRoot ".venv-web-export"
$exportPython = Join-Path $exportVenv "Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    if (Test-Path $projectVenvPython) {
        $PythonExecutable = $projectVenvPython
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw (
                "No usable Python interpreter was found. Create the main " +
                ".venv first or pass -PythonExecutable."
            )
        }
        $PythonExecutable = $pythonCommand.Source
    }
}

if (-not (Test-Path $PythonExecutable)) {
    throw "Python executable does not exist: $PythonExecutable"
}

if ((Test-Path $exportVenv) -and -not (Test-Path $exportPython)) {
    Remove-Item $exportVenv -Recurse -Force
}

if (-not (Test-Path $exportPython)) {
    & $PythonExecutable -m venv $exportVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the website export environment."
    }
}

& $exportPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $exportPython -m pip install -r requirements-web-export.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install website export dependencies."
}

& $exportPython scripts\export_web_data.py
if ($LASTEXITCODE -ne 0) {
    throw "Website data export failed."
}

& $exportPython scripts\export_web_models.py
if ($LASTEXITCODE -ne 0) {
    throw "Browser model export failed."
}

& $exportPython scripts\verify_web_assets.py
if ($LASTEXITCODE -ne 0) {
    throw "Website asset verification failed."
}

Push-Location web
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed." }

    npm run lint
    if ($LASTEXITCODE -ne 0) { throw "Website linting failed." }

    npm run typecheck
    if ($LASTEXITCODE -ne 0) { throw "Website type checking failed." }

    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Website production build failed." }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Website export and build completed successfully."
