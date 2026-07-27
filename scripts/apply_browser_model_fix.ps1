$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Remove-Item web\public\models\*.onnx -Force -ErrorAction SilentlyContinue
Remove-Item web\public\ort -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item web\scripts\copy-ort-wasm.mjs -Force -ErrorAction SilentlyContinue

Write-Host "Removed obsolete ONNX and WASM artifacts."
Write-Host "Run .\scripts\setup_web_export.ps1 next."
