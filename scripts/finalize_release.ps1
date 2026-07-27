$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    throw "Run this script from the Git repository root."
}

Write-Host "Removing public copies of local-only generated outputs..."

git rm --ignore-unmatch reports/phase_d_smoke.md

git rm --cached --ignore-unmatch `
    reports/tables/phase_e_itch_AAPL_events_sample.csv `
    reports/tables/phase_e_itch_AAPL_features_sample.csv `
    reports/tables/final_holdout_lightgbm_simulation.csv `
    reports/tables/final_holdout_logistic_simulation.csv

Write-Host "Running final quality checks..."
python -m ruff check src tests
python -m ruff format --check src tests
pytest
git diff --check

Write-Host ""
Write-Host "Release files are ready for review."
Write-Host "Inspect git status and the staged removals before committing:"
git status --short
