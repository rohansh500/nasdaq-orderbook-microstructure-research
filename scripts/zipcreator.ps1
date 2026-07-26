$projectRoot = (Get-Location).Path
$bundleFolder = Join-Path $projectRoot "_review_bundle"
$zipPath = Join-Path `
    (Split-Path $projectRoot -Parent) `
    "nasdaq-orderbook-microstructure-research-current.zip"

# Remove previous bundle
Remove-Item $bundleFolder -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Path $bundleFolder | Out-Null

# Copy relevant source and research folders
$foldersToInclude = @(
    "src",
    "scripts",
    "notebooks",
    "configs",
    "docs",
    "tests",
    "reports"
)

foreach ($folder in $foldersToInclude) {
    $source = Join-Path $projectRoot $folder

    if (Test-Path $source) {
        Copy-Item `
            -Path $source `
            -Destination $bundleFolder `
            -Recurse `
            -Force
    }
}

# Copy important root files
$filesToInclude = @(
    "README.md",
    "PROJECT_PLAN.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "LICENSE"
)

foreach ($file in $filesToInclude) {
    $source = Join-Path $projectRoot $file

    if (Test-Path $source) {
        Copy-Item `
            -Path $source `
            -Destination $bundleFolder `
            -Force
    }
}

# Remove caches and heavy generated files from the review copy
Get-ChildItem $bundleFolder -Recurse -Directory |
    Where-Object {
        $_.Name -in @(
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
            ".ruff_cache"
        )
    } |
    Remove-Item -Recurse -Force

Get-ChildItem $bundleFolder -Recurse -File |
    Where-Object {
        $_.Extension -in @(
            ".pyc",
            ".parquet",
            ".joblib",
            ".pkl",
            ".zip"
        )
    } |
    Remove-Item -Force

# Record the exact local environment
@(
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    "Project: $projectRoot"
    ""
    "Python:"
    (& python --version 2>&1)
    ""
    "Installed packages:"
    (& python -m pip freeze 2>&1)
) | Set-Content `
    (Join-Path $bundleFolder "environment_state.txt")

# Record Git status if Git has been initialized
if (Test-Path (Join-Path $projectRoot ".git")) {
    git status --short |
        Set-Content (Join-Path $bundleFolder "git_status.txt")

    git log --oneline -20 |
        Set-Content (Join-Path $bundleFolder "git_log.txt")

    git diff |
        Set-Content (Join-Path $bundleFolder "git_uncommitted_diff.txt")
}
else {
    "Git has not yet been initialized in this repository." |
        Set-Content (Join-Path $bundleFolder "git_status.txt")
}

# Create ZIP
Compress-Archive `
    -Path "$bundleFolder\*" `
    -DestinationPath $zipPath `
    -CompressionLevel Optimal

# Remove temporary copied folder
Remove-Item $bundleFolder -Recurse -Force

Write-Host ""
Write-Host "Review ZIP created:"
Write-Host $zipPath