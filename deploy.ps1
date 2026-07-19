$ErrorActionPreference = "Stop"

$AppName = "cavo-vision-bot"
$RequiredPaths = @(
    "Dockerfile",
    "fly.toml",
    "deployment/catalog/catalog-manifest.json",
    "deployment/catalog-index.npz",
    "deployment/BUILD_INFO.json"
)

if (-not (Get-Command flyctl -ErrorAction SilentlyContinue)) {
    throw "flyctl is not installed or is not available in PATH."
}

foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required deployment file is missing: $Path"
    }
}

$IndexHash = (Get-FileHash "deployment/catalog-index.npz" -Algorithm SHA256).Hash.ToLowerInvariant()
$BuildInfo = Get-Content "deployment/BUILD_INFO.json" -Raw | ConvertFrom-Json
if ($IndexHash -ne $BuildInfo.index_sha256) {
    throw "Recognition index checksum does not match deployment/BUILD_INFO.json"
}

flyctl config validate -c fly.toml
if ($LASTEXITCODE -ne 0) {
    throw "Fly configuration validation failed."
}

flyctl deploy --remote-only --depot=false --ha=false --yes -a $AppName
if ($LASTEXITCODE -ne 0) {
    throw "Fly deployment failed."
}

flyctl status -a $AppName
if ($LASTEXITCODE -ne 0) {
    throw "Fly status verification failed."
}

Write-Host "CAVO bot deployment completed successfully." -ForegroundColor Green
