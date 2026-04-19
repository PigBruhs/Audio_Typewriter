param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [string]$WebHost = "127.0.0.1",
    [int]$WebPort = 5173,
    [switch]$InstallIfMissing
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm not found in PATH. Install Node.js first."
}

if ($InstallIfMissing) {
    Write-Host "[setup] Running bootstrap (backend + web deps)..."
    & "$repoRoot\scripts\bootstrap.ps1" -InstallWeb
}

if (-not (Test-Path "$repoRoot\apps\web\node_modules")) {
    Write-Host "[setup] apps/web/node_modules not found, running npm install..."
    Push-Location "$repoRoot\apps\web"
    npm install
    Pop-Location
}

$pythonFull = (Resolve-Path $PythonExe).Path
$pyPathEntries = @(
    "$repoRoot\apps\api",
    "$repoRoot\packages\core"
)
$joinedPyPath = [string]::Join(";", $pyPathEntries)

$apiCommand = @(
    "`$env:PYTHONPATH = '$joinedPyPath'",
    "Set-Location '$repoRoot'",
    "& '$pythonFull' -m uvicorn app.main:app --host $ApiHost --port $ApiPort --app-dir apps/api"
) -join "; "

$webCommand = @(
    "Set-Location '$repoRoot\apps\web'",
    "npm run dev -- --host $WebHost --port $WebPort"
) -join "; "

Write-Host "[start] Launching API server..."
$apiProc = Start-Process powershell -ArgumentList @("-NoExit", "-Command", $apiCommand) -PassThru

$healthUrl = "http://$ApiHost`:$ApiPort/api/v1/health"
$apiReady = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $null = Invoke-RestMethod -Uri $healthUrl -Method GET -TimeoutSec 2
        $apiReady = $true
        break
    } catch {
        # Wait until backend is ready.
    }
}

if (-not $apiReady) {
    Write-Warning "API did not pass health check in time. Web will still be started."
}

Write-Host "[start] Launching Web dev server..."
$webProc = Start-Process powershell -ArgumentList @("-NoExit", "-Command", $webCommand) -PassThru

Write-Host ""
Write-Host "Audio_Typewriter started." -ForegroundColor Green
Write-Host "API: http://$ApiHost`:$ApiPort"
Write-Host "Web: http://$WebHost`:$WebPort"
Write-Host "API PID: $($apiProc.Id), Web PID: $($webProc.Id)"
Write-Host "Use the Web UI Exit button for graceful shutdown, or close the two opened PowerShell windows."

