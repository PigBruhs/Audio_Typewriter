param(
    [string]$PythonExe = "python",
    [switch]$InstallWeb
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating virtual environment"
& $PythonExe -m venv .venv

Write-Host "[2/3] Installing backend dependencies"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if ($InstallWeb) {
    Write-Host "[3/3] Installing frontend dependencies"
    Push-Location "apps\web"
    npm install
    Pop-Location
} else {
    Write-Host "[3/3] Skip frontend install (use -InstallWeb to enable)"
}

Write-Host "Bootstrap complete."

