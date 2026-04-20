@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "WEB_DIR=%ROOT%\apps\web"
set "WEB_HOST=127.0.0.1"
set "WEB_PORT=5173"
set "WEB_URL=http://%WEB_HOST%:%WEB_PORT%/"

if not exist "%WEB_DIR%\package.json" (
  echo [ERROR] Frontend project not found: "%WEB_DIR%"
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found in PATH. Please install Node.js.
  exit /b 1
)

if not exist "%WEB_DIR%\node_modules" (
  echo [SETUP] node_modules missing, running npm install...
  pushd "%WEB_DIR%"
  call npm install
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed.
    exit /b 1
  )
  popd
)

if /I "%~1"=="--dry-run" (
  echo [DRY-RUN] cd /d "%WEB_DIR%"
  echo [DRY-RUN] start "" "%WEB_URL%"
  echo [DRY-RUN] npm run dev -- --host %WEB_HOST% --port %WEB_PORT%
  exit /b 0
)

echo [START] Frontend on %WEB_URL%
echo [OPEN] Opening browser: %WEB_URL%
start "" "%WEB_URL%"
cd /d "%WEB_DIR%"
call npm run dev -- --host %WEB_HOST% --port %WEB_PORT%

