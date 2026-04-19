@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "WEB_HOST=127.0.0.1"
set "WEB_PORT=5173"
set "API_HEALTH=http://%API_HOST%:%API_PORT%/api/v1/health"
set "WEB_URL=http://%WEB_HOST%:%WEB_PORT%/"

set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: "%PYTHON_EXE%"
  echo Please run scripts\bootstrap.ps1 first.
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found in PATH. Please install Node.js.
  exit /b 1
)

if not exist "%ROOT%\apps\web\node_modules" (
  echo [SETUP] apps\web\node_modules missing, running npm install...
  pushd "%ROOT%\apps\web"
  call npm install
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed.
    exit /b 1
  )
  popd
)

set "PYTHONPATH=%ROOT%\apps\api;%ROOT%\packages\core"

if /I "%~1"=="--dry-run" (
  echo [DRY-RUN] API: cd /d "%ROOT%" ^&^& set "PYTHONPATH=%PYTHONPATH%" ^&^& "%PYTHON_EXE%" -m uvicorn app.main:app --host %API_HOST% --port %API_PORT% --app-dir apps/api
  echo [DRY-RUN] WEB: cd /d "%ROOT%\apps\web" ^&^& npm run dev -- --host %WEB_HOST% --port %WEB_PORT%
  exit /b 0
)

echo [START] Launching API...
start "Audio_Typewriter_API" cmd /k "cd /d ^"%ROOT%^" && set ^"PYTHONPATH=%PYTHONPATH%^" && ^"%PYTHON_EXE%^" -m uvicorn app.main:app --host %API_HOST% --port %API_PORT% --app-dir apps/api"

echo [WAIT] Waiting for API health...
set "STATUS=0"
for /l %%i in (1,1,60) do (
  for /f %%s in ('powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try { (Invoke-WebRequest -UseBasicParsing '%API_HEALTH%' -TimeoutSec 2).StatusCode } catch { 0 }"') do set "STATUS=%%s"
  if "!STATUS!"=="200" goto :api_ready
  timeout /t 1 >nul
)

echo [WARN] API health check timed out. Continuing startup.

:api_ready
echo [START] Launching Web...
start "Audio_Typewriter_Web" cmd /k "cd /d ^"%ROOT%\apps\web^" && npm run dev -- --host %WEB_HOST% --port %WEB_PORT%"

echo [OPEN] Opening browser: %WEB_URL%
start "" "%WEB_URL%"

echo [DONE] Backend/Web started in separate windows.
exit /b 0

