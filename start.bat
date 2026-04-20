@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BACKEND_SCRIPT=%ROOT%\start_backend.bat"
set "WEB_SCRIPT=%ROOT%\start_web.bat"
set "API_HEALTH=http://127.0.0.1:8000/api/v1/health"

if not exist "%BACKEND_SCRIPT%" (
  echo [ERROR] Missing script: "%BACKEND_SCRIPT%"
  exit /b 1
)
if not exist "%WEB_SCRIPT%" (
  echo [ERROR] Missing script: "%WEB_SCRIPT%"
  exit /b 1
)

if /I "%~1"=="--dry-run" (
  echo [DRY-RUN] Delegating to backend/web scripts...
  call "%BACKEND_SCRIPT%" --dry-run
  if errorlevel 1 exit /b 1
  call "%WEB_SCRIPT%" --dry-run
  if errorlevel 1 exit /b 1
  exit /b 0
)

echo [START] Launching backend...
start "Audio_Typewriter_API" "%BACKEND_SCRIPT%"

echo [WAIT] Waiting for backend health before launching frontend...
set "STATUS=0"
for /l %%i in (1,1,60) do (
  for /f %%s in ('powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try { (Invoke-WebRequest -UseBasicParsing '%API_HEALTH%' -TimeoutSec 2).StatusCode } catch { 0 }"') do set "STATUS=%%s"
  if "!STATUS!"=="200" goto :backend_ready
  timeout /t 1 >nul
)
echo [WARN] Backend health wait timed out. Starting frontend anyway.

:backend_ready

echo [START] Launching frontend...
start "Audio_Typewriter_Web" "%WEB_SCRIPT%"

echo [DONE] Backend and frontend launched in separate windows.
exit /b 0

