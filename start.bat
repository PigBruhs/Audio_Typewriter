@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BACKEND_SCRIPT=%ROOT%\start_backend.bat"
set "WEB_SCRIPT=%ROOT%\start_web.bat"
set "API_HEALTH=http://127.0.0.1:8000/api/v1/health"

REM MFA placeholders (optional): fill these with your own local paths if you use MFA.
REM 请网上搜索 "Montreal Forced Aligner Windows 安装教程"，完成安装后再填写。
REM set "AT_ASR_MFA_BINARY=E:\path\to\mfa.exe"
REM set "AT_ASR_MFA_DICTIONARY_PATH=E:\path\to\english_us_arpa.dict"
REM set "AT_ASR_MFA_ACOUSTIC_MODEL_PATH=E:\path\to\english_us_arpa.zip"

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
echo [INFO] 如需启用 MFA，请先在 start.bat 顶部填写 AT_ASR_MFA_* 路径。
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

