@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "PYTHON_EXE=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: "%PYTHON_EXE%"
  echo Please run scripts\bootstrap.ps1 first.
  exit /b 1
)

if not defined PYTHONPATH set "PYTHONPATH=%ROOT%\apps\api;%ROOT%\packages\core"
if not defined AT_APP_ENV set "AT_APP_ENV=dev"
if not defined AT_ASR_DEVICE set "AT_ASR_DEVICE=cuda"
if not defined AT_ASR_ALIGNMENT_BACKEND set "AT_ASR_ALIGNMENT_BACKEND=auto"
if not defined AT_ASR_MFA_BINARY set "AT_ASR_MFA_BINARY=D:\Anaconda\envs\mfa\Scripts\mfa.exe"
if not defined AT_ASR_MFA_DICTIONARY_PATH set "AT_ASR_MFA_DICTIONARY_PATH=C:\Users\Ecthelion\Documents\MFA\pretrained_models\dictionary\english_us_arpa.dict"
if not defined AT_ASR_MFA_ACOUSTIC_MODEL_PATH set "AT_ASR_MFA_ACOUSTIC_MODEL_PATH=C:\Users\Ecthelion\Documents\MFA\pretrained_models\acoustic\english_us_arpa.zip"

set "MFA_ENABLED=0"
set "MFA_READY=0"
set "MFA_STATUS=disabled"
call :compute_mfa_status

if /I "%~1"=="--dry-run" (
  echo [DRY-RUN] cd /d "%ROOT%"
  echo [DRY-RUN] set "PYTHONPATH=%PYTHONPATH%"
  echo [DRY-RUN] set "AT_APP_ENV=%AT_APP_ENV%"
  echo [DRY-RUN] set "AT_ASR_DEVICE=%AT_ASR_DEVICE%"
  echo [DRY-RUN] set "AT_ASR_ALIGNMENT_BACKEND=%AT_ASR_ALIGNMENT_BACKEND%"
  echo [DRY-RUN] set "AT_ASR_MFA_BINARY=%AT_ASR_MFA_BINARY%"
  echo [DRY-RUN] MFA_ENABLED=%MFA_ENABLED% MFA_READY=%MFA_READY% MFA_STATUS=%MFA_STATUS%
  if /I "%MFA_ENABLED%"=="1" (
    if not exist "%AT_ASR_MFA_BINARY%" echo [DRY-RUN][WARN] MFA binary not found: "%AT_ASR_MFA_BINARY%"
    if not defined AT_ASR_MFA_DICTIONARY_PATH echo [DRY-RUN][WARN] AT_ASR_MFA_DICTIONARY_PATH is not set.
    if not defined AT_ASR_MFA_ACOUSTIC_MODEL_PATH echo [DRY-RUN][WARN] AT_ASR_MFA_ACOUSTIC_MODEL_PATH is not set.
  )
  echo [DRY-RUN] "%PYTHON_EXE%" -m uvicorn app.main:app --host %API_HOST% --port %API_PORT% --app-dir apps/api
  exit /b 0
)

echo [START] Backend on http://%API_HOST%:%API_PORT%
echo [INFO] AT_ASR_ALIGNMENT_BACKEND=%AT_ASR_ALIGNMENT_BACKEND%
echo [INFO] MFA_ENABLED=%MFA_ENABLED% MFA_READY=%MFA_READY% MFA_STATUS=%MFA_STATUS%
if /I "%MFA_ENABLED%"=="1" (
  echo [INFO] AT_ASR_MFA_BINARY=%AT_ASR_MFA_BINARY%
  if not exist "%AT_ASR_MFA_BINARY%" echo [WARN] MFA binary not found: "%AT_ASR_MFA_BINARY%"
  if not defined AT_ASR_MFA_DICTIONARY_PATH echo [WARN] AT_ASR_MFA_DICTIONARY_PATH is not set.
  if not defined AT_ASR_MFA_ACOUSTIC_MODEL_PATH echo [WARN] AT_ASR_MFA_ACOUSTIC_MODEL_PATH is not set.
)

cd /d "%ROOT%"
"%PYTHON_EXE%" -m uvicorn app.main:app --host %API_HOST% --port %API_PORT% --app-dir apps/api

goto :eof

:compute_mfa_status
if /I "%AT_ASR_ALIGNMENT_BACKEND%"=="mfa" set "MFA_ENABLED=1"
if /I "%AT_ASR_ALIGNMENT_BACKEND%"=="auto" set "MFA_ENABLED=1"

if /I "%MFA_ENABLED%"=="0" (
  set "MFA_READY=0"
  set "MFA_STATUS=disabled"
  goto :eof
)

set "MFA_READY=1"
if not exist "%AT_ASR_MFA_BINARY%" set "MFA_READY=0"
if not defined AT_ASR_MFA_DICTIONARY_PATH set "MFA_READY=0"
if not defined AT_ASR_MFA_ACOUSTIC_MODEL_PATH set "MFA_READY=0"

if /I "%MFA_READY%"=="1" (
  set "MFA_STATUS=enabled_ready"
) else (
  set "MFA_STATUS=enabled_incomplete"
)
goto :eof

