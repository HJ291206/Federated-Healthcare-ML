@echo off
setlocal enabledelayedexpansion
title MediFederate AI - Local Launcher
color 0F

echo.
echo  ============================================================
echo   MediFederate AI - Privacy-Preserving Healthcare Intelligence
echo  ============================================================
echo   Academic Research Prototype - Not for Clinical Use
echo  ============================================================
echo.

:: ----------------------------------------------------------------
:: Configuration
:: ----------------------------------------------------------------
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "DASHBOARD_DIR=%PROJECT_DIR%dashboard"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"
set "DATASET=%PROJECT_DIR%data\raw\diabetes_binary_health_indicators_BRFSS2015.csv"

:: ----------------------------------------------------------------
:: Step 1: Check Python virtual environment
:: ----------------------------------------------------------------
echo [1/6] Checking Python virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo  ERROR: Python virtual environment not found at %VENV_DIR%
    echo  Please create it first:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo    pip install -r requirements-api.txt
    echo.
    pause
    exit /b 1
)
echo   [OK] Virtual environment found at %VENV_DIR%

:: ----------------------------------------------------------------
:: Step 2: Check Node.js
:: ----------------------------------------------------------------
echo [2/6] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Node.js is not installed or not in PATH.
    echo  Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version') do set "NODE_VER=%%v"
echo   [OK] Node.js %NODE_VER%

:: ----------------------------------------------------------------
:: Step 3: Check dataset
:: ----------------------------------------------------------------
echo [3/6] Checking dataset...
if not exist "%DATASET%" (
    echo.
    echo  ERROR: Primary dataset not found:
    echo    %DATASET%
    echo.
    echo  Please download the CDC BRFSS 2015 Diabetes Health Indicators dataset
    echo  and place it at the path above.
    echo.
    pause
    exit /b 1
)
echo   [OK] Dataset found

:: ----------------------------------------------------------------
:: Step 4: Check frontend dependencies
:: ----------------------------------------------------------------
echo [4/6] Checking frontend dependencies...
if not exist "%DASHBOARD_DIR%\node_modules" (
    echo   Installing frontend dependencies...
    cd /d "%DASHBOARD_DIR%"
    call npm install
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: npm install failed. Check the output above.
        pause
        exit /b 1
    )
)
echo   [OK] Frontend dependencies installed

:: ----------------------------------------------------------------
:: Step 5: Start backend
:: ----------------------------------------------------------------
echo [5/6] Checking port %BACKEND_PORT% and starting FastAPI backend...
powershell -Command "$p = Get-NetTCPConnection -LocalPort %BACKEND_PORT% -State Listen -ErrorAction SilentlyContinue; if ($p) { Write-Host '  Freeing stale process on port %BACKEND_PORT% (PID ' $p.OwningProcess ')...'; Stop-Process -Id $p.OwningProcess -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 1 }"
cd /d "%PROJECT_DIR%"
start "MediFederate AI - Backend" cmd /c "call %VENV_DIR%\Scripts\activate.bat && python -m uvicorn api.main:app --host 127.0.0.1 --port %BACKEND_PORT% --reload"
timeout /t 3 /nobreak >nul

:: Quick health check
echo   Checking backend health...
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%BACKEND_PORT%/api/health' -TimeoutSec 10 -UseBasicParsing; if ($r.StatusCode -eq 200) { Write-Host '  [OK] Backend is running' } else { Write-Host '  [WARN] Backend returned status' $r.StatusCode } } catch { Write-Host '  [WARN] Backend may still be starting...' }"

:: ----------------------------------------------------------------
:: Step 6: Start frontend
:: ----------------------------------------------------------------
echo [6/6] Starting React frontend on port %FRONTEND_PORT%...
cd /d "%DASHBOARD_DIR%"
start "MediFederate AI - Frontend" cmd /c "call npm run dev"
timeout /t 3 /nobreak >nul

:: ----------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------
echo.
echo  ============================================================
echo   MediFederate AI is running!
echo  ============================================================
echo.
echo   Frontend:  http://localhost:%FRONTEND_PORT%
echo   Backend:   http://127.0.0.1:%BACKEND_PORT%
echo   API docs:  http://127.0.0.1:%BACKEND_PORT%/docs
echo.
echo   To stop: close the Backend and Frontend terminal windows,
echo            or press Ctrl+C in each.
echo.
echo  ============================================================
echo.
pause
