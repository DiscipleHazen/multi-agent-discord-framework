@echo off
REM ============================================================================
REM DISCORD MULTI-AGENT AI FRAMEWORK v4.1 — UNIFIED ORCHESTRATOR & WEB-UI LAUNCHER
REM ============================================================================
setlocal
cd /d "%~dp0"

set "ORCH_PATH=bots\bot_info\python_modules\orchestrator.py"
set "TEST_PATH=bots\bot_info\python_modules\test_framework.py"

if not exist "%ORCH_PATH%" (
    if exist "orchestrator.py" (
        set "ORCH_PATH=orchestrator.py"
        set "TEST_PATH=test_framework.py"
    )
)

set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%~dp0bots\bot_info\python_modules;%PYTHONPATH%"

if /I "%~1"=="--test" (
    echo [TEST] Running Offline Unit Test Suite...
    python "%TEST_PATH%"
    pause
    exit /b 0
)

echo ==============================================================================
echo Launching Discord Multi-Agent Supervisor ^& Real-Time Web-UI Control Plane
echo   * Open http://127.0.0.1:8765 in your browser once online!
echo ==============================================================================
start "" http://127.0.0.1:8765
python -u "%ORCH_PATH%" %*
pause
