@echo off
REM ============================================================================
REM DISCORD MULTI-AGENT AI FRAMEWORK v4.3 — WINDOWS & SSH DAEMON MANAGER
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if exist "bots\bot_info\python_modules\orchestrator.py" (
    set "ORCH_PATH=bots\bot_info\python_modules\orchestrator.py"
    set "TEST_PATH=bots\bot_info\python_modules\test_framework.py"
    set "BOT_INFO_DIR=bots\bot_info"
    set "MOD_DIR=%~dp0bots\bot_info\python_modules"
) else if exist "bot_info\python_modules\orchestrator.py" (
    set "ORCH_PATH=bot_info\python_modules\orchestrator.py"
    set "TEST_PATH=bot_info\python_modules\test_framework.py"
    set "BOT_INFO_DIR=bot_info"
    set "MOD_DIR=%~dp0bot_info\python_modules"
) else (
    set "ORCH_PATH=orchestrator.py"
    set "TEST_PATH=test_framework.py"
    set "BOT_INFO_DIR=."
    set "MOD_DIR=%~dp0"
)

set "PID_FILE=%BOT_INFO_DIR%\orchestrator.pid"
set "STATE_FILE=%BOT_INFO_DIR%\short_term_memory\shared_image_cache.json"
set "ENV_FILE=%BOT_INFO_DIR%\.env"
set "LOG_DIR=%BOT_INFO_DIR%\bot_logs"
set "LOG_FILE=%LOG_DIR%\orchestrator_daemon.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" 2>nul
if not exist "%BOT_INFO_DIR%\short_term_memory" mkdir "%BOT_INFO_DIR%\short_term_memory" 2>nul
if not exist "%BOT_INFO_DIR%\long_term_memory" mkdir "%BOT_INFO_DIR%\long_term_memory" 2>nul
if not exist "%BOT_INFO_DIR%\config_files" mkdir "%BOT_INFO_DIR%\config_files" 2>nul

set "PYTHONUNBUFFERED=1"
set "PYTHONPATH=%MOD_DIR%;%PYTHONPATH%"

REM Direct CLI flags: manage_bots.bat start | stop | restart | status | logs | --test
if /I "%~1"=="--test" goto :RUN_TESTS
if /I "%~1"=="test" goto :RUN_TESTS
if /I "%~1"=="start" (
    set "EXTRA_ARGS=%~2 %~3"
    goto :START_DAEMON
)
if /I "%~1"=="stop" goto :STOP_DAEMON
if /I "%~1"=="kill" goto :STOP_DAEMON
if /I "%~1"=="shutdown" goto :STOP_DAEMON
if /I "%~1"=="restart" (
    set "EXTRA_ARGS=%~2 %~3"
    call :DO_STOP
    timeout /t 1 /nobreak >nul
    goto :START_DAEMON
)
if /I "%~1"=="status" goto :SHOW_STATUS
if /I "%~1"=="logs" goto :TAIL_LOGS

:MAIN_MENU
echo.
echo ==============================================================================
echo  DISCORD MULTI-AGENT CONTROL PLANE — SSH ^& WINDOWS MANAGER
echo ==============================================================================
call :PRINT_STATUS_AND_LINKS
echo ==============================================================================
echo   [1] Start / Launch in Background (SSH-Safe Daemon — Standby Mode in Web-UI)
echo   [2] Start in Background + Auto-Start All Bots (--autostart)
echo   [3] Start in Background + Public Cloudflare HTTPS Tunnel (--tunnel)
echo   [4] STOP / Shut Down Background Server ^& All Bots
echo   [5] Restart Background Server
echo   [6] View Live Console Logs (Press Ctrl+C to exit log viewer safely)
echo   [7] Run in Foreground Console Mode (Ctrl+C to stop)
echo   [8] Exit Menu
echo ==============================================================================
set "CHOICE=8"
set /p "CHOICE=Select an option [1-8]: "

if "!CHOICE!"=="1" (
    set "EXTRA_ARGS="
    goto :START_DAEMON
)
if "!CHOICE!"=="2" (
    set "EXTRA_ARGS=--autostart"
    goto :START_DAEMON
)
if "!CHOICE!"=="3" (
    set "EXTRA_ARGS=--tunnel"
    goto :START_DAEMON
)
if "!CHOICE!"=="4" goto :STOP_DAEMON
if "!CHOICE!"=="5" (
    set "EXTRA_ARGS="
    call :DO_STOP
    timeout /t 1 /nobreak >nul
    goto :START_DAEMON
)
if "!CHOICE!"=="6" goto :TAIL_LOGS
if "!CHOICE!"=="7" (
    call :DO_STOP
    echo [FOREGROUND] Starting Orchestrator in foreground console mode...
    python -u "%ORCH_PATH%" --host 0.0.0.0
    pause
    exit /b 0
)
echo Exiting menu (any background daemon will continue running).
exit /b 0


:START_DAEMON
echo.
echo ==============================================================================
echo  Launching Orchestrator + Web-UI in Background (SSH-Safe Detached Process)...
echo ==============================================================================
python -c "import os, sys, json, time, socket, subprocess; mod=r'%MOD_DIR%'; sys.path.insert(0, mod); orch=os.path.abspath(r'%ORCH_PATH%'); logf=os.path.abspath(r'%LOG_FILE%'); pf=os.path.abspath(r'%PID_FILE%'); extra='%EXTRA_ARGS%'.split(); env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env['PYTHONPATH']=mod+os.pathsep+env.get('PYTHONPATH',''); DETACHED=0x00000008|0x00000200|0x08000000; lf=open(logf,'a',encoding='utf-8'); p=subprocess.Popen([sys.executable,'-u',orch,'--host','0.0.0.0']+extra, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL, cwd=os.path.dirname(orch) or '.', env=env, creationflags=DETACHED); print(f'  * Spawned Detached Background PID: {p.pid}')"

REM Wait up to 3 seconds for port binding
timeout /t 3 /nobreak >nul
echo ==============================================================================
call :PRINT_STATUS_AND_LINKS
echo ------------------------------------------------------------------------------
echo  Safe to close your SSH session! Run 'manage_bots.bat' anytime to Stop [4].
echo ==============================================================================
exit /b 0


:STOP_DAEMON
call :DO_STOP
exit /b 0

:DO_STOP
echo.
echo [STOP] Shutting down Orchestrator Web-UI and all Discord bots...
python -c "import os,sys,json,urllib.request; pf=r'%PID_FILE%'; ef=r'%ENV_FILE%'; tok=next((l.split('=',1)[1].strip().strip('\"\'') for l in open(ef,encoding='utf-8',errors='ignore') if l.strip().startswith('WEBUI_ADMIN_TOKEN=')),'') if os.path.exists(ef) else ''; port=json.load(open(pf)).get('port',8765) if os.path.exists(pf) else 8765; req=urllib.request.Request(f'http://127.0.0.1:{port}/api/orchestrator/shutdown', data=b'{}', headers={'Content-Type':'application/json','X-Admin-Token':tok}, method='POST'); [urllib.request.urlopen(req, timeout=2) for _ in [1] if True] if os.path.exists(pf) else None" 2>nul
timeout /t 1 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'orchestrator\.py' }; " ^
    "foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
python -c "import sys; sys.path.insert(0, r'%MOD_DIR%'); import orchestrator; scripts=orchestrator.discover_bot_launcher_scripts(); orchestrator.kill_orphaned_bot_processes(scripts)" 2>nul
if exist "%PID_FILE%" del /f /q "%PID_FILE%" 2>nul
echo [STOPPED] All Discord bots and the Orchestrator Web-UI have been shut down cleanly.
goto :eof


:SHOW_STATUS
echo ==============================================================================
call :PRINT_STATUS_AND_LINKS
echo ==============================================================================
exit /b 0


:PRINT_STATUS_AND_LINKS
python -c "import os, sys, json, socket; sys.path.insert(0, r'%MOD_DIR%'); from core_system import _is_pid_alive; pf=r'%PID_FILE%'; sf=r'%STATE_FILE%'; ef=r'%ENV_FILE%'; port=8765; pid=0; alive=False;" ^
       -c "if os.path.exists(pf):" ^
       -c "  try: d=json.load(open(pf)); pid=int(d.get('pid',0) or 0); port=int(d.get('port',8765) or 8765); alive=_is_pid_alive(pid)" ^
       -c "  except Exception: pass" ^
       -c "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); lip=s.getsockname()[0]; s.close();" ^
       -c "sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM); sock.settimeout(0.4); port_open=(sock.connect_ex(('127.0.0.1',port))==0); sock.close();" ^
       -c "wi=json.load(open(sf)).get('_webui_control_plane_info',{}) if os.path.exists(sf) else {}; pub=wi.get('public_url','');" ^
       -c "tok=next((l.split('=',1)[1].strip().strip('\"\'') for l in open(ef,encoding='utf-8',errors='ignore') if l.strip().startswith('WEBUI_ADMIN_TOKEN=')),'') if os.path.exists(ef) else ''; sfx=f'/?token={tok}' if tok else '';" ^
       -c "status_str = f'RUNNING IN BACKGROUND (PID: {pid})' if (alive or port_open) else 'STOPPED';" ^
       -c "print(f'  STATUS: {status_str}');" ^
       -c "print(f'  * LAN / Desktop Browser URL: http://{lip}:{port}{sfx}');" ^
       -c "print(f'  * Localhost / SSH URL:       http://127.0.0.1:{port}{sfx}');" ^
       -c "if pub: print(f'  * Public Cloudflare URL:     {pub}{sfx}')" ^
       -c "if tok: print(f'  * Admin Access Token:        {tok}')" 2>nul
goto :eof


:TAIL_LOGS
echo Tailing %LOG_FILE% (Press Ctrl+C to exit log viewer — bots will keep running)...
if not exist "%LOG_FILE%" type nul > "%LOG_FILE%"
powershell -NoProfile -Command "Get-Content '%LOG_FILE%' -Tail 60 -Wait"
exit /b 0


:RUN_TESTS
echo [TEST] Running Offline Unit Test Suite (%TEST_PATH%)...
python "%TEST_PATH%"
exit /b 0