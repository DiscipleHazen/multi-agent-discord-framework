#!/usr/bin/env bash
# ==============================================================================
# DISCORD MULTI-AGENT AI FRAMEWORK v4.2 — SSH-SAFE BACKGROUND DAEMON MANAGER
# ==============================================================================
# Features:
#   - Launches orchestrator.py + Web-UI detached in the background (nohup + setsid
#     + disown) on 0.0.0.0:8765 so you can close your SSH terminal and keep bots
#     and the Web-UI running 24/7.
#   - Relaunching ./manage_bots.sh detects the running daemon, prints your live
#     LAN / Localhost / Tunnel Magic Admin Links, and gives you a 1-key menu to
#     Shut Down, Restart, or Tail Logs.
#   - Also supports direct CLI flags:
#       ./manage_bots.sh start [--autostart] [--tunnel]
#       ./manage_bots.sh stop
#       ./manage_bots.sh restart
#       ./manage_bots.sh status
#       ./manage_bots.sh logs
#       ./manage_bots.sh fg
#       ./manage_bots.sh --test
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "bots/bot_info/python_modules/orchestrator.py" ]; then
    ORCHESTRATOR_PATH="bots/bot_info/python_modules/orchestrator.py"
    TEST_SUITE_PATH="bots/bot_info/python_modules/test_framework.py"
    BOT_INFO_DIR="bots/bot_info"
elif [ -f "bot_info/python_modules/orchestrator.py" ]; then
    ORCHESTRATOR_PATH="bot_info/python_modules/orchestrator.py"
    TEST_SUITE_PATH="bot_info/python_modules/test_framework.py"
    BOT_INFO_DIR="bot_info"
else
    ORCHESTRATOR_PATH="orchestrator.py"
    TEST_SUITE_PATH="test_framework.py"
    BOT_INFO_DIR="."
fi

PID_FILE="${BOT_INFO_DIR}/orchestrator.pid"
DAEMON_PID_TXT="${BOT_INFO_DIR}/orchestrator_daemon.pid"
STATE_FILE="${BOT_INFO_DIR}/short_term_memory/shared_image_cache.json"
ENV_FILE="${BOT_INFO_DIR}/.env"
LOG_DIR="${BOT_INFO_DIR}/bot_logs"
LOG_FILE="${LOG_DIR}/orchestrator_daemon.log"

mkdir -p "$LOG_DIR" "${BOT_INFO_DIR}/short_term_memory" "${BOT_INFO_DIR}/long_term_memory" "${BOT_INFO_DIR}/config_files" 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(cd "$(dirname "$ORCHESTRATOR_PATH")" && pwd):${PYTHONPATH:-}"

PYTHON_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

get_running_pid() {
    local cand_pid=""
    if [ -f "$PID_FILE" ]; then
        cand_pid="$($PYTHON_BIN -c "import json,sys; d=json.load(open('$PID_FILE')); print(d.get('pid',''))" 2>/dev/null || true)"
        if [ -n "$cand_pid" ] && kill -0 "$cand_pid" 2>/dev/null; then
            echo "$cand_pid"
            return 0
        fi
    fi
    if [ -f "$DAEMON_PID_TXT" ]; then
        cand_pid="$(cat "$DAEMON_PID_TXT" 2>/dev/null | tr -d '[:space:]')"
        if [ -n "$cand_pid" ] && kill -0 "$cand_pid" 2>/dev/null; then
            echo "$cand_pid"
            return 0
        fi
    fi
    cand_pid="$(pgrep -f "orchestrator\.py" 2>/dev/null | head -n 1 || true)"
    if [ -n "$cand_pid" ] && kill -0 "$cand_pid" 2>/dev/null; then
        echo "$cand_pid"
        return 0
    fi
    echo ""
    return 1
}

print_connection_info() {
    $PYTHON_BIN - "$STATE_FILE" "$ENV_FILE" "$PID_FILE" <<'PYEOF'
import json, os, socket, sys

state_file, env_file, pid_file = sys.argv[1], sys.argv[2], sys.argv[3]
port = 8765
local_url = f"http://127.0.0.1:{port}"
lan_url = ""
pub_url = ""

def get_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

if os.path.exists(pid_file):
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pd = json.load(f)
            port = pd.get("port") or port
            local_url = pd.get("local_url") or f"http://127.0.0.1:{port}"
            lan_url = pd.get("lan_url") or ""
    except Exception:
        pass

if os.path.exists(state_file):
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            sd = json.load(f)
            wi = sd.get("_webui_control_plane_info", {})
            if isinstance(wi, dict):
                port = wi.get("port") or port
                local_url = wi.get("local_url") or local_url
                lan_url = wi.get("lan_url") or lan_url
                pub_url = wi.get("public_url") or ""
    except Exception:
        pass

if not lan_url:
    lan_url = f"http://{get_lan_ip()}:{port}"

token = ""
if os.path.exists(env_file):
    try:
        with open(env_file, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.strip().startswith("WEBUI_ADMIN_TOKEN="):
                    token = ln.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

sfx = f"/?token={token}" if token else ""
print(f"   • LAN / Wi-Fi Magic Link:   {lan_url}{sfx}")
print(f"   • Localhost / SSH Link:     {local_url}{sfx}")
if pub_url:
    print(f"   • Public HTTPS Tunnel Link: {pub_url}{sfx}")
if token:
    print(f"   • Admin Access Token:       {token}")
PYEOF
}

start_daemon() {
    local existing_pid
    existing_pid="$(get_running_pid || true)"
    if [ -n "$existing_pid" ]; then
        echo "=============================================================================="
        echo "✅ Orchestrator Daemon is ALREADY RUNNING in the background (PID: $existing_pid)"
        print_connection_info
        echo "=============================================================================="
        return 0
    fi

    echo "=============================================================================="
    echo "🚀 Starting Discord Multi-Agent Supervisor in the background (SSH-Safe)..."
    echo "   • Log File: $LOG_FILE"
    echo "=============================================================================="

    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "$PYTHON_BIN" -u "$ORCHESTRATOR_PATH" --host 0.0.0.0 "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
    else
        nohup "$PYTHON_BIN" -u "$ORCHESTRATOR_PATH" --host 0.0.0.0 "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
    fi
    local bg_pid=$!
    disown "$bg_pid" 2>/dev/null || true
    echo "$bg_pid" > "$DAEMON_PID_TXT"

    # Wait up to 4 seconds for the Web-UI HTTP server to initialize and write orchestrator.pid
    local i=0
    while [ $i -lt 16 ]; do
        sleep 0.25
        if [ -f "$PID_FILE" ]; then
            break
        fi
        if ! kill -0 "$bg_pid" 2>/dev/null; then
            break
        fi
        i=$((i + 1))
    done

    local active_pid
    active_pid="$(get_running_pid || true)"
    if [ -n "$active_pid" ]; then
        echo "✅ Background Daemon ONLINE (PID: $active_pid) — Safe to close your SSH terminal!"
        print_connection_info
        echo "------------------------------------------------------------------------------"
        echo "💡 Tip: Run './manage_bots.sh' anytime to check status, view logs, or shut down."
        echo "=============================================================================="
    else
        echo "❌ Failed to start background daemon. Last 20 lines of $LOG_FILE:"
        tail -n 20 "$LOG_FILE" 2>/dev/null || true
        return 1
    fi
}

stop_daemon() {
    local pid
    pid="$(get_running_pid || true)"
    if [ -z "$pid" ]; then
        echo "⚪ Orchestrator Daemon is not currently running."
        rm -f "$PID_FILE" "$DAEMON_PID_TXT" 2>/dev/null || true
    else
        echo "🛑 Stopping Orchestrator Daemon (PID: $pid) and all supervised Discord bots..."
        kill -TERM "$pid" 2>/dev/null || true
        local i=0
        while [ $i -lt 24 ]; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.25
            i=$((i + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️  Forcing termination of PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE" "$DAEMON_PID_TXT" 2>/dev/null || true
    fi

    # Sweep any remaining stray orchestrator or bot processes
    pkill -f "orchestrator\.py" 2>/dev/null || true
    "$PYTHON_BIN" -c "import sys; sys.path.insert(0, '$(dirname "$ORCHESTRATOR_PATH")'); import orchestrator; scripts=orchestrator.discover_bot_launcher_scripts(); orchestrator.kill_orphaned_bot_processes([s for s in scripts])" 2>/dev/null || true
    echo "✅ All Discord bots and the Orchestrator Web-UI have been shut down cleanly."
}

show_status() {
    local pid
    pid="$(get_running_pid || true)"
    echo "=============================================================================="
    if [ -n "$pid" ]; then
        echo "🟢 ORCHESTRATOR STATUS: RUNNING IN BACKGROUND (PID: $pid)"
        print_connection_info
    else
        echo "⚪ ORCHESTRATOR STATUS: STOPPED"
    fi
    echo "=============================================================================="
}

# ------------------------------------------------------------------------------
# CLI ARGUMENT DISPATCHER
# ------------------------------------------------------------------------------
case "${1:-}" in
    --test|test)
        echo "🧪 Running Offline Unit Test Suite ($TEST_SUITE_PATH)..."
        exec "$PYTHON_BIN" "$TEST_SUITE_PATH"
        ;;
    start|--start|-d|--daemon)
        shift
        start_daemon "$@"
        exit 0
        ;;
    stop|--stop|kill|shutdown)
        stop_daemon
        exit 0
        ;;
    restart|--restart)
        shift
        stop_daemon
        sleep 1
        start_daemon "$@"
        exit 0
        ;;
    status|--status)
        show_status
        exit 0
        ;;
    logs|--logs|tail)
        echo "📜 Tailing $LOG_FILE (Press Ctrl+C to exit log viewer — bots will keep running)..."
        touch "$LOG_FILE"
        exec tail -n 60 -f "$LOG_FILE"
        ;;
    fg|--fg|foreground)
        shift
        stop_daemon
        echo "🖥️  Starting Orchestrator in Foreground Console Mode (Ctrl+C to stop)..."
        exec "$PYTHON_BIN" -u "$ORCHESTRATOR_PATH" --host 0.0.0.0 "$@"
        ;;
esac

# If custom flags like --autostart or --tunnel were passed directly, launch background daemon with them
if [ $# -gt 0 ]; then
    start_daemon "$@"
    exit 0
fi

# ------------------------------------------------------------------------------
# INTERACTIVESSH / TERMINAL CONTROL MENU
# ------------------------------------------------------------------------------
RUNNING_PID="$(get_running_pid || true)"

if [ -n "$RUNNING_PID" ]; then
    echo "=============================================================================="
    echo "🤖 DISCORD MULTI-AGENT CONTROL PLANE — RUNNING IN BACKGROUND (PID: $RUNNING_PID)"
    print_connection_info
    echo "=============================================================================="
    echo "  [1] 🛑 Shut Down Background Daemon & All Bots"
    echo "  [2] 🔄 Restart Background Daemon (Standby Mode)"
    echo "  [3] ⚡ Restart Background Daemon + Auto-Start All Bots (--autostart)"
    echo "  [4] 📜 View Live Console Logs (tail -f — Ctrl+C exits logs safely)"
    echo "  [5] 🚪 Exit Menu (Keep Daemon Running in Background)"
    echo "=============================================================================="
    read -r -p "Select an option [1-5]: " choice
    case "${choice:-5}" in
        1) stop_daemon ;;
        2) stop_daemon; sleep 1; start_daemon ;;
        3) stop_daemon; sleep 1; start_daemon --autostart ;;
        4) touch "$LOG_FILE"; exec tail -n 60 -f "$LOG_FILE" ;;
        *) echo "👍 Leaving Orchestrator Daemon running in the background (PID: $RUNNING_PID)." ;;
    esac
else
    echo "=============================================================================="
    echo "🤖 DISCORD MULTI-AGENT CONTROL PLANE — CURRENTLY STOPPED"
    echo "=============================================================================="
    echo "  [1] 🚀 Launch in Background (SSH-Safe Daemon — Standby Mode in Web-UI)"
    echo "  [2] ⚡ Launch in Background + Auto-Start All Bots (--autostart)"
    echo "  [3] 🌐 Launch in Background + Public Cloudflare HTTPS Tunnel (--tunnel)"
    echo "  [4] 🖥️  Launch in Foreground (Interactive Terminal Mode)"
    echo "  [5] 🧪 Run Offline Unit Test Suite (--test)"
    echo "  [6] 🚪 Exit"
    echo "=============================================================================="
    read -r -p "Select an option [1-6, default=1]: " choice
    case "${choice:-1}" in
        1|"") start_daemon ;;
        2) start_daemon --autostart ;;
        3) start_daemon --tunnel ;;
        4) exec "$PYTHON_BIN" -u "$ORCHESTRATOR_PATH" --host 0.0.0.0 ;;
        5) exec "$PYTHON_BIN" "$TEST_SUITE_PATH" ;;
        *) echo "Exited." ;;
    esac
fi
