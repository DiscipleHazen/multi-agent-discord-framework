#!/usr/bin/env bash
# ==============================================================================
# DISCORD MULTI-AGENT AI FRAMEWORK v4.0 — UNIFIED ORCHESTRATOR & WEB-UI LAUNCHER
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORCHESTRATOR_PATH="bots/bot_info/python_modules/orchestrator.py"
TEST_SUITE_PATH="bots/bot_info/python_modules/test_framework.py"

if [ ! -f "$ORCHESTRATOR_PATH" ] && [ -f "orchestrator.py" ]; then
    ORCHESTRATOR_PATH="orchestrator.py"
    TEST_SUITE_PATH="test_framework.py"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(dirname "$ORCHESTRATOR_PATH"):${PYTHONPATH}"

if [ "${1:-}" = "--test" ] || [ "${1:-}" = "test" ]; then
    echo "🧪 Running Offline Unit Test Suite ($TEST_SUITE_PATH)..."
    python3 "$TEST_SUITE_PATH"
    exit 0
fi

echo "=============================================================================="
echo "🚀 Launching Discord Multi-Agent Supervisor & Real-Time Web-UI Control Plane"
echo "   • Orchestrator: $ORCHESTRATOR_PATH"
echo "   • Tip: Pass '--host 0.0.0.0' to access the Web-UI from any LAN device,"
echo "          or use VS Code SSH Port-Forwarding on http://127.0.0.1:8765"
echo "=============================================================================="

exec python3 -u "$ORCHESTRATOR_PATH" "$@"
