import os
import re
import ast
import sys
import json
import time
import math
import glob
import socket
import shutil
import secrets
import random
import signal
import subprocess
import ctypes
import asyncio
import argparse
import py_compile
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

from core_system import (
    load_json_file,
    update_json_file,
    _is_pid_alive,
    get_all_model_telemetry,
)
from subconscious_engine import (
    sanitize_or_reject_sensitive_pii,
    pack_salience_fact,
    unpack_salience_fact,
    format_fact_for_display,
    compute_ebbinghaus_retention,
)
from memory_and_context import (
    _extract_base_username,
    _extract_saved_rapport_state_full,
    _get_disposition_tier_label,
    _derive_relationship_archetype,
    RAPPORT_ENTRY_REGEX,
    consolidate_and_append_fact,
)

# ==============================================================================
# ORCHESTRATOR CONSTANTS, FINOPS PRICING MATRIX & PORT AUTO-SCANNER
# ==============================================================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT_START = 8765
DEFAULT_PORT_END = 8799

# DeepInfra approximate per-1M token / per-call pricing for FinOps Estimator (#5)
MODEL_FINOPS_PRICING = {
    "Qwen3-235B": {"in_per_1m": 0.18, "out_per_1m": 0.54},
    "Qwen3-30B-A3B": {"in_per_1m": 0.06, "out_per_1m": 0.18},
    "Qwen3-VL-235B": {"in_per_1m": 0.25, "out_per_1m": 0.75},
    "Qwen2.5-VL-32B": {"in_per_1m": 0.12, "out_per_1m": 0.36},
    "Llama-3.3-70B": {"in_per_1m": 0.12, "out_per_1m": 0.30},
    "FLUX-1-schnell": {"per_call": 0.0015},
    "MiMo-V2.5-tts": {"per_call": 0.0020},
    "whisper-large-v3-turbo": {"per_call": 0.0005},
}

CORE_FRAMEWORK_MODULES = frozenset({
    "config.py",
    "core_system.py",
    "text_and_parsers.py",
    "subconscious_engine.py",
    "memory_and_context.py",
    "media_engine.py",
    "ai_services.py",
    "bot_discovery.py",
    "commands_and_embeds.py",
    "main_bot.py",
    "orchestrator.py",
    "test_framework.py",
})


def find_free_localhost_port(
    host: str = DEFAULT_HOST,
    start_port: int = DEFAULT_PORT_START,
    end_port: int = DEFAULT_PORT_END,
) -> int:
    """
    Step 4.1.1 — Zero-Collision Localhost Port Auto-Scanner (127.0.0.1:8765 -> 8799):
    Avoids common web-dev ports (3000, 5173, 5500, 8000, 8080) and probes 8765..8799
    in <1ms to bind the first available port.
    """
    for port in range(int(start_port), int(end_port) + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    # Fallback: let OS pick an ephemeral free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def validate_python_file_ast(filepath: str) -> tuple[bool, str]:
    """
    Orchestrator Upgrade #2 — AST & Bytecode Validator for Zero-Downtime Hot-Reload:
    Parses `filepath` via `ast.parse()` and `py_compile` before any rolling restart.
    Returns `(True, "OK")` or `(False, error_message)`.
    """
    if not filepath or not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            src = f.read()
        ast.parse(src, filename=filepath)
        py_compile.compile(filepath, doraise=True)
        return True, "OK"
    except SyntaxError as syn_err:
        return False, f"SyntaxError in {os.path.basename(filepath)}:{syn_err.lineno}:{syn_err.offset}: {syn_err.msg}"
    except Exception as e:
        return False, f"Compile error in {os.path.basename(filepath)}: {e}"


def discover_bot_launcher_scripts(search_dir: str = ".") -> list[str]:
    """
    Automatically discovers persona launcher scripts (e.g. `guild_assistant.py`, `cat_bot.py`) across `bots/`, `search_dir`, and parent directories.
    """
    layout = resolve_project_layout(search_dir)
    search_folders = list(dict.fromkeys([
        layout["bots_dir"],
        os.path.abspath(search_dir),
        layout["workspace_root"],
    ]))
    found = []
    seen_abs = set()
    for folder in search_folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.py"))):
            abs_p = os.path.abspath(path)
            if abs_p in seen_abs:
                continue
            bname = os.path.basename(abs_p)
            if (
                bname in CORE_FRAMEWORK_MODULES
                or bname.startswith("_")
                or bname.startswith("test_")
                or "template" in bname.lower()
                or bname == "bot_editor.py"
            ):
                continue
            try:
                with open(abs_p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(25000)
                if "run_bot(" in content:
                    seen_abs.add(abs_p)
                    found.append(abs_p)
            except Exception:
                continue
    return found

def resolve_project_layout(anchor_dir: str | None = None) -> dict[str, str]:
    """
    Automatically detects the `discord_AI_app` directory topology whether
    `orchestrator.py` is placed inside `bots/bot_info/python_modules/`, `bots/`,
    or the workspace root:
      - `python_modules_dir`: `.../bots/bot_info/python_modules`
      - `bot_info_dir`:       `.../bots/bot_info`
      - `bots_dir`:           `.../bots`
      - `workspace_root`:     `.../`
      - `short_term_dir`:     `.../bots/bot_info/short_term_memory`
      - `long_term_dir`:      `.../bots/bot_info/long_term_memory`
      - `config_dir`:         `.../bots/bot_info/config_files`
      - `logs_dir`:           `.../bots/bot_info/bot_logs`
      - `env_file`:           `.../bots/bot_info/.env`
    Falls back gracefully to flat directory paths if `bot_info/` is not present
    (e.g. during standalone unit tests).
    """
    here = os.path.abspath(anchor_dir or os.path.dirname(os.path.abspath(__file__)))
    bname = os.path.basename(here)

    if bname == "python_modules":
        python_modules_dir = here
        bot_info_dir = os.path.dirname(here)
        bots_dir = os.path.dirname(bot_info_dir)
        workspace_root = os.path.dirname(bots_dir)
    elif os.path.isdir(os.path.join(here, "bots", "bot_info", "python_modules")):
        workspace_root = here
        bots_dir = os.path.join(here, "bots")
        bot_info_dir = os.path.join(bots_dir, "bot_info")
        python_modules_dir = os.path.join(bot_info_dir, "python_modules")
    elif os.path.isdir(os.path.join(here, "bot_info", "python_modules")):
        bots_dir = here
        workspace_root = os.path.dirname(here)
        bot_info_dir = os.path.join(here, "bot_info")
        python_modules_dir = os.path.join(bot_info_dir, "python_modules")
    else:
        # Flat directory fallback
        workspace_root = here
        bots_dir = here
        bot_info_dir = here
        python_modules_dir = here

    short_term_dir = os.path.join(bot_info_dir, "short_term_memory")
    long_term_dir = os.path.join(bot_info_dir, "long_term_memory")
    config_dir = os.path.join(bot_info_dir, "config_files")
    logs_dir = os.path.join(bot_info_dir, "bot_logs")

    # Use structured subfolders if `bot_info` exists, else flat fallback
    has_bot_info = os.path.basename(bot_info_dir) == "bot_info" or os.path.isdir(short_term_dir)
    if has_bot_info:
        for d in (short_term_dir, long_term_dir, config_dir, logs_dir):
            os.makedirs(d, exist_ok=True)
        shared_cache_file = os.path.join(short_term_dir, "shared_image_cache.json")
        chat_memory_file = os.path.join(short_term_dir, "chat_memory.json")
        user_profiles_file = os.path.join(long_term_dir, "user_profiles.json")
        long_term_memory_file = os.path.join(long_term_dir, "long_term_memory.json")
        general_memory_file = os.path.join(long_term_dir, "general_knowledge.json")
        channel_config_file = os.path.join(config_dir, "channel_configs.json")
        env_file = os.path.join(bot_info_dir, ".env")
    else:
        shared_cache_file = os.path.join(here, "shared_image_cache.json")
        chat_memory_file = os.path.join(here, "chat_memory.json")
        user_profiles_file = os.path.join(here, "user_profiles.json")
        long_term_memory_file = os.path.join(here, "long_term_memory.json")
        general_memory_file = os.path.join(here, "general_knowledge.json")
        channel_config_file = os.path.join(here, "channel_configs.json")
        env_file = os.path.join(here, ".env")

    return {
        "workspace_root": workspace_root,
        "bots_dir": bots_dir,
        "bot_info_dir": bot_info_dir,
        "python_modules_dir": python_modules_dir,
        "logs_dir": logs_dir if has_bot_info else here,
        "shared_cache_file": shared_cache_file,
        "chat_memory_file": chat_memory_file,
        "user_profiles_file": user_profiles_file,
        "long_term_memory_file": long_term_memory_file,
        "general_memory_file": general_memory_file,
        "channel_config_file": channel_config_file,
        "env_file": env_file,
    }


def inspect_bot_script_config(script_path: str) -> dict:
    """
    Bot Studio Inspector (replaces terminal `bot_editor.py`):
    Reads a persona launcher `.py` script (`guild_assistant.py`, `cat_bot.py`, etc.) using AST
    to extract both module-level `CONSTANTS` and `run_bot(...)` arguments for the Web UI.
    """
    if not script_path or not os.path.exists(script_path):
        return {}
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return {}

    info = {
        "name": os.path.splitext(os.path.basename(script_path))[0],
        "filename": os.path.basename(script_path),
        "path": script_path,
        "token_env_var": "",
        "text_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "vision_model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "image_gen_model": "black-forest-labs/FLUX-1-schnell",
        "voice_model": "XiaomiMiMo/MiMo-V2.5-tts-voicedesign",
        "tts_voice": "",
        "temperature": 0.75,
        "top_p": 0.88,
        "output_length": 550,
        "context_window": 50,
        "emoji_context_length": 12,
        "chance_to_respond": 0.01,
        "random_emoji_chance": 0.35,
        "default_random_interval": 0,
        "enable_vision": True,
        "enable_image_gen": True,
        "enable_voice": False,
        "enable_gifs": True,
        "keywords": [],
        "source_code": src,
    }
    const_to_field = {
        "TEXT_MODEL": "text_model",
        "VISION_MODEL": "vision_model",
        "IMAGE_GEN_MODEL": "image_gen_model",
        "VOICE_MODEL": "voice_model",
        "TTS_VOICE": "tts_voice",
        "ENABLE_VISION": "enable_vision",
        "ENABLE_GIFS": "enable_gifs",
        "ENABLE_IMAGE_GEN": "enable_image_gen",
        "ENABLE_VOICE": "enable_voice",
        "TEMPERATURE": "temperature",
        "TOP_P": "top_p",
        "OUTPUT_LENGTH": "output_length",
        "CONTEXT_WINDOW": "context_window",
        "EMOJI_CONTEXT_LENGTH": "emoji_context_length",
        "CHANCE_TO_RESPOND": "chance_to_respond",
        "RANDOM_EMOJI_CHANCE": "random_emoji_chance",
        "DEFAULT_RANDOM_INTERVAL": "default_random_interval",
        "TRIGGER_KEYWORDS": "keywords",
    }
    m_tok = re.search(r'TOKEN\s*=\s*os\.getenv\(\s*["\']([^"\']+)["\']\s*\)', src)
    if m_tok:
        info["token_env_var"] = m_tok.group(1)

    try:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                t_name = node.targets[0].id
                if t_name in const_to_field:
                    f_key = const_to_field[t_name]
                    if isinstance(node.value, ast.Constant):
                        info[f_key] = node.value.value
                    elif f_key == "keywords" and isinstance(node.value, (ast.List, ast.Tuple)):
                        info["keywords"] = [
                            elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_bot":
                for kw in node.keywords:
                    if kw.arg in info and isinstance(kw.value, ast.Constant):
                        info[kw.arg] = kw.value.value
                    elif kw.arg == "keywords" and isinstance(kw.value, (ast.List, ast.Tuple)):
                        info["keywords"] = [
                            elt.value for elt in kw.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    except Exception:
        pass
    return info

# ==============================================================================
# .ENV MANAGER, PRE-LAUNCH ENV CHECKER, BOT CREATOR & STRUCTURED CONFIG EDITOR
# (100% Feature Parity with `manage_bots.bat`, `manage_bots.sh` & `bot_editor.py`)
# ==============================================================================
_ENV_TOKEN_VAR_REGEX = re.compile(r'TOKEN\s*=\s*os\.getenv\(\s*["\']([^"\']+)["\']\s*\)')
_ENV_API_KEY_VAR_REGEX = re.compile(r'API_KEY\s*=\s*os\.getenv\(\s*["\']([^"\']+)["\']\s*\)')
_ENV_GIPHY_VAR_REGEX = re.compile(r'GIPHY_KEY\s*=\s*os\.getenv\(\s*["\']([^"\']+)["\']\s*\)')
_ENABLE_GIFS_TRUE_REGEX = re.compile(r'ENABLE_GIFS\s*=\s*True\b')


def get_lan_ip_address() -> str:
    """Detects the machine's primary LAN IPv4 address (e.g. 192.168.x.x) in <1ms."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def ensure_webui_admin_token(env_path: str) -> str:
    """
    Ensures `WEBUI_ADMIN_TOKEN` exists in `bots/bot_info/.env`.
    If missing, generates a cryptographically random 24-char token (`secrets.token_hex(12)`)
    and saves it to `.env` so remote/LAN Discord admins can authenticate securely.
    """
    env_map = read_env_file_dict(env_path)
    existing = (env_map.get("WEBUI_ADMIN_TOKEN") or os.getenv("WEBUI_ADMIN_TOKEN", "")).strip()
    if existing and len(existing) >= 8:
        os.environ["WEBUI_ADMIN_TOKEN"] = existing
        return existing
    new_tok = secrets.token_hex(12)
    upsert_env_variable(env_path, "WEBUI_ADMIN_TOKEN", new_tok)
    os.environ["WEBUI_ADMIN_TOKEN"] = new_tok
    return new_tok


def is_webui_request_authorized(
    peer_ip: str,
    headers: dict[str, str],
    query_params: dict[str, list[str]],
    admin_token: str,
) -> bool:
    """
    Verifies whether an HTTP/SSE request to `/api/*` is authorized:
    1. If the caller provides a valid `X-Admin-Token` header or `?token=...` query parameter
       matching `admin_token` (constant-time `secrets.compare_digest`), returns True.
    2. If the request originates from `127.0.0.1` / `::1` and `WEBUI_STRICT_LOCAL_AUTH`
       is not enabled, allows local machine access automatically.
    """
    if not admin_token:
        return True
    provided = (
        headers.get("x-admin-token", "").strip()
        or (query_params.get("token", [""])[0] or "").strip()
    )
    if provided and secrets.compare_digest(provided, admin_token):
        return True
    strict_local = os.getenv("WEBUI_STRICT_LOCAL_AUTH", "0").strip() in ("1", "true", "yes")
    if not strict_local and peer_ip in ("127.0.0.1", "::1", "localhost"):
        return True
    return False


def read_env_file_dict(env_path: str) -> dict[str, str]:
    """Reads `bots/bot_info/.env` into a clean `{key: value}` dictionary."""
    out: dict[str, str] = {}
    if not env_path or not os.path.exists(env_path):
        return out
    try:
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_ln in f:
                ln = raw_ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    out[k] = v
    except Exception:
        pass
    return out


def upsert_env_variable(env_path: str, key: str, value: str) -> bool:
    """
    Atomically updates or appends `key=value` inside `bots/bot_info/.env`
    (matching the exact behavior of `:ADD_KEY`, `:SET_HISTORY`, and `:CREATE_BOT`).
    """
    clean_k = re.sub(r"[^A-Za-z0-9_]", "", (key or "").strip())
    if not clean_k or not env_path:
        return False
    clean_v = str(value or "").strip().replace("\r", "").replace("\n", "")
    os.makedirs(os.path.dirname(os.path.abspath(env_path)), exist_ok=True)

    lines = []
    found = False
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except Exception:
            lines = []

    new_lines = []
    for ln in lines:
        stripped = ln.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k_part = stripped.split("=", 1)[0].strip()
            if k_part == clean_k:
                new_lines.append(f"{clean_k}={clean_v}")
                found = True
                continue
        new_lines.append(ln)

    if not found:
        new_lines.append(f"{clean_k}={clean_v}")

    with open(env_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")
    return True


def mask_env_secret(key: str, val: str) -> str:
    """Masks sensitive API tokens for safe Web-UI display while showing `MAX_HISTORY_LENGTH` in full."""
    if key == "MAX_HISTORY_LENGTH":
        return val
    if not val:
        return "(empty)"
    if len(val) <= 8:
        return "••••••••"
    return f"{val[:4]}••••••••{val[-4:]}"


def verify_bot_script_env_keys(script_path: str, env_path: str) -> tuple[bool, list[str]]:
    """
    Pre-Launch `.env` Key Validator (ported from `manage_bots.bat` & `manage_bots.sh`):
    Checks `script_path` for `TOKEN = os.getenv(...)`, `API_KEY = os.getenv(...)`,
    and (when `ENABLE_GIFS = True`) `GIPHY_KEY = os.getenv(...)`, returning
    `(all_present, list_of_missing_env_keys)`.
    """
    if not script_path or not os.path.exists(script_path):
        return False, ["script_not_found"]
    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            py_txt = f.read()
    except Exception:
        return True, []

    env_map = read_env_file_dict(env_path)
    # Also allow keys present in current OS environment
    missing = []

    def _check_var(var_name: str):
        if not var_name:
            return
        val = env_map.get(var_name) or os.getenv(var_name, "")
        if not val or not str(val).strip():
            if var_name not in missing:
                missing.append(var_name)

    m_tok = _ENV_TOKEN_VAR_REGEX.search(py_txt)
    if m_tok:
        _check_var(m_tok.group(1))

    m_api = _ENV_API_KEY_VAR_REGEX.search(py_txt)
    if m_api:
        _check_var(m_api.group(1))

    if _ENABLE_GIFS_TRUE_REGEX.search(py_txt):
        m_gif = _ENV_GIPHY_VAR_REGEX.search(py_txt)
        if m_gif:
            _check_var(m_gif.group(1))

    return (len(missing) == 0), missing


def generate_new_bot_script(
    *,
    bots_dir: str,
    env_path: str,
    bot_name: str,
    discord_token: str,
    system_prompt_identity: str,
    text_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    vision_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
    image_gen_model: str = "black-forest-labs/FLUX-1-schnell",
    voice_model: str = "XiaomiMiMo/MiMo-V2.5-tts-voicedesign",
    tts_voice: str = "",
    enable_vision: bool = True,
    enable_gifs: bool = True,
    enable_image_gen: bool = True,
    enable_voice: bool = False,
    temperature: float = 0.75,
    top_p: float = 0.88,
    output_length: int = 550,
    context_window: int = 50,
    emoji_context_length: int = 12,
    chance_to_respond: float = 0.01,
    random_emoji_chance: float = 0.35,
    default_random_interval: int = 0,
    keywords: list[str] | None = None,
) -> tuple[bool, str, str]:
    """
    Ported from `:CREATE_BOT` in `manage_bots.bat` / `manage_bots.sh`:
    1. Normalizes `bot_name` to `safe_name` (`a-z0-9_`) and saves `<SAFE_NAME>_DISCORD_TOKEN`
       to `bots/bot_info/.env` (if `discord_token` is provided).
    2. Generates `bots/<safe_name>.py` with per-bot memory/config paths and AST-validates it.
    Returns `(success, safe_name, script_path_or_error)`.
    """
    safe_name = re.sub(r"[^a-z0-9_]", "", (bot_name or "").strip().lower().replace(" ", "_"))
    if not safe_name:
        return False, "", "Invalid bot name."

    tok_key = f"{safe_name.upper()}_DISCORD_TOKEN"
    if discord_token and discord_token.strip():
        upsert_env_variable(env_path, tok_key, discord_token.strip())

    kw_list = [k.strip().lower() for k in (keywords or [safe_name]) if k and k.strip()]
    if not kw_list:
        kw_list = [safe_name]

    clean_identity = (system_prompt_identity or f"You are {bot_name}, a witty Discord personality.").strip()
    clean_identity_escaped = clean_identity.replace('"""', '\\"\\"\\"')

    script_content = f'''import sys, os
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'bot_info', 'python_modules'))
from main_bot import run_bot
from bot_discovery import build_universal_media_instructions
load_dotenv(os.path.join(current_dir, 'bot_info', '.env'))

def get_system_prompt(enable_image_gen: bool, enable_voice: bool, enable_gifs: bool = True) -> dict:
    media_section = build_universal_media_instructions(
        enable_image_gen=enable_image_gen,
        enable_voice=enable_voice,
        enable_gifs=enable_gifs
    )
    return {{
        'role': 'system',
        'content': f"""{clean_identity_escaped}

### Strict Formatting & Spacing Rules
- Write your response as a single compact line or tight paragraph with NO blank lines between sentences.
- Wrap physical actions/emotes in single asterisks like *sighs* or *grins*, never square brackets.

### Memory & Smart Learning (3-Tier Protocol)
- User Personal Facts (self-stated only): [LEARN_USER: username | concise objective fact]
- Shared General Knowledge (universal trivia/science): [LEARN_GENERAL: concise objective fact]
- Server-Specific Lore (Discord server events/jokes only): [LEARN: concise objective fact]

{{media_section}}"""
    }}

# --- BOT CONFIGURATION CONSTANTS ---
TOKEN = os.getenv('{tok_key}')
API_KEY = os.getenv('DEEPINFRA_TOKEN')
GIPHY_KEY = os.getenv('GIPHY_TOKEN')

CHAT_MEMORY_FILE = os.path.join(current_dir, 'bot_info', 'short_term_memory', 'chat_memory.json')
SHARED_IMAGE_CACHE_FILE = os.path.join(current_dir, 'bot_info', 'short_term_memory', 'shared_image_cache.json')
LONG_TERM_MEMORY_FILE = os.path.join(current_dir, 'bot_info', 'long_term_memory', '{safe_name}_long_term_memory.json')
USER_PROFILES_FILE = os.path.join(current_dir, 'bot_info', 'long_term_memory', '{safe_name}_user_profiles.json')
CHANNEL_CONFIG_FILE = os.path.join(current_dir, 'bot_info', 'config_files', '{safe_name}_channel_configs.json')

TEXT_MODEL = {repr(str(text_model))}
VISION_MODEL = {repr(str(vision_model))}
IMAGE_GEN_MODEL = {repr(str(image_gen_model))}
VOICE_MODEL = {repr(str(voice_model))}
TTS_VOICE = {repr(str(tts_voice))}

ENABLE_VISION = {bool(enable_vision)}
ENABLE_GIFS = {bool(enable_gifs)}
ENABLE_IMAGE_GEN = {bool(enable_image_gen)}
ENABLE_VOICE = {bool(enable_voice)}

TEMPERATURE = {float(temperature)}
TOP_P = {float(top_p)}
OUTPUT_LENGTH = {int(output_length)}
CONTEXT_WINDOW = {int(context_window)}
EMOJI_CONTEXT_LENGTH = {int(emoji_context_length)}
CHANCE_TO_RESPOND = {float(chance_to_respond)}
RANDOM_EMOJI_CHANCE = {float(random_emoji_chance)}
DEFAULT_RANDOM_INTERVAL = {int(default_random_interval)}
TRIGGER_KEYWORDS = {repr(kw_list)}

if __name__ == '__main__':
    system_prompt = get_system_prompt(ENABLE_IMAGE_GEN, ENABLE_VOICE, ENABLE_GIFS)
    run_bot(
        token=TOKEN, api_key=API_KEY, giphy_key=GIPHY_KEY, system_prompt=system_prompt,
        chat_memory_file=CHAT_MEMORY_FILE, shared_image_cache_file=SHARED_IMAGE_CACHE_FILE,
        long_term_memory_file=LONG_TERM_MEMORY_FILE, user_profiles_file=USER_PROFILES_FILE,
        channel_config_file=CHANNEL_CONFIG_FILE, text_model=TEXT_MODEL, vision_model=VISION_MODEL,
        image_gen_model=IMAGE_GEN_MODEL, voice_model=VOICE_MODEL, tts_voice=TTS_VOICE,
        enable_vision=ENABLE_VISION, enable_gifs=ENABLE_GIFS, enable_image_gen=ENABLE_IMAGE_GEN,
        enable_voice=ENABLE_VOICE, temperature=TEMPERATURE, top_p=TOP_P,
        output_length=OUTPUT_LENGTH, context_window=CONTEXT_WINDOW,
        emoji_context_length=EMOJI_CONTEXT_LENGTH, chance_to_respond=CHANCE_TO_RESPOND,
        random_emoji_chance=RANDOM_EMOJI_CHANCE, default_random_interval=DEFAULT_RANDOM_INTERVAL,
        keywords=TRIGGER_KEYWORDS
    )
'''
    os.makedirs(bots_dir, exist_ok=True)
    target_py = os.path.join(bots_dir, f"{safe_name}.py")
    try:
        ast.parse(script_content, filename=target_py)
        with open(target_py, "w", encoding="utf-8", newline="\n") as f:
            f.write(script_content)
        py_compile.compile(target_py, doraise=True)
        return True, safe_name, target_py
    except Exception as e:
        return False, safe_name, str(e)


def update_bot_script_constants(script_path: str, updates: dict) -> tuple[bool, str]:
    """
    Visual Config Editor Updater (replaces terminal `bot_editor.py`):
    Updates module-level constants (`TEXT_MODEL`, `VISION_MODEL`, `IMAGE_GEN_MODEL`,
    `VOICE_MODEL`, `TTS_VOICE`, `ENABLE_VISION`, `ENABLE_GIFS`, `ENABLE_IMAGE_GEN`,
    `ENABLE_VOICE`, `TEMPERATURE`, `TOP_P`, `OUTPUT_LENGTH`, `CONTEXT_WINDOW`,
    `EMOJI_CONTEXT_LENGTH`, `CHANCE_TO_RESPOND`, `RANDOM_EMOJI_CHANCE`,
    `DEFAULT_RANDOM_INTERVAL`, `TRIGGER_KEYWORDS`) inside a bot script and validates
    via `ast.parse()` before saving.
    """
    if not script_path or not os.path.exists(script_path):
        return False, "Bot script not found."
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return False, str(e)

    const_map = {
        "text_model": ("TEXT_MODEL", str),
        "vision_model": ("VISION_MODEL", str),
        "image_gen_model": ("IMAGE_GEN_MODEL", str),
        "voice_model": ("VOICE_MODEL", str),
        "tts_voice": ("TTS_VOICE", str),
        "enable_vision": ("ENABLE_VISION", bool),
        "enable_gifs": ("ENABLE_GIFS", bool),
        "enable_image_gen": ("ENABLE_IMAGE_GEN", bool),
        "enable_voice": ("ENABLE_VOICE", bool),
        "temperature": ("TEMPERATURE", float),
        "top_p": ("TOP_P", float),
        "output_length": ("OUTPUT_LENGTH", int),
        "context_window": ("CONTEXT_WINDOW", int),
        "emoji_context_length": ("EMOJI_CONTEXT_LENGTH", int),
        "chance_to_respond": ("CHANCE_TO_RESPOND", float),
        "random_emoji_chance": ("RANDOM_EMOJI_CHANCE", float),
        "default_random_interval": ("DEFAULT_RANDOM_INTERVAL", int),
    }

    for field_k, (const_name, typ) in const_map.items():
        if field_k in updates and updates[field_k] is not None:
            try:
                val = typ(updates[field_k])
                rep_val = repr(val)
                src = re.sub(
                    rf"(?m)^{const_name}\s*=\s*.*$",
                    f"{const_name} = {rep_val}",
                    src,
                )
            except Exception:
                continue

    if "keywords" in updates and isinstance(updates["keywords"], list):
        kw_clean = [str(x).strip() for x in updates["keywords"] if str(x).strip()]
        src = re.sub(
            r"(?m)^TRIGGER_KEYWORDS\s*=\s*.*$",
            f"TRIGGER_KEYWORDS = {repr(kw_clean)}",
            src,
        )

    try:
        ast.parse(src, filename=script_path)
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        py_compile.compile(script_path, doraise=True)
        return True, "OK"
    except SyntaxError as syn_e:
        return False, f"SyntaxError line {syn_e.lineno}: {syn_e.msg}"
    except Exception as e:
        return False, str(e)




# ==============================================================================
# UPGRADE #1: LEAKY TOKEN-BUCKET RESTART LIMITER & CRASH-LOOP QUARANTINE
# ==============================================================================
class LeakyBucketRestartLimiter:
    """
    Orchestrator Upgrade #1 — Leaky Token-Bucket Restart Guard & Crash-Loop Quarantine:
      B_t = min(B_max, B_{t-1} + r * delta_t) - 1
    with full-jitter exponential backoff:
      t_sleep = min(300.0, (2 ** k) + U(0.0, 1.5))
    Trips into `QUARANTINED` status if >= 4 crashes occur within 90 seconds or tokens
    drop below 0, protecting Discord's gateway from token/syntax crash loops.
    """
    def __init__(self, max_tokens: float = 4.0, refill_rate_per_sec: float = 1.0 / 45.0):
        self.max_tokens = float(max_tokens)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(max_tokens)
        self.last_update_ts = time.monotonic()
        self.crash_timestamps: list[float] = []
        self.consecutive_crashes = 0
        self.quarantined = False

    def record_crash_and_get_backoff(self, now_mono: float | None = None) -> tuple[bool, float]:
        """
        Records a process crash and returns `(is_allowed_to_restart, backoff_seconds)`.
        If `is_allowed_to_restart` is False, the bot enters `QUARANTINED` state.
        """
        now_m = now_mono if now_mono is not None else time.monotonic()
        dt = max(0.0, now_m - self.last_update_ts)
        self.last_update_ts = now_m
        self.tokens = min(self.max_tokens, self.tokens + self.refill_rate * dt)

        # Sliding 90s crash window check
        self.crash_timestamps = [ts for ts in self.crash_timestamps if (now_m - ts) <= 90.0]
        self.crash_timestamps.append(now_m)
        self.consecutive_crashes += 1

        if len(self.crash_timestamps) >= 4 or self.tokens < 1.0:
            self.quarantined = True
            return False, 0.0

        self.tokens -= 1.0
        k = min(8, self.consecutive_crashes)
        backoff_sec = min(300.0, float(2 ** k) + random.uniform(0.0, 1.5))
        return True, round(backoff_sec, 2)

    def reset_quarantine(self):
        self.tokens = self.max_tokens
        self.last_update_ts = time.monotonic()
        self.crash_timestamps.clear()
        self.consecutive_crashes = 0
        self.quarantined = False


# ==============================================================================
# UPGRADE #4: ORPHANED FILE-LOCK & TEMP-FILE REAPER
# ==============================================================================
def kill_orphaned_bot_processes(script_names: list[str], exclude_pids: set[int] | None = None) -> list[int]:
    """
    Hunts down and terminates any detached/orphaned `python` OS processes (left over
    from legacy `manage_bots.bat` / `manage_bots.sh` background launches) that are
    running any of `script_names` (e.g., `['guild_assistant.py', 'cat_bot.py']`).
    Works on both Windows (WMI / CIM `Win32_Process`) and Linux/POSIX (`/proc/<pid>/cmdline`).
    Returns the list of killed PIDs.
    """
    killed_pids: list[int] = []
    safe_excludes = set(exclude_pids or set()) | {os.getpid()}
    target_basenames = [os.path.basename(s).lower() for s in script_names if s]
    if not target_basenames:
        return killed_pids

    if os.name == "nt":
        try:
            # Fast single PowerShell CIM query to find and kill detached python.exe/pythonw.exe processes
            names_regex = "|".join(re.escape(b) for b in target_basenames)
            excl_str = ",".join(str(p) for p in safe_excludes)
            ps_cmd = (
                f"$excl = @({excl_str}); "
                "$procs = Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='pythonw.exe'\" -ErrorAction SilentlyContinue | "
                f"Where-Object {{ $_.CommandLine -match '({names_regex})' -and ($excl -notcontains $_.ProcessId) }}; "
                "foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Output $p.ProcessId }"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=8,
            )
            for ln in (res.stdout or "").splitlines():
                ln_s = ln.strip()
                if ln_s.isdigit():
                    killed_pids.append(int(ln_s))
        except Exception:
            pass
    else:
        # Pure-Python /proc/<pid>/cmdline scanner for Linux/POSIX
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid in safe_excludes:
                    continue
                cmdline_path = f"/proc/{pid}/cmdline"
                try:
                    with open(cmdline_path, "rb") as cf:
                        raw_cmd = cf.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore").lower()
                except Exception:
                    continue
                if "python" in raw_cmd and any(b in raw_cmd for b in target_basenames):
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed_pids.append(pid)
                    except Exception:
                        pass
        except Exception:
            pass

    return killed_pids


def reap_orphaned_locks_and_temps(search_dirs: list[str], active_child_pids: set[int] | None = None) -> int:
    """
    Orchestrator Upgrade #4 — Orphaned `.lock` & `.tmp` File Reaper:
    Sweeps `search_dirs` on startup and every 60s, evicting any `.lock` or `.tmp` file
    owned by a dead PID (or older than 30s for `.tmp` files). Returns count of reaped files.
    """
    reaped = 0
    now_ts = time.time()
    active_pids = set(active_child_pids or set()) | {os.getpid()}

    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for pat in ("*.lock", "*.tmp"):
            for fpath in glob.glob(os.path.join(d, pat)):
                try:
                    age = now_ts - os.path.getmtime(fpath)
                    if fpath.endswith(".tmp") and age > 30.0:
                        os.remove(fpath)
                        reaped += 1
                        continue
                    if fpath.endswith(".lock"):
                        if age > 10.0:
                            os.remove(fpath)
                            reaped += 1
                            continue
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as lf:
                            raw = lf.read(64).strip()
                        if "|" in raw:
                            pid_s = raw.split("|", 1)[0].strip()
                            if pid_s.isdigit():
                                lock_pid = int(pid_s)
                                if lock_pid not in active_pids and not _is_pid_alive(lock_pid):
                                    os.remove(fpath)
                                    reaped += 1
                except Exception:
                    continue
    return reaped


# ==============================================================================
# UPGRADE #6: ZERO-DEPENDENCY OS RAM & CPU PROCESS PROFILER (WIN32 & POSIX)
# ==============================================================================
_CPU_TIME_SNAPSHOTS: dict[int, tuple[float, float]] = {}  # {pid: (wall_mono, proc_cpu_seconds)}


def query_process_ram_and_cpu(pid: int) -> dict[str, float]:
    """
    Orchestrator Upgrade #6 — Zero-Dependency OS RAM (MB) & CPU (%) Profiler:
    Uses native Win32 `ctypes` (`GetProcessMemoryInfo` / `GetProcessTimes`) on Windows
    and `/proc/<pid>/statm` + `/proc/<pid>/stat` on Linux/POSIX with zero `psutil` dependency.
    """
    if not pid or pid <= 0 or not _is_pid_alive(pid):
        _CPU_TIME_SNAPSHOTS.pop(pid, None)
        return {"ram_mb": 0.0, "cpu_percent": 0.0}

    ram_mb = 0.0
    cpu_pct = 0.0
    now_mono = time.monotonic()

    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_VM_READ = 0x0010
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid))
            if h_proc:
                try:
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                    if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(pmc), pmc.cb):
                        ram_mb = round(float(pmc.WorkingSetSize) / (1024.0 * 1024.0), 2)

                    creation_t = ctypes.c_ulonglong()
                    exit_t = ctypes.c_ulonglong()
                    kernel_t = ctypes.c_ulonglong()
                    user_t = ctypes.c_ulonglong()
                    if kernel32.GetProcessTimes(
                        h_proc,
                        ctypes.byref(creation_t),
                        ctypes.byref(exit_t),
                        ctypes.byref(kernel_t),
                        ctypes.byref(user_t),
                    ):
                        # 100-nanosecond intervals -> seconds
                        proc_secs = float(kernel_t.value + user_t.value) / 1e7
                        prev = _CPU_TIME_SNAPSHOTS.get(pid)
                        _CPU_TIME_SNAPSHOTS[pid] = (now_mono, proc_secs)
                        if prev is not None and (now_mono - prev[0]) >= 0.25:
                            cores = max(1, os.cpu_count() or 1)
                            cpu_pct = round(
                                min(100.0, max(0.0, ((proc_secs - prev[1]) / (now_mono - prev[0])) * 100.0 / cores)),
                                1,
                            )
                finally:
                    kernel32.CloseHandle(h_proc)
        except Exception:
            pass
    else:
        try:
            statm_path = f"/proc/{int(pid)}/statm"
            if os.path.exists(statm_path):
                with open(statm_path, "r", encoding="utf-8") as f:
                    parts = f.read().split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    page_sz = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
                    ram_mb = round((rss_pages * page_sz) / (1024.0 * 1024.0), 2)

            stat_path = f"/proc/{int(pid)}/stat"
            if os.path.exists(stat_path):
                with open(stat_path, "r", encoding="utf-8") as f:
                    raw_stat = f.read()
                rpar = raw_stat.rfind(")")
                if rpar != -1:
                    fields = raw_stat[rpar + 2 :].split()
                    if len(fields) >= 13:
                        utime_ticks = int(fields[11])
                        stime_ticks = int(fields[12])
                        clk_tck = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
                        proc_secs = float(utime_ticks + stime_ticks) / max(1.0, clk_tck)
                        prev = _CPU_TIME_SNAPSHOTS.get(pid)
                        _CPU_TIME_SNAPSHOTS[pid] = (now_mono, proc_secs)
                        if prev is not None and (now_mono - prev[0]) >= 0.25:
                            cores = max(1, os.cpu_count() or 1)
                            cpu_pct = round(
                                min(100.0, max(0.0, ((proc_secs - prev[1]) / (now_mono - prev[0])) * 100.0 / cores)),
                                1,
                            )
        except Exception:
            pass

    return {"ram_mb": ram_mb, "cpu_percent": cpu_pct}


# ==============================================================================
# UPGRADE #5 & #7: FINOPS COST ESTIMATOR & LIVE EWMA MODEL HEALTH MATRIX
# ==============================================================================
def compute_finops_and_model_health_matrix(shared_cache_data: dict) -> dict:
    """
    Orchestrator Upgrades #5 & #7:
    Aggregates model EWMA latency, error rates, circuit-breaker status (`CLOSED`,
    `HALF-OPEN`, `OPEN`), token throughput, Vision dHash cache size, and estimated
    DeepInfra spend ($) across all child bot heartbeats.
    """
    hb_map = shared_cache_data.get("_bot_process_heartbeats", {})
    if not isinstance(hb_map, dict):
        hb_map = {}

    merged_models: dict[str, dict] = {}
    for _, bot_hb in hb_map.items():
        if not isinstance(bot_hb, dict):
            continue
        m_tel = bot_hb.get("model_telemetry", {})
        if not isinstance(m_tel, dict):
            continue
        for model_key, stats in m_tel.items():
            if not isinstance(stats, dict):
                continue
            entry = merged_models.setdefault(
                model_key,
                {
                    "model_key": model_key,
                    "ewma_latency_sec": 0.0,
                    "ewma_error_rate": 0.0,
                    "calls": 0,
                    "failures": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "circuit_status": "CLOSED",
                    "estimated_cost_usd": 0.0,
                },
            )
            entry["calls"] += int(stats.get("calls", 0))
            entry["failures"] += int(stats.get("failures", 0))
            entry["prompt_tokens"] += int(stats.get("prompt_tokens", 0))
            entry["completion_tokens"] += int(stats.get("completion_tokens", 0))
            lat = float(stats.get("ewma_latency_sec", 0.0))
            if lat > 0:
                entry["ewma_latency_sec"] = round(
                    lat if entry["ewma_latency_sec"] == 0 else (0.5 * entry["ewma_latency_sec"] + 0.5 * lat),
                    3,
                )
            entry["ewma_error_rate"] = round(max(entry["ewma_error_rate"], float(stats.get("ewma_error_rate", 0.0))), 3)
            c_stat = str(stats.get("circuit_status", "CLOSED"))
            if c_stat == "OPEN" or (c_stat == "HALF-OPEN" and entry["circuit_status"] == "CLOSED"):
                entry["circuit_status"] = c_stat

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0

    for m_key, entry in merged_models.items():
        p_tok = entry["prompt_tokens"]
        c_tok = entry["completion_tokens"]
        calls = entry["calls"]
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        cost = 0.0
        matched_pricing = None
        for price_key, p_info in MODEL_FINOPS_PRICING.items():
            if price_key.lower() in m_key.lower():
                matched_pricing = p_info
                break
        if matched_pricing:
            if "per_call" in matched_pricing:
                cost = calls * float(matched_pricing["per_call"])
            else:
                cost = (p_tok / 1_000_000.0) * float(matched_pricing.get("in_per_1m", 0.15)) + (
                    c_tok / 1_000_000.0
                ) * float(matched_pricing.get("out_per_1m", 0.45))
        else:
            cost = ((p_tok + c_tok) / 1_000_000.0) * 0.20

        entry["estimated_cost_usd"] = round(cost, 5)
        total_cost_usd += cost

    dhash_entries = len(shared_cache_data.get("_dhash_vision_cache", {}) or {})
    gen_img_entries = len(shared_cache_data.get("_generated_images", {}) or [])

    return {
        "models": list(merged_models.values()),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_estimated_cost_usd": round(total_cost_usd, 4),
        "vision_dhash_cached_items": dhash_entries,
        "generated_images_logged": gen_img_entries,
    }


# ==============================================================================
# SOCIAL GRAPH & MEMORY SALIENCE AGGREGATOR FOR WEB-UI CONTROL PLANE
# ==============================================================================
def build_control_plane_state_snapshot(
    supervisor: "MultiBotSupervisor",
    shared_cache_file: str,
    user_profiles_file: str,
    long_term_memory_file: str,
    general_memory_file: str,
    channel_config_file: str,
) -> dict:
    """
    Builds the unified real-time JSON state payload for `web_dashboard.html`:
    - Child bot process cards (status, PID, RAM MB, CPU %, WS latency, restarts)
    - Signed directed Social Graph nodes & edges G = (V, E, w) for Fruchterman-Reingold Canvas (#9)
    - Subconscious `[THOUGHT:]` X-Ray log, `_room_state`, and TTL `_bot_internal_state` (#10)
    - Memory Salience & Ebbinghaus Retention CRUD table across all JSON stores (#11)
    - Per-channel tuning sliders & toggles (#12)
    - FinOps & EWMA Model Circuit-Breaker Matrix (#5 & #7)
    """
    scache = load_json_file(shared_cache_file, {}) if shared_cache_file else {}
    uprof = dict(load_json_file(user_profiles_file, {}) if user_profiles_file else {})
    ltm = list(load_json_file(long_term_memory_file, []) if long_term_memory_file else [])
    gkm = list(load_json_file(general_memory_file, []) if general_memory_file else [])
    ch_configs = dict(load_json_file(channel_config_file, {}) if channel_config_file else {})

    # Also merge per-bot `<bot>_user_profiles.json`, `<bot>_long_term_memory.json`, and `<bot>_channel_configs.json`
    ltm_dir = os.path.dirname(long_term_memory_file) if long_term_memory_file else ""
    cfg_dir = os.path.dirname(channel_config_file) if channel_config_file else ""
    if ltm_dir and os.path.isdir(ltm_dir):
        for up_path in sorted(glob.glob(os.path.join(ltm_dir, "*_user_profiles.json"))):
            sub_u = load_json_file(up_path, {})
            if isinstance(sub_u, dict):
                for uk, uv in sub_u.items():
                    if isinstance(uv, list):
                        uprof.setdefault(uk, []).extend(x for x in uv if x not in uprof.get(uk, []))
        for lt_path in sorted(glob.glob(os.path.join(ltm_dir, "*_long_term_memory.json"))):
            sub_l = load_json_file(lt_path, [])
            if isinstance(sub_l, list):
                for item in sub_l:
                    if item not in ltm:
                        ltm.append(item)
    if cfg_dir and os.path.isdir(cfg_dir):
        for cf_path in sorted(glob.glob(os.path.join(cfg_dir, "*_channel_configs.json"))):
            sub_c = load_json_file(cf_path, {})
            if isinstance(sub_c, dict):
                for ck, cv in sub_c.items():
                    if isinstance(cv, dict):
                        ch_configs.setdefault(ck, {}).update(cv)

    # 1. Social Graph Nodes & Signed Directed Edges
    active_bots_reg = scache.get("_active_bots", {}) if isinstance(scache.get("_active_bots"), dict) else {}
    bot_names = [
        str(info.get("display_name", "")).strip()
        for info in active_bots_reg.values()
        if isinstance(info, dict) and info.get("display_name")
    ]
    nodes_map: dict[str, dict] = {}
    for bn in bot_names:
        b_key = _extract_base_username(bn)
        if b_key:
            nodes_map[b_key] = {"id": b_key, "label": bn, "type": "bot", "affinity_avg": 0}

    edges = []
    for target_user, facts_list in (uprof or {}).items():
        if not isinstance(target_user, str) or target_user.startswith("_") or not isinstance(facts_list, list):
            continue
        t_key = _extract_base_username(target_user)
        if not t_key:
            continue
        is_bot_target = any(_extract_base_username(bn) == t_key for bn in bot_names)
        sc, tr, cat, _ = _extract_saved_rapport_state_full(facts_list, target_name=target_user)
        tier = _get_disposition_tier_label(sc)
        arch = _derive_relationship_archetype(sc, tr, is_fellow_bot=is_bot_target, had_past_friction=(sc < 0))
        nodes_map.setdefault(
            t_key,
            {
                "id": t_key,
                "label": target_user,
                "type": "bot" if is_bot_target else "human",
                "affinity": sc,
                "tier": tier,
                "archetype": arch,
                "trend": tr,
                "catalyst": cat,
            },
        )
        nodes_map[t_key].update({"affinity": sc, "tier": tier, "archetype": arch, "trend": tr, "catalyst": cat})
        source_id = _extract_base_username(bot_names[0]) if bot_names else "server_bots"
        if source_id not in nodes_map:
            nodes_map[source_id] = {"id": source_id, "label": bot_names[0] if bot_names else "Server Bots", "type": "bot"}
        if source_id != t_key:
            edges.append({
                "source": source_id,
                "target": t_key,
                "weight": sc,
                "tier": tier,
                "archetype": arch,
                "trend": tr,
                "catalyst": cat,
            })

    # 2. Memory Salience & Ebbinghaus Decay CRUD Table (#11)
    now_ts = time.time()
    memory_rows = []
    for uname, flist in (uprof or {}).items():
        if not isinstance(uname, str) or uname.startswith("_") or not isinstance(flist, list):
            continue
        for idx, raw_f in enumerate(flist):
            s_clean = str(raw_f).strip()
            if not s_clean or RAPPORT_ENTRY_REGEX.match(s_clean):
                continue
            info = unpack_salience_fact(s_clean)
            ret = compute_ebbinghaus_retention(s_clean, now_ts=now_ts)
            memory_rows.append({
                "store": "user_profiles",
                "owner": uname,
                "index": idx,
                "text": info["text"],
                "salience": info["salience"],
                "category": info["category"],
                "date": info["date"],
                "recalls": info["recalls"],
                "retention_pct": round(min(100.0, ret), 1),
                "raw": s_clean,
            })

    for idx, raw_f in enumerate(ltm if isinstance(ltm, list) else []):
        s_clean = str(raw_f).strip()
        if not s_clean:
            continue
        info = unpack_salience_fact(s_clean)
        ret = compute_ebbinghaus_retention(s_clean, now_ts=now_ts)
        memory_rows.append({
            "store": "long_term_memory",
            "owner": "Server Lore",
            "index": idx,
            "text": info["text"],
            "salience": info["salience"],
            "category": info["category"],
            "date": info["date"],
            "recalls": info["recalls"],
            "retention_pct": round(min(100.0, ret), 1),
            "raw": s_clean,
        })

    for idx, raw_f in enumerate(gkm if isinstance(gkm, list) else []):
        s_clean = str(raw_f).strip()
        if not s_clean:
            continue
        info = unpack_salience_fact(s_clean)
        ret = compute_ebbinghaus_retention(s_clean, now_ts=now_ts)
        memory_rows.append({
            "store": "general_knowledge",
            "owner": "General Knowledge",
            "index": idx,
            "text": info["text"],
            "salience": info["salience"],
            "category": info["category"],
            "date": info["date"],
            "recalls": info["recalls"],
            "retention_pct": round(min(100.0, ret), 1),
            "raw": s_clean,
        })

    return {
        "timestamp": now_ts,
        "bots": supervisor.get_bots_status_list(scache),
        "finops": compute_finops_and_model_health_matrix(scache),
        "social_graph": {"nodes": list(nodes_map.values()), "edges": edges},
        "room_states": scache.get("_room_state", {}),
        "bot_internal_states": scache.get("_bot_internal_state", {}),
        "subconscious_xray_log": (scache.get("_subconscious_xray_log", []) or [])[-30:],
        "episodic_chapters": scache.get("_episodic_chapters", {}),
        "memory_table": memory_rows[-180:],
        "channel_configs": {str(k): v for k, v in (ch_configs or {}).items()},
        "tunnel_info": {
            "status": supervisor.tunnel_manager.status if supervisor.tunnel_manager else "OFFLINE",
            "provider": supervisor.tunnel_manager.provider if supervisor.tunnel_manager else "stopped",
            "public_url": supervisor.tunnel_manager.public_url if supervisor.tunnel_manager else os.getenv("WEBUI_PUBLIC_URL", ""),
            "magic_url": (
                f"{supervisor.tunnel_manager.public_url}/?token={os.getenv('WEBUI_ADMIN_TOKEN', '')}"
                if (supervisor.tunnel_manager and supervisor.tunnel_manager.public_url)
                else ""
            ),
        },
        "env_summary": {
            "env_file": supervisor.layout["env_file"],
            "max_history_length": int(read_env_file_dict(supervisor.layout["env_file"]).get("MAX_HISTORY_LENGTH", "200") or 200),
            "keys": [
                {"key": k, "masked": mask_env_secret(k, v)}
                for k, v in sorted(read_env_file_dict(supervisor.layout["env_file"]).items())
            ],
        },
        "recent_logs": list(supervisor.log_ring_buffer[-80:]),
    }


# ==============================================================================
# MANAGED BOT CHILD PROCESS SUPERVISOR & HOT-RELOAD ENGINE
# ==============================================================================
class BuiltInPublicTunnelManager:
    """
    Zero-Install Remote Admin Tunnel Manager (with Automatic `cloudflared` Provisioning):
    1. Checks system PATH or `bots/bot_info/python_modules/` for `cloudflared`.
    2. If not installed yet, automatically downloads the official standalone `cloudflared`
       binary (~35 MB from Cloudflare's official GitHub release) into `python_modules/`
       so `https://xxxx.trycloudflare.com` works out-of-the-box with zero SSL errors!
    3. Falls back to `ssh -R 80:127.0.0.1:<port> nokey@localhost.run` (matching only the
       top-level `https://<id>.lhr.life` domain, NEVER the nested `.app.lhr.life` subdomain
       which causes `ERR_SSL_PROTOCOL_ERROR` due to wildcard TLS limits).
    """
    def __init__(self, supervisor: "MultiBotSupervisor", local_port: int, shared_cache_file: str, admin_token: str):
        self.supervisor = supervisor
        self.local_port = int(local_port)
        self.shared_cache_file = shared_cache_file
        self.admin_token = admin_token
        self.proc: asyncio.subprocess.Process | None = None
        self.public_url: str = os.getenv("WEBUI_PUBLIC_URL", "").strip().rstrip("/")
        self.provider: str = "env" if self.public_url else "stopped"
        self.status: str = "ONLINE" if self.public_url else "OFFLINE"

    def _sync_public_url(self, new_url: str, status: str, provider: str):
        self.public_url = new_url.strip().rstrip("/")
        self.status = status
        self.provider = provider
        os.environ["WEBUI_PUBLIC_URL"] = self.public_url

        def _upd(c_data):
            info = c_data.setdefault("_webui_control_plane_info", {})
            info["public_url"] = self.public_url
            info["tunnel_status"] = self.status
            info["tunnel_provider"] = self.provider
            info["updated_at"] = time.time()
            return c_data

        try:
            update_json_file(self.shared_cache_file, {}, _upd)
        except Exception:
            pass

    def _resolve_or_download_cloudflared(self) -> str | None:
        """
        Finds `cloudflared` on PATH or inside `python_modules/`, or auto-downloads
        the official standalone binary for Windows (`amd64`) or Linux (`amd64`/`arm64`).
        """
        sys_cf = shutil.which("cloudflared")
        if sys_cf:
            return sys_cf

        bin_name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
        local_bin = os.path.join(self.supervisor.data_dir, bin_name)
        if os.path.exists(local_bin) and os.path.getsize(local_bin) > 5_000_000:
            if os.name != "nt":
                try:
                    os.chmod(local_bin, 0o755)
                except Exception:
                    pass
            return local_bin

        import platform
        import urllib.request
        arch = platform.machine().lower()
        if os.name == "nt":
            dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        elif "aarch64" in arch or "arm64" in arch:
            dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        else:
            dl_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

        try:
            self.supervisor.append_log("Tunnel", f"Auto-downloading standalone cloudflared binary ({bin_name})...", "INFO")
            tmp_bin = local_bin + ".tmp"
            urllib.request.urlretrieve(dl_url, tmp_bin)
            if os.path.exists(tmp_bin) and os.path.getsize(tmp_bin) > 5_000_000:
                if os.name != "nt":
                    os.chmod(tmp_bin, 0o755)
                os.replace(tmp_bin, local_bin)
                self.supervisor.append_log("Tunnel", "Downloaded cloudflared binary successfully!", "INFO")
                return local_bin
        except Exception as e:
            self.supervisor.append_log("Tunnel", f"Cloudflared auto-download skipped ({e}); falling back to SSH.", "WARN")
        return None

    async def start_tunnel(self) -> tuple[bool, str]:
        if self.proc and self.proc.returncode is None and self.public_url:
            return True, self.public_url

        await self.stop_tunnel()
        self._sync_public_url("", "STARTING", "auto")

        cf_bin = await asyncio.to_thread(self._resolve_or_download_cloudflared)
        has_ssh = shutil.which("ssh") is not None

        if cf_bin:
            cmd = [cf_bin, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.local_port}"]
            prov = "cloudflared"
        elif has_ssh:
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ExitOnForwardFailure=yes",
                "-R", f"80:127.0.0.1:{self.local_port}",
                "nokey@localhost.run",
            ]
            prov = "ssh (localhost.run)"
        else:
            self._sync_public_url("", "OFFLINE", "none")
            return False, "Neither 'cloudflared' nor 'ssh' is available."

        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.supervisor.append_log("Tunnel", f"Starting public HTTPS tunnel via {prov}...", "INFO")
            asyncio.create_task(self._watch_tunnel_output(self.proc, prov))
            for _ in range(40):
                await asyncio.sleep(0.25)
                if self.public_url:
                    return True, self.public_url
                if self.proc.returncode is not None:
                    break
            return bool(self.public_url), (self.public_url or "Tunnel started — waiting for URL assignment in logs...")
        except Exception as e:
            self._sync_public_url("", "OFFLINE", "error")
            return False, str(e)

    async def _watch_tunnel_output(self, proc: asyncio.subprocess.Process, prov: str):
        # Strictly match ONLY single-level subdomains (`https://<id>.trycloudflare.com` or `https://<id>.lhr.life`)
        # so nested `.app.lhr.life` strings never trigger `ERR_SSL_PROTOCOL_ERROR`!
        cf_url_regex = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com\b")
        lhr_url_regex = re.compile(r"https://([a-zA-Z0-9\-]+)(?:\.[a-zA-Z0-9\-]+)*\.lhr\.life\b")
        if not proc.stdout:
            return
        try:
            while True:
                line_b = await proc.stdout.readline()
                if not line_b:
                    break
                line_s = line_b.decode("utf-8", errors="ignore").strip()
                if not line_s:
                    continue
                found_url = ""
                m_cf = cf_url_regex.search(line_s)
                if m_cf:
                    found_url = m_cf.group(0).rstrip("/")
                else:
                    m_lhr = lhr_url_regex.search(line_s)
                    if m_lhr:
                        sub_id = m_lhr.group(1)
                        if sub_id not in ("admin", "docs", "app"):
                            # Always normalize `*.app.lhr.life` -> `https://<id>.lhr.life` (valid wildcard TLS cert!)
                            found_url = f"https://{sub_id}.lhr.life"

                if found_url and found_url != self.public_url:
                    self._sync_public_url(found_url, "ONLINE", prov)
                    magic = f"{found_url}/?token={self.admin_token}" if self.admin_token else found_url
                    self.supervisor.append_log("Tunnel", f"PUBLIC ADMIN MAGIC LINK ACTIVE: {magic}", "INFO")
                    print(f"\n🌐 [PUBLIC ADMIN TUNNEL ONLINE] Magic Link: {magic}\n")
        except Exception:
            pass
        finally:
            if self.proc is proc:
                self._sync_public_url("", "OFFLINE", "stopped")
                self.supervisor.append_log("Tunnel", "Public HTTPS tunnel closed.", "WARN")

    async def stop_tunnel(self):
        proc = self.proc
        self.proc = None
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._sync_public_url("", "OFFLINE", "stopped")


class ManagedBotProcess:
    def __init__(self, script_path: str):
        self.script_path = os.path.abspath(script_path)
        self.name = os.path.splitext(os.path.basename(script_path))[0]
        self.process: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.status = "STOPPED"  # RUNNING | BACKOFF | QUARANTINED | STOPPED
        self.limiter = LeakyBucketRestartLimiter()
        self.started_at: float = 0.0
        self.restarts: int = 0
        self.ram_mb: float = 0.0
        self.cpu_percent: float = 0.0
        self.last_stdout_ts: float = 0.0
        self.manual_stop: bool = False


class MultiBotSupervisor:
    def __init__(
        self,
        script_paths: list[str],
        data_dir: str = ".",
        enable_hot_reload: bool = True,
    ):
        self.layout = resolve_project_layout(data_dir)
        self.data_dir = self.layout["python_modules_dir"]
        self.logs_dir = self.layout["logs_dir"]
        self.enable_hot_reload = enable_hot_reload
        self.bots: dict[str, ManagedBotProcess] = {
            os.path.splitext(os.path.basename(p))[0]: ManagedBotProcess(p)
            for p in script_paths
        }
        self.log_ring_buffer: list[dict] = []
        self.sse_queues: set[asyncio.Queue] = set()
        self.watched_mtimes: dict[str, int] = {}
        self.tunnel_manager: BuiltInPublicTunnelManager | None = None
        self.running = True
        self._snapshot_watched_files()

    def _get_watch_folders(self) -> list[str]:
        return list(dict.fromkeys([
            self.layout["python_modules_dir"],
            self.layout["bots_dir"],
        ]))

    def _snapshot_watched_files(self):
        for folder in self._get_watch_folders():
            for py_file in glob.glob(os.path.join(folder, "*.py")):
                try:
                    self.watched_mtimes[os.path.abspath(py_file)] = os.stat(py_file).st_mtime_ns
                except Exception:
                    pass

    def append_log(self, bot_name: str, line: str, level: str = "INFO"):
        now_dt = datetime.now()
        entry = {
            "ts": time.time(),
            "time_str": now_dt.strftime("%H:%M:%S"),
            "bot": bot_name,
            "level": level,
            "msg": line.strip()[:350],
        }
        self.log_ring_buffer.append(entry)
        if len(self.log_ring_buffer) > 250:
            self.log_ring_buffer = self.log_ring_buffer[-250:]
        if self.logs_dir and os.path.isdir(self.logs_dir):
            try:
                safe_b = re.sub(r"[^a-zA-Z0-9_\-]", "_", bot_name or "orchestrator")
                log_f = os.path.join(self.logs_dir, f"{safe_b}.log")
                with open(log_f, "a", encoding="utf-8") as lf:
                    lf.write(f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {line.strip()}\n")
            except Exception:
                pass
        self.broadcast_sse_event("log", entry)

    def broadcast_sse_event(self, event_type: str, data_obj: dict):
        payload = f"event: {event_type}\ndata: {json.dumps(data_obj, ensure_ascii=False)}\n\n"
        dead_q = []
        for q in self.sse_queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead_q.append(q)
        for dq in dead_q:
            self.sse_queues.discard(dq)

    def get_bots_status_list(self, scache: dict | None = None) -> list[dict]:
        hb_map = (scache or {}).get("_bot_process_heartbeats", {}) if isinstance(scache, dict) else {}
        out = []
        now_t = time.time()
        for b_name, mb in self.bots.items():
            ws_lat = 0.0
            for _, hb in (hb_map or {}).items():
                if isinstance(hb, dict) and hb.get("pid") == mb.pid:
                    ws_lat = float(hb.get("ws_latency_ms", 0.0))
                    break
            uptime_s = int(now_t - mb.started_at) if (mb.status == "RUNNING" and mb.started_at > 0) else 0
            out.append({
                "name": b_name,
                "script": os.path.basename(mb.script_path),
                "pid": mb.pid,
                "status": mb.status,
                "uptime_sec": max(0, uptime_s),
                "restarts": mb.restarts,
                "ram_mb": mb.ram_mb,
                "cpu_percent": mb.cpu_percent,
                "ws_latency_ms": ws_lat,
                "quarantined": mb.limiter.quarantined,
                "tokens_remaining": round(mb.limiter.tokens, 2),
            })
        return out

    async def start_bot(self, bot_name: str):
        mb = self.bots.get(bot_name)
        if not mb or mb.status == "RUNNING":
            return
        if mb.limiter.quarantined:
            self.append_log(bot_name, "Cannot start while QUARANTINED. Click Unquarantine first.", "WARN")
            return

        ok, err_msg = validate_python_file_ast(mb.script_path)
        if not ok:
            mb.status = "QUARANTINED"
            mb.limiter.quarantined = True
            self.append_log(bot_name, f"Pre-launch AST check failed: {err_msg}", "ERROR")
            return

        env_ok, missing_keys = verify_bot_script_env_keys(mb.script_path, self.layout["env_file"])
        if not env_ok:
            mb.status = "STOPPED"
            self.append_log(
                bot_name,
                f"[ENV ERROR] Missing required key(s) in bot_info/.env: {', '.join(missing_keys)}. Add them in the '🔑 API Keys & .env' tab!",
                "ERROR",
            )
            return

        mb.manual_stop = False
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        existing_pypath = env.get("PYTHONPATH", "")
        mod_dir = self.layout["python_modules_dir"]
        env["PYTHONPATH"] = f"{mod_dir}{os.pathsep}{existing_pypath}" if existing_pypath else mod_dir
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                mb.script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.path.dirname(mb.script_path) or self.data_dir,
                env=env,
            )
            mb.process = proc
            mb.pid = proc.pid
            mb.status = "RUNNING"
            mb.started_at = time.time()
            mb.last_stdout_ts = time.time()
            self.append_log(bot_name, f"Spawned child process PID={proc.pid} ({os.path.basename(mb.script_path)})", "INFO")
            asyncio.create_task(self._stream_bot_stdout(mb, proc))
            asyncio.create_task(self._monitor_bot_exit(mb, proc))
        except Exception as e:
            mb.status = "STOPPED"
            self.append_log(bot_name, f"Failed to spawn process: {e}", "ERROR")

    async def _stream_bot_stdout(self, mb: ManagedBotProcess, proc: asyncio.subprocess.Process):
        if not proc.stdout:
            return
        try:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line_str = line_bytes.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    mb.last_stdout_ts = time.time()
                    lvl = "ERROR" if "[ERROR]" in line_str or "Traceback" in line_str else ("WARN" if "[WARN]" in line_str else "INFO")
                    self.append_log(mb.name, line_str, lvl)
        except Exception:
            pass

    async def _monitor_bot_exit(self, mb: ManagedBotProcess, proc: asyncio.subprocess.Process):
        rc = await proc.wait()
        if mb.process is not proc:
            return
        mb.pid = None
        mb.ram_mb = 0.0
        mb.cpu_percent = 0.0
        if mb.manual_stop or not self.running:
            mb.status = "STOPPED"
            self.append_log(mb.name, f"Process stopped cleanly (exit={rc}).", "INFO")
            return

        allowed, backoff_s = mb.limiter.record_crash_and_get_backoff()
        if not allowed:
            mb.status = "QUARANTINED"
            self.append_log(
                mb.name,
                f"Process crashed (exit={rc}). Crash-loop threshold tripped -> QUARANTINED!",
                "ERROR",
            )
            return

        mb.status = "BACKOFF"
        mb.restarts += 1
        self.append_log(
            mb.name,
            f"Process exited (code={rc}). Restart #{mb.restarts} in {backoff_s:.1f}s (Token Bucket={mb.limiter.tokens:.1f})...",
            "WARN",
        )
        await asyncio.sleep(backoff_s)
        if not mb.manual_stop and self.running and not mb.limiter.quarantined:
            await self.start_bot(mb.name)

    async def stop_bot(self, bot_name: str, manual: bool = True):
        mb = self.bots.get(bot_name)
        if not mb:
            return
        mb.manual_stop = manual
        proc = mb.process
        mb.process = None
        if proc and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=4.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Also sweep OS process table for any detached ghost process running this script
        active_other_pids = {
            other.pid for k, other in self.bots.items()
            if k != bot_name and other.pid is not None
        }
        ghost_pids = await asyncio.to_thread(
            kill_orphaned_bot_processes, [os.path.basename(mb.script_path)], active_other_pids
        )
        mb.pid = None
        mb.ram_mb = 0.0
        mb.cpu_percent = 0.0
        mb.status = "STOPPED"
        if ghost_pids:
            self.append_log(bot_name, f"Stopped bot process + killed detached ghost PID(s): {ghost_pids}", "WARN")
        else:
            self.append_log(bot_name, "Stopped bot process.", "INFO")

    async def restart_bot(self, bot_name: str, unquarantine: bool = False):
        mb = self.bots.get(bot_name)
        if not mb:
            return
        if unquarantine:
            mb.limiter.reset_quarantine()
        await self.stop_bot(bot_name, manual=True)
        await asyncio.sleep(0.35)
        await self.start_bot(bot_name)

    async def rolling_hot_reload_all(self, changed_file: str):
        """
        Orchestrator Upgrade #2 — AST-Validated Zero-Downtime Rolling Hot-Reload:
        Validates syntax via `ast.parse()` + `py_compile` first, then restarts active bots
        one at a time with a 2.0s stagger so the server never goes completely offline.
        """
        ok, err_msg = validate_python_file_ast(changed_file)
        if not ok:
            self.append_log("HotReload", f"ABORTED hot-reload! {err_msg}", "ERROR")
            return

        self.append_log(
            "HotReload",
            f"AST validated '{os.path.basename(changed_file)}'. Starting zero-downtime rolling reload...",
            "INFO",
        )
        for b_name, mb in list(self.bots.items()):
            if mb.status == "RUNNING":
                await self.restart_bot(b_name, unquarantine=False)
                await asyncio.sleep(2.0)

    async def telemetry_and_watchdog_loop(self, shared_cache_file: str):
        """
        Polls OS RAM/CPU (#6), checks Hot-Reload file mtimes (#2), reaps orphaned
        locks/temps (#4), and checks IPC Heartbeat freeze watchdog (#3).
        """
        last_reap_ts = time.time()
        while self.running:
            await asyncio.sleep(3.0)
            try:
                # 1. Poll OS RAM & CPU for each running child PID
                active_pids = set()
                for mb in self.bots.values():
                    if mb.status == "RUNNING" and mb.pid:
                        active_pids.add(mb.pid)
                        prof = query_process_ram_and_cpu(mb.pid)
                        mb.ram_mb = prof["ram_mb"]
                        mb.cpu_percent = prof["cpu_percent"]

                # 2. Check AST Hot-Reload file watcher across python_modules/ and bots/
                if self.enable_hot_reload:
                    all_watched = [
                        p
                        for folder in self._get_watch_folders()
                        for p in glob.glob(os.path.join(folder, "*.py"))
                    ]
                    for py_path in all_watched:
                        abs_p = os.path.abspath(py_path)
                        if os.path.basename(abs_p) == "orchestrator.py":
                            continue
                        try:
                            cur_ns = os.stat(abs_p).st_mtime_ns
                        except Exception:
                            continue
                        prev_ns = self.watched_mtimes.get(abs_p)
                        if prev_ns is not None and cur_ns > prev_ns:
                            self.watched_mtimes[abs_p] = cur_ns
                            await self.rolling_hot_reload_all(abs_p)
                            break
                        else:
                            self.watched_mtimes[abs_p] = cur_ns

                # 3. Periodic Orphaned Lock & Temp Reaper (every 60s)
                if (time.time() - last_reap_ts) >= 60.0:
                    last_reap_ts = time.time()
                    reaped_cnt = reap_orphaned_locks_and_temps([self.data_dir], active_pids)
                    if reaped_cnt > 0:
                        self.append_log("Reaper", f"Evicted {reaped_cnt} orphaned .lock/.tmp file(s).", "INFO")
            except Exception as e:
                print(f"[Orchestrator] Telemetry loop warn: {e}")


# ==============================================================================
# UPGRADE #8: ZERO-DEPENDENCY ASYNCIO HTTP + SSE CONTROL PLANE SERVER
# ==============================================================================
async def start_webui_http_server(
    supervisor: MultiBotSupervisor,
    host: str,
    port: int,
    shared_cache_file: str,
    user_profiles_file: str,
    long_term_memory_file: str,
    general_memory_file: str,
    channel_config_file: str,
    admin_token: str = "",
):
    dashboard_html_path = os.path.join(supervisor.data_dir, "web_dashboard.html")

    async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            req_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not req_line:
                writer.close()
                return
            parts = req_line.decode("utf-8", errors="ignore").strip().split(" ")
            if len(parts) < 2:
                writer.close()
                return
            method, raw_path = parts[0].upper(), parts[1]
            parsed_url = urlparse(raw_path)
            path = parsed_url.path

            headers = {}
            while True:
                h_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not h_line or h_line in (b"\r\n", b"\n"):
                    break
                h_str = h_line.decode("utf-8", errors="ignore").strip()
                if ":" in h_str:
                    k, v = h_str.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_len = int(headers.get("content-length", "0") or 0)
            body_bytes = b""
            if content_len > 0 and content_len <= 1024 * 512:
                body_bytes = await asyncio.wait_for(reader.readexactly(content_len), timeout=5.0)

            def _send_json(status_code: int, payload_obj: dict):
                status_text = "Unauthorized" if status_code == 401 else ("Bad Request" if status_code == 400 else "OK")
                raw_b = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
                hdr = (
                    f"HTTP/1.1 {status_code} {status_text}\r\n"
                    "Content-Type: application/json; charset=utf-8\r\n"
                    f"Content-Length: {len(raw_b)}\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("utf-8")
                writer.write(hdr + raw_b)

            peer_info = writer.get_extra_info("peername")
            peer_ip = str(peer_info[0]) if isinstance(peer_info, (tuple, list)) and peer_info else "127.0.0.1"
            query_params = parse_qs(parsed_url.query)

            # Enforce Admin Token Authentication on all /api/* routes for remote/LAN clients
            if path.startswith("/api/"):
                if not is_webui_request_authorized(peer_ip, headers, query_params, admin_token):
                    _send_json(401, {"ok": False, "error": "Unauthorized: Valid WEBUI_ADMIN_TOKEN required."})
                    await writer.drain()
                    writer.close()
                    return

            # 1. Serve `web_dashboard.html` at GET /
            if method == "GET" and path in ("/", "/index.html", "/dashboard"):
                if os.path.exists(dashboard_html_path):
                    with open(dashboard_html_path, "rb") as f:
                        html_bytes = f.read()
                else:
                    html_bytes = b"<html><body style='background:#0d1117;color:#e6edf3;font-family:sans-serif;padding:2rem;'><h2>Discord Multi-Agent Control Plane API Online</h2><p>Place <code>web_dashboard.html</code> in the root directory to load the full UI.</p></body></html>"
                hdr = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html; charset=utf-8\r\n"
                    f"Content-Length: {len(html_bytes)}\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("utf-8")
                writer.write(hdr + html_bytes)
                await writer.drain()
                writer.close()
                return

            # 2. Live State Snapshot at GET /api/state
            if method == "GET" and path == "/api/state":
                snap = build_control_plane_state_snapshot(
                    supervisor,
                    shared_cache_file,
                    user_profiles_file,
                    long_term_memory_file,
                    general_memory_file,
                    channel_config_file,
                )
                _send_json(200, snap)
                await writer.drain()
                writer.close()
                return

            # 3. Server-Sent Events (SSE) Live Telemetry Stream at GET /api/stream
            if method == "GET" and path == "/api/stream":
                sse_hdr = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream; charset=utf-8\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Connection: keep-alive\r\n"
                    "Access-Control-Allow-Origin: *\r\n\r\n"
                ).encode("utf-8")
                writer.write(sse_hdr)
                await writer.drain()

                q: asyncio.Queue = asyncio.Queue(maxsize=64)
                supervisor.sse_queues.add(q)
                try:
                    while supervisor.running:
                        try:
                            ev_str = await asyncio.wait_for(q.get(), timeout=3.0)
                            writer.write(ev_str.encode("utf-8"))
                            await writer.drain()
                        except asyncio.TimeoutError:
                            snap = build_control_plane_state_snapshot(
                                supervisor,
                                shared_cache_file,
                                user_profiles_file,
                                long_term_memory_file,
                                general_memory_file,
                                channel_config_file,
                            )
                            ping_frame = f"event: state\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"
                            writer.write(ping_frame.encode("utf-8"))
                            await writer.drain()
                except Exception:
                    pass
                finally:
                    supervisor.sse_queues.discard(q)
                    writer.close()
                return

            # 4. POST Endpoints for Bot Control, Memory CRUD, Channel Sliders & Director Mode
            if method == "POST":
                req_json = json.loads(body_bytes.decode("utf-8", errors="ignore") or "{}")

                if path == "/api/bot/action":
                    b_name = str(req_json.get("bot", "")).strip()
                    action = str(req_json.get("action", "")).strip().lower()
                    if b_name == "__ALL__":
                        for k_name in list(supervisor.bots.keys()):
                            if action == "start":
                                await supervisor.start_bot(k_name)
                                await asyncio.sleep(0.8)
                            elif action == "stop":
                                await supervisor.stop_bot(k_name, manual=True)
                            elif action == "restart":
                                await supervisor.restart_bot(k_name, unquarantine=True)
                                await asyncio.sleep(1.2)
                    else:
                        if action == "start":
                            await supervisor.start_bot(b_name)
                        elif action == "stop":
                            await supervisor.stop_bot(b_name, manual=True)
                        elif action in ("restart", "unquarantine"):
                            await supervisor.restart_bot(b_name, unquarantine=True)
                    _send_json(200, {"ok": True, "bot": b_name, "action": action})

                elif path == "/api/bot/inspect":
                    b_name = str(req_json.get("bot", "")).strip()
                    mb = supervisor.bots.get(b_name)
                    if not mb:
                        _send_json(404, {"ok": False, "error": f"Bot '{b_name}' not found"})
                    else:
                        _send_json(200, {"ok": True, "config": inspect_bot_script_config(mb.script_path)})

                elif path == "/api/bot/save_source":
                    # Web-UI Bot Studio Editor (replaces terminal bot_editor.py with AST pre-validation!)
                    b_name = str(req_json.get("bot", "")).strip()
                    new_src = str(req_json.get("source_code", ""))
                    auto_restart = bool(req_json.get("restart", True))
                    mb = supervisor.bots.get(b_name)
                    safe_bname = re.sub(r"[^a-zA-Z0-9_-]", "", b_name)
                    target_path = mb.script_path if mb else os.path.join(supervisor.layout["bots_dir"], f"{safe_bname}.py")
                    try:
                        ast.parse(new_src, filename=target_path)
                    except SyntaxError as syn_err:
                        _send_json(400, {"ok": False, "error": f"SyntaxError on line {syn_err.lineno}: {syn_err.msg}"})
                        await writer.drain()
                        writer.close()
                        return
                    with open(target_path, "w", encoding="utf-8", newline="\n") as sf:
                        sf.write(new_src)
                    supervisor.watched_mtimes[os.path.abspath(target_path)] = os.stat(target_path).st_mtime_ns
                    if not mb:
                        new_key = os.path.splitext(os.path.basename(target_path))[0]
                        supervisor.bots[new_key] = ManagedBotProcess(target_path)
                        b_name = new_key
                    supervisor.append_log("BotStudio", f"Saved & AST-verified '{os.path.basename(target_path)}'.", "INFO")
                    if auto_restart:
                        await supervisor.restart_bot(b_name, unquarantine=True)
                    _send_json(200, {"ok": True, "bot": b_name})

                elif path == "/api/bot/update_config":
                    # [E] Visual Bot Config Editor (updates constants + optional restart)
                    b_name = str(req_json.get("bot", "")).strip()
                    updates = req_json.get("updates", {})
                    auto_restart = bool(req_json.get("restart", True))
                    mb = supervisor.bots.get(b_name)
                    if not mb or not isinstance(updates, dict):
                        _send_json(404, {"ok": False, "error": f"Bot '{b_name}' not found"})
                    else:
                        ok_u, msg_u = update_bot_script_constants(mb.script_path, updates)
                        if not ok_u:
                            _send_json(400, {"ok": False, "error": msg_u})
                        else:
                            supervisor.watched_mtimes[os.path.abspath(mb.script_path)] = os.stat(mb.script_path).st_mtime_ns
                            supervisor.append_log("BotStudio", f"Updated config constants in '{os.path.basename(mb.script_path)}'.", "INFO")
                            if auto_restart and mb.status == "RUNNING":
                                await supervisor.restart_bot(b_name, unquarantine=True)
                            _send_json(200, {"ok": True, "bot": b_name})

                elif path == "/api/bot/create":
                    # [C] Create New Bot Wizard (generates bots/<name>.py + saves token to .env + optionally starts!)
                    b_name = str(req_json.get("bot_name", "")).strip()
                    d_token = str(req_json.get("discord_token", "")).strip()
                    prompt_id = str(req_json.get("system_prompt", "")).strip()
                    kw_raw = req_json.get("keywords", [])
                    kw_list = [str(x).strip() for x in kw_raw if str(x).strip()] if isinstance(kw_raw, list) else []
                    ok_c, safe_n, path_or_err = generate_new_bot_script(
                        bots_dir=supervisor.layout["bots_dir"],
                        env_path=supervisor.layout["env_file"],
                        bot_name=b_name,
                        discord_token=d_token,
                        system_prompt_identity=prompt_id,
                        text_model=str(req_json.get("text_model", "Qwen/Qwen3-235B-A22B-Instruct-2507")),
                        vision_model=str(req_json.get("vision_model", "Qwen/Qwen3-VL-235B-A22B-Instruct")),
                        enable_vision=bool(req_json.get("enable_vision", True)),
                        enable_gifs=bool(req_json.get("enable_gifs", True)),
                        enable_image_gen=bool(req_json.get("enable_image_gen", True)),
                        enable_voice=bool(req_json.get("enable_voice", False)),
                        temperature=float(req_json.get("temperature", 0.75)),
                        top_p=float(req_json.get("top_p", 0.88)),
                        chance_to_respond=float(req_json.get("chance_to_respond", 0.01)),
                        random_emoji_chance=float(req_json.get("random_emoji_chance", 0.35)),
                        default_random_interval=int(req_json.get("default_random_interval", 0)),
                        keywords=kw_list,
                    )
                    if not ok_c:
                        _send_json(400, {"ok": False, "error": path_or_err})
                    else:
                        supervisor.bots[safe_n] = ManagedBotProcess(path_or_err)
                        supervisor.watched_mtimes[os.path.abspath(path_or_err)] = os.stat(path_or_err).st_mtime_ns
                        supervisor.append_log("BotWizard", f"Created new bot '{safe_n}.py' and saved token to .env!", "INFO")
                        if req_json.get("start_now", False):
                            await supervisor.start_bot(safe_n)
                        _send_json(200, {"ok": True, "bot": safe_n, "path": path_or_err})

                elif path == "/api/env/set":
                    # [A] & [H] API Keys & Global MAX_HISTORY_LENGTH Manager in bots/bot_info/.env
                    k_name = str(req_json.get("key", "")).strip()
                    k_val = str(req_json.get("value", "")).strip()
                    ok_e = upsert_env_variable(supervisor.layout["env_file"], k_name, k_val)
                    if ok_e:
                        os.environ[re.sub(r"[^A-Za-z0-9_]", "", k_name)] = k_val
                        supervisor.append_log("EnvManager", f"Updated '{k_name}' in bot_info/.env", "INFO")
                        _send_json(200, {"ok": True, "key": k_name})
                    else:
                        _send_json(400, {"ok": False, "error": "Invalid key name"})

                elif path == "/api/memory/mutate":
                    # Upgrade #11 — Memory Salience CRUD Inspector (Pin S:98, Edit, Add, Delete with Zero-PII Firewall)
                    store = str(req_json.get("store", "user_profiles"))
                    owner = str(req_json.get("owner", ""))
                    op = str(req_json.get("op", "")).lower()
                    idx = int(req_json.get("index", -1))
                    new_text = str(req_json.get("text", "")).strip()
                    new_sal = int(req_json.get("salience", 75) or 75)
                    new_cat = str(req_json.get("category", "GEN") or "GEN").upper()

                    if op in ("add", "edit", "pin"):
                        safe_t = sanitize_or_reject_sensitive_pii(new_text)
                        if not safe_t:
                            _send_json(400, {"ok": False, "error": "Blocked by Zero-PII Security Firewall."})
                            await writer.drain()
                            writer.close()
                            return
                        packed = pack_salience_fact(
                            safe_t,
                            salience=(98 if op == "pin" else new_sal),
                            category=("CORE" if op == "pin" else new_cat),
                        )
                    else:
                        packed = ""

                    if store == "user_profiles" and owner:
                        def _mut_u(uprof_d):
                            lst = uprof_d.setdefault(owner, [])
                            if op == "delete" and 0 <= idx < len(lst):
                                lst.pop(idx)
                            elif op in ("edit", "pin") and 0 <= idx < len(lst):
                                lst[idx] = packed
                            elif op == "add" and packed:
                                consolidate_and_append_fact(packed, lst, max_items=35)
                            return uprof_d
                        update_json_file(user_profiles_file, {}, _mut_u)
                    else:
                        target_f = long_term_memory_file if store == "long_term_memory" else general_memory_file
                        def _mut_l(lst):
                            if not isinstance(lst, list):
                                lst = []
                            if op == "delete" and 0 <= idx < len(lst):
                                lst.pop(idx)
                            elif op in ("edit", "pin") and 0 <= idx < len(lst):
                                lst[idx] = packed
                            elif op == "add" and packed:
                                consolidate_and_append_fact(packed, lst, max_items=100)
                            return lst
                        update_json_file(target_f, [], _mut_l)
                    _send_json(200, {"ok": True})

                elif path == "/api/channel/config":
                    # Upgrade #12 — Live Per-Channel Sliders & Media Toggles
                    ch_id_raw = str(req_json.get("channel_id", "")).strip()
                    updates = req_json.get("updates", {})
                    if ch_id_raw and isinstance(updates, dict):
                        ch_key = int(ch_id_raw) if ch_id_raw.isdigit() else ch_id_raw
                        def _mut_ch(c_map):
                            c_map.setdefault(ch_key, {}).update(updates)
                            return c_map
                        update_json_file(channel_config_file, {}, _mut_ch)
                    _send_json(200, {"ok": True})

                elif path == "/api/tunnel/toggle":
                    action = str(req_json.get("action", "toggle")).strip().lower()
                    tm = supervisor.tunnel_manager
                    if not tm:
                        _send_json(400, {"ok": False, "error": "Tunnel manager not initialized"})
                    else:
                        if action == "stop" or (action == "toggle" and tm.status == "ONLINE"):
                            await tm.stop_tunnel()
                            _send_json(200, {"ok": True, "status": "OFFLINE", "public_url": ""})
                        else:
                            ok_t, url_or_err = await tm.start_tunnel()
                            if ok_t:
                                magic_u = f"{url_or_err}/?token={tm.admin_token}" if tm.admin_token else url_or_err
                                _send_json(200, {"ok": True, "status": tm.status, "public_url": url_or_err, "magic_url": magic_u})
                            else:
                                _send_json(400, {"ok": False, "error": url_or_err})

                elif path == "/api/director/inject":
                    # Upgrade #13 — Director Mode Stage Prompt & World Event Injector
                    ch_id_str = str(req_json.get("channel_id", "0")).strip() or "0"
                    topic = str(req_json.get("active_topic", "")).strip()
                    vibe = str(req_json.get("room_vibe", "")).strip()
                    prop_item = str(req_json.get("prop", "")).strip()
                    hook_txt = str(req_json.get("open_hook", "")).strip()

                    def _mut_dir(sc):
                        r_map = sc.setdefault("_room_state", {})
                        cur_r = r_map.setdefault(ch_id_str, {})
                        if topic:
                            cur_r["active_topic"] = topic[:80]
                        if vibe:
                            cur_r["room_vibe"] = vibe[:50]
                        if prop_item:
                            props = cur_r.get("active_props", [])
                            if not isinstance(props, list):
                                props = []
                            props.append({"item": prop_item[:65], "ttl": 5})
                            cur_r["active_props"] = props[-3:]
                        if hook_txt:
                            cur_r["open_hook"] = {"speaker": "Director Mode", "hook": hook_txt[:120], "ts": time.time()}
                        cur_r["updated_at"] = time.time()
                        return sc

                    update_json_file(shared_cache_file, {}, _mut_dir)
                    supervisor.append_log("DirectorMode", f"Injected stage event into channel {ch_id_str}: topic='{topic}', prop='{prop_item}'", "INFO")
                    _send_json(200, {"ok": True})
                else:
                    _send_json(404, {"error": "Unknown endpoint"})

                await writer.drain()
                writer.close()
                return

            _send_json(404, {"error": "Not found"})
            await writer.drain()
            writer.close()
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(_handle_client, host, port)
    return server


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
async def async_main():
    parser = argparse.ArgumentParser(description="Discord Multi-Agent AI Supervisor & Web-UI Control Plane")
    parser.add_argument("--bots", nargs="*", help="Explicit bot launcher .py scripts to supervise")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Localhost bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("ORCHESTRATOR_PORT", "0")), help="Preferred Web-UI port (default: auto 8765..8799)")
    parser.add_argument("--autostart", action="store_true", help="Automatically start all bots on launch (default: False — start via Web-UI)")
    parser.add_argument("--tunnel", action="store_true", help="Automatically start a public HTTPS tunnel (via built-in ssh localhost.run or cloudflared)")
    parser.add_argument("--no-webui", action="store_true", help="Disable the local Web-UI HTTP/SSE server")
    parser.add_argument("--no-hot-reload", action="store_true", help="Disable AST rolling hot-reload watcher")
    args = parser.parse_args()

    layout = resolve_project_layout()
    reaped = reap_orphaned_locks_and_temps([
        layout["workspace_root"],
        layout["bots_dir"],
        layout["bot_info_dir"],
        layout["python_modules_dir"],
        os.path.dirname(layout["shared_cache_file"]),
        os.path.dirname(layout["user_profiles_file"]),
        os.path.dirname(layout["channel_config_file"]),
    ])
    if reaped:
        print(f"[ORCHESTRATOR] Startup reaper cleaned {reaped} stale .lock/.tmp file(s).")

    scripts = [os.path.abspath(p) for p in args.bots] if args.bots else discover_bot_launcher_scripts(layout["bots_dir"])
    supervisor = MultiBotSupervisor(scripts, data_dir=layout["python_modules_dir"], enable_hot_reload=(not args.no_hot_reload))

    shared_cache_file = layout["shared_cache_file"]
    user_profiles_file = layout["user_profiles_file"]
    long_term_memory_file = layout["long_term_memory_file"]
    general_memory_file = layout["general_memory_file"]
    channel_config_file = layout["channel_config_file"]

    web_server = None
    if not args.no_webui:
        admin_token = ensure_webui_admin_token(layout["env_file"])
        bound_port = args.port if args.port > 0 else find_free_localhost_port(args.host, DEFAULT_PORT_START, DEFAULT_PORT_END)
        lan_ip = get_lan_ip_address()
        public_url = os.getenv("WEBUI_PUBLIC_URL", "").strip().rstrip("/")
        local_url = f"http://127.0.0.1:{bound_port}"
        lan_url = f"http://{lan_ip}:{bound_port}"

        def _save_webui_info(c_data):
            c_data["_webui_control_plane_info"] = {
                "local_url": local_url,
                "lan_url": lan_url,
                "public_url": public_url,
                "port": bound_port,
                "host": args.host,
                "updated_at": time.time(),
            }
            return c_data

        update_json_file(shared_cache_file, {}, _save_webui_info)

        web_server = await start_webui_http_server(
            supervisor=supervisor,
            host=args.host,
            port=bound_port,
            shared_cache_file=shared_cache_file,
            user_profiles_file=user_profiles_file,
            long_term_memory_file=long_term_memory_file,
            general_memory_file=general_memory_file,
            channel_config_file=channel_config_file,
            admin_token=admin_token,
        )
        print("=" * 78)
        print(f"🚀 DISCORD MULTI-AGENT CONTROL PLANE ONLINE: {local_url}")
        print(f"   • LAN / Admin Magic Link: {lan_url}/?token={admin_token}")
        if public_url:
            print(f"   • Public Tunnel Magic Link: {public_url}/?token={admin_token}")
        print(f"   • Discord Admin Command:  Type '@BotName admin webui' in Discord for a 1-click link!")
        print(f"   • Supervised Bots:        {', '.join(supervisor.bots.keys()) or 'None discovered yet'}")
        print("=" * 78)

    if not args.no_webui:
        supervisor.tunnel_manager = BuiltInPublicTunnelManager(
            supervisor=supervisor,
            local_port=bound_port,
            shared_cache_file=shared_cache_file,
            admin_token=admin_token,
        )
        if args.tunnel:
            await supervisor.tunnel_manager.start_tunnel()

    # Sweep and terminate any detached ghost bot processes left over from prior sessions
    all_script_names = [os.path.basename(mb.script_path) for mb in supervisor.bots.values()]
    killed_on_boot = await asyncio.to_thread(kill_orphaned_bot_processes, all_script_names)
    if killed_on_boot:
        print(f"[ORCHESTRATOR] Terminated {len(killed_on_boot)} detached ghost bot process(es) from prior session: {killed_on_boot}")
        supervisor.append_log("Supervisor", f"Terminated {len(killed_on_boot)} detached ghost bot PIDs on startup: {killed_on_boot}", "WARN")
        await asyncio.sleep(1.0)

    if args.autostart:
        for b_name in list(supervisor.bots.keys()):
            await supervisor.start_bot(b_name)
            await asyncio.sleep(1.2)
    else:
        print("[ORCHESTRATOR] Standby Mode: All bots are STOPPED ready for manual launch in the Web-UI!")
        supervisor.append_log(
            "Supervisor",
            f"Ready in STANDBY mode ({len(supervisor.bots)} bots discovered). Click 'Start' on any bot or 'Start All' in the Web-UI!",
            "INFO",
        )

    try:
        await supervisor.telemetry_and_watchdog_loop(shared_cache_file)
    except asyncio.CancelledError:
        pass
    finally:
        supervisor.running = False
        if supervisor.tunnel_manager:
            await supervisor.tunnel_manager.stop_tunnel()
        for b_name in list(supervisor.bots.keys()):
            await supervisor.stop_bot(b_name, manual=True)
        if web_server:
            web_server.close()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Shutting down all supervised bots cleanly...")
