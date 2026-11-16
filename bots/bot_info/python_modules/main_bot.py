import os
import json
import time
import random
import asyncio
import re
import copy
import base64
import discord
from datetime import datetime

# ==============================================================================
# MODULAR IMPORTS
# ==============================================================================
from config import (
    ERROR_REPLY_STRIP_REGEX, LEARN_USER_STRIP_REGEX, LEARN_GENERAL_STRIP_REGEX,
    LEARN_STRIP_REGEX, EMOJI_REACT_STRIP_REGEX, INTERNAL_REACT_STRIP_REGEX,
    IMAGE_DESC_STRIP_REGEX, GIF_STRIP_REGEX, GIF_EXTRACT_REGEX,
    UNCLOSED_GIF_EXTRACT_REGEX, MEME_GEN_REGEX, UNCLOSED_MEME_GEN_REGEX,
    IMAGE_GEN_REGEX, UNCLOSED_IMAGE_GEN_REGEX, URL_REGEX,
    EXPLICIT_MEME_REQ_REGEX, EXPLICIT_IMAGE_REQ_REGEX, BOT_IMAGE_PROMISE_REGEX,
    EXPLICIT_VOICE_REQ_REGEX, EXPLICIT_GIF_REQ_REGEX,
    LEARN_USER_EXTRACT_REGEX, LEARN_GENERAL_EXTRACT_REGEX, LEARN_EXTRACT_REGEX,
    FALLBACK_IMAGE_NOTE_REGEX, FALLBACK_VISION_MODELS, FALLBACK_TEXT_MODELS,
    TOPIC_CHANGE_REQUEST_REGEX, BARE_FILLER_UTTERANCE_REGEX,
    INTENTIONAL_NO_RESPONSE_REGEX, PERSONAL_FACT_STATEMENT_REGEX,
    IMAGE_REMIX_REQ_REGEX, IMAGE_DESC_EXTRACT_REGEX
)

from core_system import (
    load_json_file, update_json_file, post_with_backoff, sanitize_url,
    is_model_circuit_open, rank_models_by_ewma_latency, get_all_model_telemetry
)
from text_and_parsers import (
    extract_and_strip_voice_msg, extract_and_strip_thought_tag,
    clean_and_compact_reply, derive_chat_text_from_voice,
    parse_meme_gen_tag, evaluate_deterministic_grounding_tools
)
from media_engine import (
    fetch_image_bytes_from_message, fetch_online_gif, get_youtube_metadata,
    extract_gif_slug, is_visual_media_candidate, extract_webpage_embed_context,
    is_audio_voice_attachment, fetch_audio_bytes_from_message
)
from memory_and_context import (
    extract_glitch_summary, build_api_messages_from_history, upsert_chat_memory_entry,
    verify_and_resolve_user_facts, is_duplicate_fact, find_canonical_username,
    route_server_vs_general_facts, build_temporal_context_block,
    build_active_user_context_block, select_relevant_memories,
    is_ephemeral_or_banter_fact, consolidate_and_append_fact,
    extract_implicit_user_facts_from_message, get_user_affinity_score,
    _extract_base_username, _extract_saved_rapport_state_full,
    _get_disposition_tier_label, _derive_relationship_archetype,
    RAPPORT_ENTRY_REGEX
)
from subconscious_engine import (
    update_message_reaction_ledger,
    run_subconscious_cognitive_turn,
    get_and_tick_bot_internal_state,
)
from ai_services import (
    summarize_image_for_memory, auto_react, ensure_unique_image_prompt,
    ensure_unique_meme_captions, synthesize_missing_image_prompt,
    synthesize_missing_meme_spec, generate_image_attachment,
    generate_voice_attachment, transcribe_voice_attachment
)
from bot_discovery import (
    register_active_bot, get_other_active_bots, is_command_directed_at_other_bot,
    detect_addressed_bot_name, resolve_contextual_followup,
    compute_spontaneous_interest_multiplier, is_other_bot_in_active_exchange,
    is_directly_addressed_by_fellow_bot, _term_boundary_pattern,
    claim_or_coordinate_turn_wave, mark_turn_wave_complete,
    pick_fellow_bot_interaction_target
)
from commands_and_embeds import handle_bot_commands, create_generated_media_view, register_hub_slash_command

# ==============================================================================
# IN-FILE HELPER FUNCTIONS (INCLUDING UPGRADES #8 & #12)
# ==============================================================================
_TERMINAL_PUNCT_CHARS = (".", "?", "!", "~", '"', "”", "'", "’", ")")
_TECHNICAL_QUERY_CUE_REGEX = re.compile(
    r"```|\b(?:def\s+\w+|class\s+\w+|import\s+\w+|traceback|syntaxerror|exception|"
    r"compile|segfault|pointer|algorithm|complexity|big[\s\-]?o|sql|regex|dockerfile|"
    r"asyncio|pthread| shader|calculus|derivative|integral|matrix|eigen|calculate|solve)\b",
    re.IGNORECASE,
)
_BANTER_ROAST_CUE_REGEX = re.compile(
    r"\b(?:roast\b|meme\b|shitpost|skill\s+issue|ratio\b|1v1\b|fight\s+me|bet\s+you\s+can['’]?t|"
    r"banter|joke\b|funny|hilarious|lmao|lmfao)\b",
    re.IGNORECASE,
)


def compute_adaptive_coalesce_delay(clean_text: str, is_author_bot: bool = False) -> float:
    """
    Upgrade #8 — Velocity & Punctuation-Aware Adaptive Coalescing Window:
    - Fellow bot messages: `0.30s`
    - Complete punctuated human sentences (>= 5 words ending in `.`, `?`, `!`, `~`):
      `0.12s` fast-path (saving ~730ms of perceived typing latency on every turn!).
    - Short unpunctuated fragments (<= 3 words without terminal punctuation):
      `1.25s` window so rapid split-line typers can finish their second line.
    - Medium unpunctuated messages: `0.45s`.
    """
    if is_author_bot:
        return 0.30
    stripped = (clean_text or "").strip()
    if not stripped:
        return 0.20
    words = stripped.split()
    has_term_punct = stripped.endswith(_TERMINAL_PUNCT_CHARS)
    if len(words) >= 5 and has_term_punct:
        return 0.12
    if len(words) <= 3 and not has_term_punct:
        return 1.25
    return 0.45


def compute_situational_sampling_params(
    user_text: str,
    base_temp: float = 0.78,
    base_top_p: float = 0.90,
    is_meme_or_roast: bool = False,
) -> tuple[float, float]:
    """
    Upgrade #12 — Situational Sampling Modulation:
    Dynamically tunes `(temperature, top_p)` per turn:
    - Technical / Code / Math mode: `(min(base_temp, 0.45), 0.85)` for high precision.
    - Banter / Roast / Meme mode: `(max(base_temp, 0.85), 0.92)` for punchy creativity.
    - Standard conversational mode: `(base_temp, base_top_p)`.
    """
    txt = (user_text or "").strip()
    if _TECHNICAL_QUERY_CUE_REGEX.search(txt):
        return round(min(float(base_temp), 0.45), 2), 0.85
    if is_meme_or_roast or _BANTER_ROAST_CUE_REGEX.search(txt):
        return round(max(float(base_temp), 0.85), 2), 0.92
    return float(base_temp), float(base_top_p)
def extract_persona_voice_defaults(system_prompt_content: str, bot_name: str) -> tuple[str, str]:
    role_match = re.search(r'Role:\s*([^\n`\]]+)', system_prompt_content, re.IGNORECASE)
    scene_match = re.search(r'Scene:\s*([^\n`\]]+)', system_prompt_content, re.IGNORECASE)
    guidance_match = re.search(r'Guidance:\s*([^\n`\]]+)', system_prompt_content, re.IGNORECASE)
    style_match = re.search(r'\n\(([A-Za-z\s]{3,20})\)\s+[A-Za-z]', system_prompt_content)

    role_str = role_match.group(1).strip() if role_match else ""
    scene_str = scene_match.group(1).strip() if scene_match else ""
    guidance_str = guidance_match.group(1).strip() if guidance_match else ""

    if not role_str or role_str.startswith("["):
        first_para = system_prompt_content.strip().split("\n")[0]
        first_para = re.sub(r'^You are\s+', '', first_para, flags=re.IGNORECASE).strip()
        role_str = f"A distinctive character speaker matching this persona: {first_para[:160]}"
    if not scene_str or scene_str.startswith("["):
        scene_str = f"Chatting naturally with friends in a Discord server as {bot_name}."
    if not guidance_str or guidance_str.startswith("["):
        guidance_str = "Speak with natural cadence, expressive tone, and clear articulation."

    default_style = style_match.group(1).strip() if style_match else "Natural"
    default_director_block = f"Role: {role_str}\nScene: {scene_str}\nGuidance: {guidance_str}"
    return default_director_block, default_style


def build_glitch_awareness_rules(bot_name: str) -> str:
    return (
        "\n\n### IN-CHARACTER GLITCH & ERROR AWARENESS RULES:\n"
        f"- If you see a `[System Glitch Context: ...]` or `[System Glitch Event: ...]` note in the conversation history, you are self-aware of it! You can naturally joke, grumble, make excuses, or tease other bots in-character about brain-freezes, 504 lag spikes, dropped packets, or scuffed renders.\n"
        f"- **STRICT ANTI-LOOP RULE**: NEVER output literal raw error strings like `*(API Error 504 - Provider busy. Try again in a moment.)*` or `*(System: ...)*`. Always express any reaction to a glitch in your own natural character voice."
    )


def build_memory_attribution_rules(bot_name: str, other_bot_names: list[str] = None) -> str:
    other_bots_str = ", ".join(other_bot_names) if other_bot_names else "other bots in the server"
    return (
        "\n\n### CRITICAL IDENTITY, SITUATIONAL RADAR, RELATIONSHIP DYNAMICS & MEMORY RULES:\n"
        f"1. **STRICT PERSONA & SPEECH-STYLE ISOLATION (YOU ARE {bot_name.upper()}, NEVER ANOTHER BOT)**: You are **{bot_name}**, and ONLY **{bot_name}**. "
        f"You share this channel with other bots ({other_bots_str}), which are completely separate characters with their own species, roles, and speech rules. "
        f"NEVER confuse yourself with another bot, NEVER mimic another bot's speech restrictions or vocalizations, and when history says `[Username (speaking to OtherBot, NOT to {bot_name})]: ...`, remember they were addressing `OtherBot`, NOT you!\n"
        "2. **SITUATIONAL & SOCIAL GRAPH AWARENESS (USERS & FELLOW BOTS)**:\n"
        "   - Check `### Live Situational Awareness, Room Social Graph & Dossier` on every turn. Pay attention to the **Immediate Situation** (never mistake external venting about a game crash/bug for an insult toward you!), your **Relationship Archetype** with the speaker (whether Human Regular or Fellow Server Bot), **Room Relationship Contrast** (treat your Trusted Inner Circle favorites and Bonded Companions noticeably warmer than newcomers or rivals in the same room), **PROTECTIVE LOYALTY TRIGGERS** (defend friends/companions you are fond of when someone is rude to them!), and **SHARED BOND / RIVALRY TRIGGERS**.\n"
        "3. **PROACTIVE LEARNING, NICKNAME BOUNDARIES & DISPOSITION CATALYSTS (`[LEARN_USER: name | fact]`)**:\n"
        "   - Output `[LEARN_USER: name | specific 8-20 word objective fact]` whenever a human user OR a fellow bot reveals a meaningful detail, identity trait (gender/pronouns, name, age, family), personal boundary (e.g. 'don\'t call me sweetie', preferred nicknames), preference/favorite, game/project they are playing or building, or notable interaction.\n"
        "   - CRITICAL: Writing 'note to self: ...' in normal chat text does NOT save to memory! You MUST include the literal `[LEARN_USER: name | fact]` or `[LEARN: server event]` bracket tag at the very end of your message whenever you want to remember something.\n"
        "   - When an interaction with a user or fellow bot genuinely warms you up, earns your respect, or annoys you, log the catalyst via `[LEARN_USER: name | Disposition: concise reason why your attitude shifted]`.\n"
        "4. **NOTABLE SERVER SITUATIONS & GENERAL KNOWLEDGE (1 TO 2 TAGS MAX PER TURN)**:\n"
        "   - Use `[LEARN: specific 10-22 word summary]` to record notable server situations, funny group events, ongoing gaming/project sessions, or evolving relationships/alliances/rivalries between server members and bots.\n"
        "   - Use `[LEARN_GENERAL: fact]` for interesting universal trivia, gaming mechanics, science, or real-world facts.\n"
        "   - **BALANCE RULE**: Log rich, specific details when something new or notable happens, never shoehorn unrelated saved facts out of context, and attach at most 1–2 learning tags per turn."
    )


def build_media_and_meme_rules(enable_image_gen: bool, enable_gifs: bool) -> str:
    rules = ["\n\n### EXPRESSIVE VISUALS, SPONTANEITY & MEME BALANCE RULES:"]
    rules.append(
        "- **HUMAN-LIKE CONVERSATIONAL AGENCY**: Respond like a real person hanging out in Discord—vary your reply length naturally (from a quick 1-sentence quip or specific observation to a fuller remark), react to specific details in what people say or draw, and never repeat the same stock catchphrases or 3-emote templates across turns."
    )
    if enable_image_gen:
        rules.append(
            "- **PROACTIVE IMAGES & MEMES (`[IMAGE_GEN:]` vs `[MEME_GEN:]`)**: Actively look for organic opportunities to generate custom visuals when a moment is funny, dramatic, or descriptive! "
            "Use standard `[IMAGE_GEN: detailed visual scene]` for candid scenes, portraits, or user-requested pictures that don't need text overlays, and use `[MEME_GEN: vivid visual scene (no text in scene) | <witty setup line> | <witty punchline>]` when a classic top/bottom caption punchline makes the joke funnier (NEVER write literal placeholder words like 'TOP CAPTION' or 'BOTTOM PUNCHLINE', and do NOT attach an image/meme on consecutive turns unless asked!).\n"
            "  - **STRICT SUBJECT FIDELITY**: When a user asks you to generate an image of a specific subject, person, food, or scene, your visual prompt MUST include all requested subjects and details—never substitute yourself or random wizards!\n"
            "  - **100% FRESH MEME CAPTIONS**: Never reuse stock captions like `ABSOLUTELY UNACCEPTABLE` or repeat captions from earlier turns. Keep a balanced mix between uncaptioned `[IMAGE_GEN:]`, captioned `[MEME_GEN:]`, and pure text replies."
        )
    if enable_gifs:
        rules.append(
            "- **USE GIFS SPARINGLY (`[GIF: ...]`)**: Do NOT attach `[GIF:]` to every message. Only use `[GIF: specific 2-4 word emotion/reaction]` occasionally (~15% of turns) when you are NOT generating an image or meme. When in doubt, prefer `[IMAGE_GEN:]`, `[MEME_GEN:]`, or text alone over a Giphy link."
        )
    return "\n".join(rules) if len(rules) > 1 else ""


async def resolve_reply_context(
    message: discord.Message,
    bot_user_id: int,
    shared_image_cache_file: str
) -> tuple[bool, str, discord.Message | None]:
    if not message.reference or not message.reference.message_id:
        return False, "", None

    ref_msg = message.reference.resolved
    if not isinstance(ref_msg, discord.Message):
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None

    if not ref_msg:
        return False, "", None

    replied_to_self = (ref_msg.author.id == bot_user_id)
    ref_author = ref_msg.author.display_name
    raw_ref_content = ref_msg.content or ""
    glitch_summary = extract_glitch_summary(raw_ref_content)
    ref_text = ERROR_REPLY_STRIP_REGEX.sub('', raw_ref_content).strip()
    if len(ref_text) > 160:
        ref_text = ref_text[:157] + "..."

    img_cache = load_json_file(shared_image_cache_file, {})
    ref_media_desc = img_cache.get(ref_msg.id, "")
    if isinstance(ref_media_desc, str) and ref_media_desc.startswith("PROCESSING"):
        ref_media_desc = ""

    parts = []
    if ref_text:
        parts.append(f"\"{ref_text}\"")
    if glitch_summary:
        parts.append(f"[had a glitch: {glitch_summary}]")
    if ref_media_desc:
        parts.append(ref_media_desc)
    elif ref_msg.attachments:
        parts.append("[attached media]")

    summary_str = " ".join(parts) if parts else "[previous message]"
    return replied_to_self, f"[Replying to {ref_author}: {summary_str}]", ref_msg


# ==============================================================================
# MAIN BOT RUNNER
# ==============================================================================
def run_bot(
    token: str, api_key: str, system_prompt: dict, giphy_key: str = None,
    chat_memory_file: str = "chat_memory.json", shared_image_cache_file: str = "shared_image_cache.json",
    long_term_memory_file: str = "long_term_memory.json", user_profiles_file: str = "user_profiles.json",
    channel_config_file: str = "channel_configs.json", general_memory_file: str = None,
    text_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507", vision_model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
    image_gen_model: str = "black-forest-labs/FLUX-1-schnell", voice_model: str = "XiaomiMiMo/MiMo-V2.5-tts-voicedesign", tts_voice: str = "",
    enable_vision: bool = True, enable_gifs: bool = True, enable_image_gen: bool = False, enable_voice: bool = False,
    temperature: float = 0.78, top_p: float = 0.90, output_length: int = 550, context_window: int = 50, emoji_context_length: int = 12,
    chance_to_respond: float = 0.05, random_emoji_chance: float = 0.45, default_random_interval: int = 0,
    api_url: str = "https://api.deepinfra.com/v1/openai/chat/completions", keywords: list = None,
    emoji_reaction_hint: str = "", default_voice_role: str = "", default_voice_style: str = "",
    followup_window_messages: int = 2, max_followup_turns: int = 2, followup_timeout: float = 180.0
):
    if max_followup_turns == 1:
        max_followup_turns = 2
    api_url = sanitize_url(api_url)
    if not general_memory_file:
        ltm_dir = os.path.dirname(os.path.abspath(long_term_memory_file))
        general_memory_file = os.path.join(ltm_dir, "general_knowledge.json")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    if hasattr(intents, "reactions"):
        intents.reactions = True
    client = discord.Client(intents=intents)
    hub_cmd_tree = register_hub_slash_command(
        client,
        system_prompt=system_prompt,
        chat_memory_file=chat_memory_file,
        shared_image_cache_file=shared_image_cache_file,
        user_profiles_file=user_profiles_file,
        channel_config_file=channel_config_file,
    )
    slash_tree_synced = False
    max_history_length = int(os.getenv("MAX_HISTORY_LENGTH", 200))

    channel_gen_locks: dict[int, asyncio.Lock] = {}
    channel_in_coalesce_window: dict[int, bool] = {}
    channel_pending_waiters: dict[int, int] = {}
    channel_last_replied_msg_id: dict[int, int] = {}
    channel_followup_state: dict[int, dict] = {}
    channel_last_gif_time: dict[int, float] = {}
    random_loop_started = False
    last_status_sync_ts = 0.0
    last_synced_status_text = ""

    extracted_voice_role = default_voice_role
    extracted_voice_style = default_voice_style

    def record_ipc_heartbeat():
        """Writes live process telemetry, PID, WebSocket latency, and EWMA model stats for orchestrator.py."""
        if not client.user:
            return
        ws_ms = round(float(getattr(client, "latency", 0.0) or 0.0) * 1000.0, 1)
        b_id_str = str(client.user.id)
        b_name = client.user.display_name
        now_h = time.time()
        model_tel = get_all_model_telemetry()

        def _mut_hb(cache: dict) -> dict:
            hb_map = cache.setdefault("_bot_process_heartbeats", {})
            hb_map[b_id_str] = {
                "bot_id": client.user.id,
                "bot_name": b_name,
                "pid": os.getpid(),
                "ws_latency_ms": ws_ms,
                "timestamp": now_h,
                "model_telemetry": model_tel,
            }
            return cache

        try:
            update_json_file(shared_image_cache_file, {}, _mut_hb)
        except Exception:
            pass

    async def sync_bot_custom_status(active_channel_id: int = 0):
        """
        Crown Jewel #1 — Live Discord Custom Status & Activity Sync (`client.change_presence`):
        Updates the bot's Discord sidebar status from its Subconscious `mood` or `active_topic`
        (rate-limited to once per 180s, 0 extra API calls).
        """
        nonlocal last_status_sync_ts, last_synced_status_text
        now_s = time.time()
        if (now_s - last_status_sync_ts) < 180.0 or not client.user:
            return
        try:
            scache = load_json_file(shared_image_cache_file, {})
            ch_k = str(active_channel_id or 0)
            b_state = scache.get("_bot_internal_state", {}).get(ch_k, {}).get(client.user.display_name, {})
            r_state = scache.get("_room_state", {}).get(ch_k, {})
            mood_s = str((b_state or {}).get("mood", "") or "").strip()
            topic_s = str((r_state or {}).get("active_topic", "") or "").strip()
            status_parts = [p for p in (mood_s, topic_s) if p]
            if not status_parts:
                return
            new_status = " • ".join(status_parts)[:120]
            if new_status and new_status != last_synced_status_text and hasattr(discord, "CustomActivity"):
                last_status_sync_ts = now_s
                last_synced_status_text = new_status
                await client.change_presence(activity=discord.CustomActivity(name=new_status))
        except Exception:
            pass

    async def trigger_welcome_greeting(member: discord.Member, target_channel, is_test: bool = False):
        """Generates and sends an in-character server-join welcome message for `member`."""
        if not target_channel or not client.user:
            return
        if not is_test:
            stagger_sec = 1.2 + (((client.user.id >> 22) % 5) * 1.6)
            await asyncio.sleep(stagger_sec)
        try:
            dyn_sys = copy.deepcopy(system_prompt)
            welcome_prompt = (
                f"A new member named **{member.display_name}** just joined the Discord server! "
                f"Welcome **{member.display_name}** to the server in 1-2 short, punchy sentences strictly in your character voice as {client.user.display_name}. "
                "Do NOT output any [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [VOICE_MSG:] tags."
            )
            w_payload = {
                "model": text_model,
                "messages": [
                    dyn_sys,
                    {"role": "user", "content": welcome_prompt},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 180,
            }
            w_res = await post_with_backoff(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json_payload=w_payload,
                timeout=(6, 18),
                max_retries=2,
                bot_name=client.user.display_name,
            )
            if w_res and w_res.status_code == 200:
                w_reply = w_res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                _, w_reply = extract_and_strip_thought_tag(w_reply)
                _, w_reply = extract_and_strip_voice_msg(w_reply)
                w_reply = clean_and_compact_reply(w_reply)
                if w_reply:
                    sent_w = await target_channel.send(w_reply)
                    upsert_chat_memory_entry(
                        chat_memory_file,
                        target_channel.id,
                        {
                            "role": "assistant",
                            "content": w_reply,
                            "author_id": client.user.id,
                            "author_name": client.user.display_name,
                            "message_id": sent_w.id if sent_w else f"welcome_{time.time()}",
                            "author_is_bot": True,
                        },
                        max_history_length,
                    )
        except Exception as w_err:
            print(f"[{client.user.display_name}] [WARN] Welcome greeter error: {w_err}")

    async def random_post_loop():
        await client.wait_until_ready()
        last_post_times = {}
        while not client.is_closed():
            await asyncio.sleep(60)
            try:
                register_active_bot(shared_image_cache_file, client.user.id, client.user.display_name, keywords)
                record_ipc_heartbeat()

                conf_data = load_json_file(channel_config_file, {})
                for ch_id_int, conf in conf_data.items():
                    if not isinstance(ch_id_int, int):
                        continue
                    interval = conf.get("random_interval", default_random_interval)

                    if not conf.get("enabled", True) or interval <= 0 or conf.get("react_only", False):
                        continue

                    if ch_id_int not in last_post_times:
                        last_post_times[ch_id_int] = time.time()

                    if (time.time() - last_post_times[ch_id_int]) >= (interval * 60):
                        last_post_times[ch_id_int] = time.time()

                        channel = client.get_channel(ch_id_int)
                        if not channel:
                            continue

                        chat_memory = load_json_file(chat_memory_file, {})
                        hist = chat_memory.get(ch_id_int, [])

                        other_bots_loop_meta = get_other_active_bots(shared_image_cache_file, client.user.id)
                        other_bots_loop_names = [ob.get("display_name") for ob in other_bots_loop_meta if ob.get("display_name")]
                        dynamic_system = copy.deepcopy(system_prompt)
                        now_str = datetime.now().strftime("%A, %I:%M %p")
                        dynamic_system["content"] += build_temporal_context_block(hist[-context_window:], now_str)
                        dynamic_system["content"] += build_glitch_awareness_rules(client.user.display_name)
                        dynamic_system["content"] += build_memory_attribution_rules(client.user.display_name, other_bots_loop_names)

                        recent_Loop_text = " ".join(
                            m.get("content", "") for m in hist[-4:]
                            if isinstance(m, dict) and m.get("role") != "system" and isinstance(m.get("content"), str)
                        )
                        long_term_memory = select_relevant_memories(
                            load_json_file(long_term_memory_file, []), recent_Loop_text, max_items=4
                        )
                        if long_term_memory:
                            dynamic_system["content"] += "\n### Relevant Server Knowledge (Passive Reference Only):\n" + "\n".join(f"- {f}" for f in long_term_memory)

                        general_memory = select_relevant_memories(
                            load_json_file(general_memory_file, []), recent_Loop_text, max_items=4
                        )
                        if general_memory:
                            dynamic_system["content"] += "\n### Relevant General Knowledge (Passive Reference Only):\n" + "\n".join(f"- {f}" for f in general_memory)

                        uprof_loop = load_json_file(user_profiles_file, {})
                        active_loop_block = build_active_user_context_block(
                            user_profiles=uprof_loop,
                            raw_history=hist[-context_window:],
                            current_speaker=None,
                            combined_recent_prompt=recent_Loop_text,
                            is_spontaneous=True,
                            persona_prompt=system_prompt.get("content", ""),
                            other_bot_names=other_bots_loop_names,
                            my_bot_name=client.user.display_name
                        )
                        if active_loop_block:
                            dynamic_system["content"] += active_loop_block

                        # Check _room_state for active live banter (skip interrupting) or Crown Jewel #5 open_hook
                        scache_loop = load_json_file(shared_image_cache_file, {})
                        room_st_loop = scache_loop.get("_room_state", {}).get(str(ch_id_int), {})
                        if not isinstance(room_st_loop, dict):
                            room_st_loop = {}
                        if hist and isinstance(hist[-1].get("timestamp"), (int, float)):
                            if (time.time() - float(hist[-1]["timestamp"])) < 75.0:
                                # Channel is actively chatting right now; don't interrupt with an idle timer post
                                continue

                        open_hook_info = room_st_loop.get("open_hook")
                        skit_target = None
                        if isinstance(open_hook_info, dict) and open_hook_info.get("hook"):
                            # Clear the open_hook atomically so it is only checked in on once!
                            def _clear_open_hook(c_data, ck=str(ch_id_int)):
                                r_m = c_data.get("_room_state", {}).get(ck, {})
                                if isinstance(r_m, dict):
                                    r_m.pop("open_hook", None)
                                return c_data
                            update_json_file(shared_image_cache_file, {}, _clear_open_hook)
                            hook_speaker = open_hook_info.get("speaker", "a server member")
                            hook_text = open_hook_info.get("hook", "")
                            loop_directive_text = (
                                f"System Directive (Proactive Check-In): Earlier in chat, **{hook_speaker}** mentioned: '{hook_text}'. "
                                f"Naturally check in on **{hook_speaker}** or follow up on that topic in 1-2 short, in-character sentences! "
                                "DO NOT output [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [VOICE_MSG:]."
                            )
                        elif random.random() < 0.40:
                            # Phase 2 — Autonomous Multi-Bot Mini-Interaction (40% of idle posts)
                            skit_target = pick_fellow_bot_interaction_target(
                                shared_image_cache_file,
                                client.user.id,
                                user_profiles=uprof_loop,
                                persona_prompt=system_prompt.get("content", ""),
                                guild_members=getattr(channel.guild, "members", None) if getattr(channel, "guild", None) else None,
                            )
                            if skit_target and skit_target.get("display_name"):
                                t_bot_name = skit_target["display_name"]
                                t_aff = skit_target.get("affinity", 0)
                                rel_tone = "friendly ally/companion" if t_aff >= 15 else ("comedic rival/foil" if t_aff <= -8 else "fellow bot co-star")
                                loop_directive_text = (
                                    f"System Directive (Autonomous Inter-Bot Banter): Spontaneously address or tease your {rel_tone} **{t_bot_name}** by name "
                                    f"in 1-2 short, punchy in-character sentences to spark a quick 2-turn interaction! "
                                    "DO NOT output [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [VOICE_MSG:]."
                                )
                            else:
                                loop_directive_text = (
                                    "System Directive: Spontaneously initiate conversation, share a random thought, or make a natural in-character observation "
                                    "tied to the time of day or recent server vibe. Keep it to a single compact line or paragraph with no blank lines. "
                                    "DO NOT output [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [VOICE_MSG:]."
                                )
                        else:
                            loop_directive_text = (
                                "System Directive: Spontaneously initiate conversation, share a random thought, or make a natural in-character observation "
                                "tied to the time of day or recent server vibe. Keep it to a single compact line or paragraph with no blank lines. "
                                "DO NOT output [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [VOICE_MSG:]."
                            )

                        api_messages = build_api_messages_from_history(
                            dynamic_system, hist[-context_window:], client.user.id, client.user.display_name
                        )
                        api_messages.append({
                            "role": "system",
                            "content": loop_directive_text
                        })

                        payload = {
                            "model": text_model,
                            "messages": api_messages,
                            "temperature": conf.get("temperature", temperature),
                            "top_p": top_p,
                            "frequency_penalty": 0.42,
                            "presence_penalty": 0.30,
                            "max_tokens": output_length
                        }
                        res = await post_with_backoff(
                            api_url,
                            headers={"Authorization": f"Bearer {api_key}"},
                            json_payload=payload,
                            timeout=45,
                            max_retries=2,
                            bot_name=client.user.display_name
                        )

                        if res and res.status_code == 200:
                            reply = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                            if reply:
                                _, reply = extract_and_strip_thought_tag(reply)
                                _, reply = extract_and_strip_voice_msg(reply)
                                reply = ERROR_REPLY_STRIP_REGEX.sub('', reply).strip()
                                reply = LEARN_USER_STRIP_REGEX.sub('', reply).strip()
                                reply = LEARN_GENERAL_STRIP_REGEX.sub('', reply).strip()
                                reply = LEARN_STRIP_REGEX.sub('', reply).strip()
                                reply = EMOJI_REACT_STRIP_REGEX.sub('', reply).strip()
                                reply = INTERNAL_REACT_STRIP_REGEX.sub('', reply).strip()
                                reply = IMAGE_DESC_STRIP_REGEX.sub('', reply).strip()
                                reply = GIF_STRIP_REGEX.sub('', reply).strip()
                                reply = MEME_GEN_REGEX.sub('', reply).strip()
                                reply = UNCLOSED_MEME_GEN_REGEX.sub('', reply).strip()
                                reply = IMAGE_GEN_REGEX.sub('', reply).strip()
                                reply = UNCLOSED_IMAGE_GEN_REGEX.sub('', reply).strip()
                                reply = clean_and_compact_reply(reply)

                                if reply:
                                    sent_msg = None
                                    try:
                                        sent_msg = await channel.send(reply)
                                    except Exception:
                                        pass
                                    entry_id = sent_msg.id if sent_msg else f"random_{time.time()}_{client.user.id}"

                                    channel_followup_state[ch_id_int] = {
                                        "timestamp": time.time(),
                                        "turns_used": 0,
                                        "active": bool(max_followup_turns > 0),
                                        "bot_msg_id": entry_id
                                    }
                                    upsert_chat_memory_entry(
                                        chat_memory_file,
                                        ch_id_int,
                                        {
                                            "role": "assistant",
                                            "content": reply,
                                            "author_id": client.user.id,
                                            "author_name": client.user.display_name,
                                            "message_id": entry_id,
                                            "author_is_bot": True
                                        },
                                        max_history_length
                                    )
            except Exception as e:
                print(f"[{client.user.display_name}] [ERROR] Random post loop error: {e}")

    @client.event
    async def on_ready():
        nonlocal random_loop_started, extracted_voice_role, extracted_voice_style, slash_tree_synced
        print(f"[STARTUP] Bot authenticated as {client.user.display_name} (ID: {client.user.id})")
        register_active_bot(shared_image_cache_file, client.user.id, client.user.display_name, keywords)
        if hub_cmd_tree is not None and not slash_tree_synced:
            slash_tree_synced = True
            async def _sync_slash_bg():
                try:
                    # 1. Instant Guild-Level Sync (0-second propagation in all active servers!)
                    for g in list(getattr(client, "guilds", [])):
                        try:
                            hub_cmd_tree.copy_global_to(guild=g)
                            await hub_cmd_tree.sync(guild=g)
                        except Exception as g_err:
                            print(f"[{client.user.display_name}] [WARN] Guild slash sync warn ({getattr(g, 'name', g.id)}): {g_err}")
                    # 2. Clear global commands so Discord never shows BOTH a global and guild duplicate of '/hub'!
                    hub_cmd_tree.clear_commands(guild=None)
                    await hub_cmd_tree.sync()
                    print(f"[{client.user.display_name}] [SLASH] Synced single '/hub' slash command across {len(getattr(client, 'guilds', []))} server(s) (global duplicates cleared)!")
                except Exception as s_err:
                    print(f"[{client.user.display_name}] [WARN] Slash command sync skipped: {s_err}")
            client.loop.create_task(_sync_slash_bg())

        auto_role, auto_style = extract_persona_voice_defaults(system_prompt.get("content", ""), client.user.display_name)
        if not extracted_voice_role:
            extracted_voice_role = auto_role
        if not extracted_voice_style:
            extracted_voice_style = auto_style

        # Auto-heal persistent memory files on startup by pruning any ephemeral meme/chat recaps
        def _clean_ltm_list(lst):
            if not isinstance(lst, list):
                return []
            return [f for f in lst if isinstance(f, str) and not is_ephemeral_or_banter_fact(f)]

        def _clean_uprof_dict(d):
            if not isinstance(d, dict):
                return {}
            for u_key in list(d.keys()):
                if isinstance(d[u_key], list):
                    d[u_key] = [f for f in d[u_key] if isinstance(f, str) and not is_ephemeral_or_banter_fact(f)]
            return d

        update_json_file(long_term_memory_file, [], _clean_ltm_list)
        update_json_file(general_memory_file, [], _clean_ltm_list)

        def _clean_and_seed_uprof(d):
            if not isinstance(d, dict):
                d = {}
            for u_key in list(d.keys()):
                if isinstance(d[u_key], list):
                    d[u_key] = [f for f in d[u_key] if isinstance(f, str) and not is_ephemeral_or_banter_fact(f)]
            # Seed initial fellow-bot relationships from persona bio if not yet logged
            other_b_init = get_other_active_bots(shared_image_cache_file, client.user.id)
            from memory_and_context import infer_persona_affinity_seed, _format_rapport_entry
            bio_txt = system_prompt.get("content", "")
            for ob in other_b_init:
                ob_name = str(ob.get("display_name", "") or "").strip()
                if not ob_name:
                    continue
                canon_ob = find_canonical_username(ob_name, d, {ob_name})
                existing_ob = d.get(canon_ob, [])
                if not existing_ob:
                    seed_sc, seed_cat = infer_persona_affinity_seed(bio_txt, ob_name)
                    if seed_sc != 0 or seed_cat:
                        d[canon_ob] = [_format_rapport_entry(seed_sc, "Persona Bond" if seed_sc > 0 else "Comedic Foil", seed_cat)]
            return d

        update_json_file(user_profiles_file, {}, _clean_and_seed_uprof)

        if not random_loop_started:
            random_loop_started = True
            client.loop.create_task(random_post_loop())

    @client.event
    async def on_message(message: discord.Message):
        if message.author == client.user:
            return

        channel_id = message.channel.id

        if message.author.bot and not message.attachments and not message.embeds and not getattr(message, "stickers", []):
            raw_bot_content = message.content or ""
            bot_glitch_summary = extract_glitch_summary(raw_bot_content)
            stripped_bot_msg = ERROR_REPLY_STRIP_REGEX.sub('', raw_bot_content).strip()
            if not stripped_bot_msg:
                if bot_glitch_summary:
                    upsert_chat_memory_entry(
                        chat_memory_file,
                        channel_id,
                        {
                            "role": "system",
                            "content": (
                                f"[System Glitch Event: {message.author.display_name} just glitched out in chat ({bot_glitch_summary}). "
                                "Other bots may tease, roast, or comment on this glitch in-character, but NEVER output raw '*(API Error...)*' text.]"
                            ),
                            "message_id": message.id,
                            "author_id": message.author.id,
                            "author_name": message.author.display_name,
                            "is_glitch_event": True
                        },
                        max_history_length
                    )
                return

        is_admin = bool(
            message.guild
            and getattr(getattr(message.author, "guild_permissions", None), "administrator", False)
        )
        content_lower = (message.content or "").lower()

        # ======================================================================
        # COMMAND SUITE (Delegated to commands_and_embeds.py)
        # ======================================================================
        if await handle_bot_commands(
            message=message,
            client=client,
            is_admin=is_admin,
            channel_id=channel_id,
            system_prompt=system_prompt,
            chat_memory_file=chat_memory_file,
            shared_image_cache_file=shared_image_cache_file,
            long_term_memory_file=long_term_memory_file,
            user_profiles_file=user_profiles_file,
            channel_config_file=channel_config_file,
            general_memory_file=general_memory_file,
            channel_followup_state=channel_followup_state,
            keywords=keywords,
            chance_to_respond=chance_to_respond,
            random_emoji_chance=random_emoji_chance,
            default_random_interval=default_random_interval,
            temperature=temperature,
            enable_vision=enable_vision,
            enable_image_gen=enable_image_gen,
            enable_voice=enable_voice,
            enable_gifs=enable_gifs,
            welcome_test_callback=trigger_welcome_greeting,
        ):
            return

        # ======================================================================
        # CORE EVENT ROUTING
        # ======================================================================
        conf = load_json_file(channel_config_file, {}).get(channel_id, {})
        if not conf.get("enabled", True):
            if client.user.mentioned_in(message):
                try:
                    await message.channel.send(f"*(🔇 {client.user.display_name} is currently disabled in this channel.)*")
                except Exception:
                    pass
            return

        current_resp_chance = conf.get("resp_chance", chance_to_respond)
        current_emoji_chance = conf.get("emoji_chance", random_emoji_chance)
        current_temp = conf.get("temperature", temperature)
        ch_enable_vision = conf.get("enable_vision", enable_vision)
        ch_enable_gifs = conf.get("enable_gifs", enable_gifs)
        ch_enable_image_gen = conf.get("enable_image", enable_image_gen)
        ch_enable_voice = conf.get("enable_voice", enable_voice)
        is_react_only = conf.get("react_only", False)
        is_author_bot = message.author.bot

        bot_stagger = ((client.user.id >> 22) % 4) * 0.11
        if is_author_bot:
            await asyncio.sleep(0.45 + bot_stagger)
        elif bot_stagger > 0:
            await asyncio.sleep(bot_stagger)

        replied_to_self, reply_context_str, ref_message = await resolve_reply_context(
            message, client.user.id, shared_image_cache_file
        )

        incoming_glitch_summary = extract_glitch_summary(message.content or "") if is_author_bot else ""
        clean_content = ERROR_REPLY_STRIP_REGEX.sub('', message.content or '').strip()
        for user in message.mentions:
            clean_content = clean_content.replace(f"<@{user.id}>", f"@{user.display_name}")
            clean_content = clean_content.replace(f"<@!{user.id}>", f"@{user.display_name}")
        for ch_mention in getattr(message, "channel_mentions", []):
            clean_content = clean_content.replace(f"<#{ch_mention.id}>", f"#{getattr(ch_mention, 'name', 'channel')}")
        for role_mention in getattr(message, "role_mentions", []):
            clean_content = clean_content.replace(f"<@&{role_mention.id}>", f"@{getattr(role_mention, 'name', 'role')}")

        has_keyword = any(
            re.search(rf'\b{_term_boundary_pattern(kw.lower())}', clean_content.lower())
            for kw in (keywords or []) if kw and kw.strip()
        )
        directed_at_other_bot = is_command_directed_at_other_bot(
            clean_content, message, client.user.id, client.user.display_name, keywords, shared_image_cache_file
        )
        addressed_bot_name = detect_addressed_bot_name(
            clean_content, message, client.user.id, client.user.display_name, keywords, shared_image_cache_file
        )
        if not addressed_bot_name and ref_message and getattr(ref_message.author, "bot", False):
            addressed_bot_name = ref_message.author.display_name.strip()

        chat_memory = load_json_file(chat_memory_file, {})
        chat_history_list = chat_memory.get(channel_id, [])

        user_profiles_snapshot = load_json_file(user_profiles_file, {})
        speaker_affinity = get_user_affinity_score(
            user_profiles_snapshot,
            message.author.display_name,
            persona_prompt=system_prompt.get("content", "")
        )

        addressed_by_fellow_bot = (
            is_directly_addressed_by_fellow_bot(
                clean_content, message, client.user.id, client.user.display_name,
                keywords, shared_image_cache_file, replied_to_self=replied_to_self
            )
            if is_author_bot
            else False
        )
        is_explicit_trigger = client.user.mentioned_in(message) or has_keyword or replied_to_self or addressed_by_fellow_bot

        if is_explicit_trigger and not directed_at_other_bot and not is_author_bot:
            channel_followup_state[channel_id] = {
                "timestamp": time.time(),
                "turns_used": 0,
                "active": True,
                "bot_msg_id": "",
                "user_id": message.author.id
            }
            def _set_active_followup_owner(c):
                f_map = c.get("_active_followup_bot", {})
                if not isinstance(f_map, dict):
                    f_map = {}
                msg_owners = c.get("_msg_addressed_bots", {})
                if not isinstance(msg_owners, dict):
                    msg_owners = {}
                m_key = str(message.id)
                cur_list = msg_owners.get(m_key, [])
                if not isinstance(cur_list, list):
                    cur_list = []
                if client.user.id not in cur_list:
                    cur_list.append(client.user.id)
                msg_owners[m_key] = cur_list
                if len(msg_owners) > 50:
                    for old_k in list(msg_owners.keys())[:-50]:
                        msg_owners.pop(old_k, None)
                c["_msg_addressed_bots"] = msg_owners
                f_map[channel_id] = cur_list if len(cur_list) > 1 else client.user.id
                c["_active_followup_bot"] = f_map
                return c
            update_json_file(shared_image_cache_file, {}, _set_active_followup_owner)

        is_contextual_followup = False
        followup_context_str = ""
        if not is_explicit_trigger and not is_author_bot and not directed_at_other_bot:
            is_contextual_followup, followup_context_str = await resolve_contextual_followup(
                message=message,
                clean_content=clean_content,
                chat_history_list=chat_history_list,
                my_bot_id=client.user.id,
                my_display_name=client.user.display_name,
                my_keywords=keywords,
                shared_image_cache_file=shared_image_cache_file,
                channel_followup_state=channel_followup_state,
                followup_window_messages=followup_window_messages,
                max_followup_turns=max_followup_turns,
                followup_timeout=followup_timeout,
                api_key=api_key,
                text_model=text_model,
                api_url=api_url,
                speaker_affinity=speaker_affinity
            )
            if is_contextual_followup and not reply_context_str:
                reply_context_str = followup_context_str

        is_relevant_trigger = (is_explicit_trigger and not directed_at_other_bot) or is_contextual_followup

        bot_streak = 0
        if is_author_bot:
            for msg in reversed(chat_history_list):
                if msg.get("author_is_bot"):
                    bot_streak += 1
                else:
                    break

        # Allow a responsive 60s cooldown when a fellow bot directly addresses/replies to this bot,
        # while keeping a 240s cooldown for unprompted bot-to-bot chime-ins (capped at 2 consecutive bot turns)
        bot_to_bot_cooldown = 60 if addressed_by_fellow_bot else 240
        time_since_last_b2b = time.time() - conf.get("last_bot_to_bot", 0)

        has_media_candidate = is_visual_media_candidate(message)
        has_voice_note_input = (not is_author_bot) and is_audio_voice_attachment(message)
        voice_transcript_text = ""
        if has_voice_note_input and not directed_at_other_bot:
            audio_raw, audio_fname = await fetch_audio_bytes_from_message(message)
            if audio_raw:
                voice_transcript_text = await transcribe_voice_attachment(
                    audio_raw, filename=audio_fname, api_key=api_key, bot_name=client.user.display_name
                )
                if voice_transcript_text:
                    clean_content = (
                        f"{clean_content}\n[Spoken Voice Note Transcript: '{voice_transcript_text}']".strip()
                        if clean_content
                        else f"[Spoken Voice Note Transcript: '{voice_transcript_text}']"
                    )

        spontaneous_mult = compute_spontaneous_interest_multiplier(
            clean_content=clean_content,
            has_media=(has_media_candidate or bool(voice_transcript_text)),
            is_author_bot=is_author_bot,
            persona_prompt=system_prompt.get("content", ""),
            chat_history_list=chat_history_list,
            my_bot_id=client.user.id,
            speaker_affinity=speaker_affinity
        )
        effective_resp_chance = min(0.85, current_resp_chance * spontaneous_mult)

        will_respond = False
        if is_react_only:
            if client.user.mentioned_in(message) and not directed_at_other_bot:
                try:
                    await message.channel.send(f"*(🤐 {client.user.display_name} is in react-only mode here.)*")
                except Exception:
                    pass
        elif is_author_bot and (bot_streak >= 2 or time_since_last_b2b < bot_to_bot_cooldown):
            pass
        elif is_relevant_trigger:
            will_respond = True
        elif (
            not directed_at_other_bot
            and not is_other_bot_in_active_exchange(chat_history_list, client.user.id, window_seconds=18.0)
            and random.random() < effective_resp_chance
        ):
            # Atomically claim the channel's 15-second unprompted chime-in cooldown slot
            # so multiple bots never dogpile the exact same unprompted message simultaneously
            claimed_slot = False

            def _try_claim_unprompted(c):
                nonlocal claimed_slot
                chimes = c.get("_last_unprompted_chime", {})
                if not isinstance(chimes, dict):
                    chimes = {}
                last_chime_ts = float(chimes.get(channel_id, chimes.get(str(channel_id), 0)) or 0)
                if (time.time() - last_chime_ts) >= 15.0:
                    chimes[channel_id] = time.time()
                    c["_last_unprompted_chime"] = chimes
                    claimed_slot = True
                return c

            update_json_file(shared_image_cache_file, {}, _try_claim_unprompted)
            will_respond = claimed_slot

        # Vision Prep
        image_desc = ""
        image_bytes = None

        if ch_enable_vision and has_media_candidate:
            msg_id = message.id
            claimed_vision = False
            cached_val = None

            def _try_claim_vision_lock(c):
                nonlocal claimed_vision, cached_val
                existing = c.get(msg_id, c.get(str(msg_id)))
                now_v = time.time()
                if isinstance(existing, str) and existing and not existing.startswith("PROCESSING"):
                    cached_val = existing
                    return c
                if isinstance(existing, str) and existing.startswith("PROCESSING"):
                    parts = existing.split("|")
                    lock_t = float(parts[1]) if len(parts) > 1 else now_v
                    if (now_v - lock_t) < 22.0:
                        cached_val = existing
                        return c
                c[msg_id] = f"PROCESSING|{now_v}|{client.user.id}"
                claimed_vision = True
                return c

            update_json_file(shared_image_cache_file, {}, _try_claim_vision_lock)

            if claimed_vision:
                image_bytes = await fetch_image_bytes_from_message(message)
                if image_bytes:
                    image_desc = await summarize_image_for_memory(image_bytes, api_key, vision_model, api_url, shared_image_cache_file=shared_image_cache_file)

                def _save_vision_desc(c):
                    if image_desc:
                        c[msg_id] = image_desc
                    else:
                        c.pop(msg_id, None)
                        c.pop(str(msg_id), None)
                    numeric_keys = sorted([k for k in c.keys() if isinstance(k, int)])
                    if len(numeric_keys) > 500:
                        for old_k in numeric_keys[:-500]:
                            c.pop(old_k, None)
                    return c
                update_json_file(shared_image_cache_file, {}, _save_vision_desc)
            else:
                if isinstance(cached_val, str) and cached_val.startswith("PROCESSING"):
                    for _ in range(18):
                        await asyncio.sleep(0.65)
                        img_cache = load_json_file(shared_image_cache_file, {})
                        cached_val = img_cache.get(msg_id, img_cache.get(str(msg_id)))
                        if not (isinstance(cached_val, str) and cached_val.startswith("PROCESSING")):
                            break
                if isinstance(cached_val, str) and cached_val and not cached_val.startswith("PROCESSING"):
                    image_desc = cached_val
                if will_respond:
                    image_bytes = await fetch_image_bytes_from_message(message)

        if will_respond and ch_enable_vision and not image_bytes and ref_message:
            if is_visual_media_candidate(ref_message):
                image_bytes = await fetch_image_bytes_from_message(ref_message)

        # Split-Message Media Lookback: Check in-RAM chat_history_list first so we only call Discord's
        # channel.history REST endpoint when the current author actually posted within the last 60s!
        recent_same_author_in_ram = False
        now_check_ts = time.time()
        for prev_entry in reversed(chat_history_list[-3:]):
            if prev_entry.get("role") == "system":
                continue
            if prev_entry.get("author_id") == message.author.id:
                p_ts = prev_entry.get("timestamp")
                if p_ts is None or (now_check_ts - float(p_ts)) <= 65.0:
                    recent_same_author_in_ram = True
            break

        if will_respond and ch_enable_vision and not image_bytes and recent_same_author_in_ram:
            try:
                async for prev_m in message.channel.history(limit=3, before=message):
                    if (message.created_at - prev_m.created_at).total_seconds() > 60:
                        break
                    if prev_m.author.id != message.author.id:
                        break
                    prev_has_media = is_visual_media_candidate(prev_m)
                    if prev_has_media:
                        img_cache_chk = load_json_file(shared_image_cache_file, {})
                        prev_desc = img_cache_chk.get(prev_m.id, "")
                        if isinstance(prev_desc, str) and prev_desc.startswith("PROCESSING"):
                            for _ in range(16):
                                await asyncio.sleep(0.6)
                                img_cache_chk = load_json_file(shared_image_cache_file, {})
                                prev_desc = img_cache_chk.get(prev_m.id, "")
                                if not (isinstance(prev_desc, str) and prev_desc.startswith("PROCESSING")):
                                    break

                        image_bytes = await fetch_image_bytes_from_message(prev_m)
                        if not prev_desc and image_bytes:
                            prev_desc = await summarize_image_for_memory(image_bytes, api_key, vision_model, api_url, shared_image_cache_file=shared_image_cache_file)
                            if prev_desc:
                                def _save_prev_desc(c, pid=prev_m.id, pd=prev_desc):
                                    c[pid] = pd
                                    return c
                                update_json_file(shared_image_cache_file, {}, _save_prev_desc)

                        if prev_desc:
                            prev_line = f"[{prev_m.author.display_name}]: {prev_m.content.strip()}" if prev_m.content.strip() else f"[{prev_m.author.display_name} uploaded media]"
                            upsert_chat_memory_entry(
                                chat_memory_file,
                                channel_id,
                                {
                                    "role": "user",
                                    "content": f"{prev_line}\n{prev_desc}".strip(),
                                    "message_id": prev_m.id,
                                    "author_id": prev_m.author.id,
                                    "author_is_bot": prev_m.author.bot,
                                    "author_name": prev_m.author.display_name
                                },
                                max_history_length
                            )
                        break
            except Exception:
                pass

        gif_slug = extract_gif_slug(clean_content) if ch_enable_gifs else ""
        yt_meta = await get_youtube_metadata(clean_content)
        web_embed_meta = extract_webpage_embed_context(message) if not image_desc else ""

        shared_text_line = f"[{message.author.display_name}]: {clean_content.strip()}" if clean_content.strip() else f"[{message.author.display_name} uploaded media]"
        if reply_context_str:
            shared_text_line = f"{reply_context_str}\n{shared_text_line}"

        emoji_text_context = shared_text_line
        if image_desc:
            emoji_text_context += f"\n{image_desc}"
        if gif_slug:
            emoji_text_context += f"\n[Contains GIF of: {gif_slug}]"
        if yt_meta:
            emoji_text_context += f"\n{yt_meta}"
        if web_embed_meta:
            emoji_text_context += f"\n{web_embed_meta}"

        guild_members = getattr(message.guild, "members", None) if message.guild else None
        other_bots_meta = get_other_active_bots(shared_image_cache_file, client.user.id, guild_members)
        my_own_kw_set = {k.strip().lower() for k in (keywords or []) if k and k.strip()}
        for part in re.findall(r'[a-z0-9]+', client.user.display_name.lower()):
            if len(part) >= 3 and part not in ("bot", "guild", "the"):
                my_own_kw_set.add(part)

        other_bot_kw_set = set()
        other_bot_names_set = set()
        other_bot_display_list = []
        for ob in other_bots_meta:
            if ob.get("display_name"):
                other_bot_names_set.add(ob["display_name"].lower())
                other_bot_display_list.append(ob["display_name"])
            for kw_item in ob.get("keywords", []):
                kw_l = kw_item.strip().lower()
                if len(kw_l) >= 3 and kw_l not in my_own_kw_set:
                    other_bot_kw_set.add(kw_l)
            for al_item in ob.get("aliases", []):
                al_low = al_item.lower()
                other_bot_names_set.add(al_low)
                if "'s " not in al_low and "’s " not in al_low and "bots " not in al_low:
                    first_tok = al_low.split()[0] if al_low.split() else ""
                    if len(first_tok) >= 3 and first_tok not in ("bot", "guild", "the") and first_tok not in my_own_kw_set:
                        other_bot_kw_set.add(first_tok)

        bot_emoji_scale = (0.45 if abs(speaker_affinity) >= 15 else 0.25) if is_author_bot else min(1.45, spontaneous_mult)
        effective_emoji_chance = min(0.95, current_emoji_chance * bot_emoji_scale)
        reacted_emoji = None
        react_task = None
        should_react = (is_explicit_trigger and current_emoji_chance > 0 and (not is_author_bot or addressed_by_fellow_bot)) or (random.random() < effective_emoji_chance)

        if should_react:
            react_task = asyncio.create_task(
                auto_react(
                    message,
                    api_key=api_key,
                    personality_prompt=system_prompt["content"],
                    text_model=text_model,
                    vision_model=vision_model,
                    image_bytes=(None if image_desc else image_bytes),
                    api_url=api_url,
                    chat_history=chat_history_list,
                    text=emoji_text_context,
                    emoji_context_length=emoji_context_length,
                    bot_name=client.user.display_name,
                    other_bot_keywords=other_bot_kw_set,
                    emoji_reaction_hint=emoji_reaction_hint,
                    speaker_affinity=speaker_affinity
                )
            )

        text_context = shared_text_line
        if gif_slug:
            text_context += f"\n[Contains GIF of: {gif_slug}]"
        if yt_meta:
            text_context += f"\n{yt_meta}"
        if web_embed_meta:
            text_context += f"\n{web_embed_meta}"
        if image_desc:
            text_context += f"\n{image_desc}"

        msg_created_ts = getattr(getattr(message, "created_at", None), "timestamp", lambda: time.time())()
        user_mem_entry = {
            "role": "user",
            "content": text_context,
            "message_id": message.id,
            "author_id": message.author.id,
            "author_is_bot": is_author_bot,
            "author_name": message.author.display_name,
            "timestamp": msg_created_ts
        }
        if addressed_bot_name:
            user_mem_entry["addressed_bot"] = addressed_bot_name

        upsert_chat_memory_entry(
            chat_memory_file,
            channel_id,
            user_mem_entry,
            max_history_length
        )

        if incoming_glitch_summary:
            upsert_chat_memory_entry(
                chat_memory_file,
                channel_id,
                {
                    "role": "system",
                    "content": (
                        f"[System Glitch Event: {message.author.display_name}'s media module glitched ({incoming_glitch_summary}) on that message. "
                        "Bots may tease or comment on it in-character, but NEVER output raw '*(API Error...)*' text.]"
                    ),
                    "message_id": f"glitch_media_{message.id}",
                    "author_id": message.author.id,
                    "author_name": message.author.display_name,
                    "is_glitch_event": True
                },
                max_history_length
            )

        # ======================================================================
        # AI GENERATION PIPELINE
        # ======================================================================
        if will_respond:
            ch_lock = channel_gen_locks.setdefault(channel_id, asyncio.Lock())
            is_queued_waiter = False
            if ch_lock.locked():
                if (
                    not is_explicit_trigger
                    or channel_in_coalesce_window.get(channel_id, False)
                    or channel_pending_waiters.get(channel_id, 0) >= 1
                ):
                    print(f"[{client.user.display_name}] [INFO] Coalescing rapid message from {message.author.display_name} into active/queued channel turn.")
                    return
                channel_pending_waiters[channel_id] = channel_pending_waiters.get(channel_id, 0) + 1
                is_queued_waiter = True

            # Two-Wave Multi-Bot Coordination & Crowd Cap (3 responders max on 'hey bots', max 1 wait hop <= 2.2s)
            wave_allowed, wave_slot = await claim_or_coordinate_turn_wave(
                shared_image_cache_file=shared_image_cache_file,
                message_id=message.id,
                my_bot_id=client.user.id,
                my_display_name=client.user.display_name,
                clean_content=clean_content,
            )
            if not wave_allowed:
                print(f"[{client.user.display_name}] [CROWD CAP] Skipping collective turn (3-bot responder cap reached).")
                return

            async with ch_lock:
                if is_queued_waiter:
                    channel_pending_waiters[channel_id] = max(0, channel_pending_waiters.get(channel_id, 1) - 1)
                channel_in_coalesce_window[channel_id] = True
                try:
                    # Upgrade #8 — Velocity & Punctuation-Aware Adaptive Coalescing Window (0.12s vs 1.25s)
                    coalesce_delay = (
                        0.08
                        if wave_slot >= 1
                        else compute_adaptive_coalesce_delay(clean_content, is_author_bot=is_author_bot)
                    )
                    await asyncio.sleep(coalesce_delay)
                finally:
                    channel_in_coalesce_window[channel_id] = False

                if channel_last_replied_msg_id.get(channel_id, 0) >= message.id:
                    return

                if is_author_bot:
                    def _update_b2b(c):
                        c.setdefault(channel_id, {})["last_bot_to_bot"] = time.time()
                        return c
                    update_json_file(channel_config_file, {}, _update_b2b)

                trigger_mode_str = (
                    "contextual follow-up"
                    if is_contextual_followup
                    else ("spontaneous chime-in" if not is_explicit_trigger else "trigger")
                )
                print(f"[{client.user.display_name}] [INFO] Generating response ({trigger_mode_str}) to {message.author.display_name} in #{message.channel.name}")

                final_reply = ""
                final_gif = None
                sys_error = None
                is_error_reply = False
                image_file = None
                voice_file = None
                img_prompt = None
                meme_top_text = ""
                meme_bottom_text = ""
                voice_text = None
                gif_search_term = None
                img_err = None
                voice_err = None
                turn_glitch_tracker = []

                async with message.channel.typing():
                    try:
                        active_model = vision_model if (image_bytes and ch_enable_vision) else text_model
                        dynamic_system = copy.deepcopy(system_prompt)
                        persona_summary = dynamic_system.get("content", "")[:400]

                        chat_memory = load_json_file(chat_memory_file, {})
                        raw_history = copy.deepcopy(chat_memory.get(channel_id, [])[-context_window:])

                        now_str = datetime.now().strftime("%A, %I:%M %p")
                        dynamic_system["content"] += build_temporal_context_block(raw_history, now_str)
                        dynamic_system["content"] += build_glitch_awareness_rules(client.user.display_name)
                        dynamic_system["content"] += build_memory_attribution_rules(client.user.display_name, other_bot_display_list)
                        dynamic_system["content"] += build_media_and_meme_rules(ch_enable_image_gen, ch_enable_gifs)

                        coalesced_user_lines = []
                        for m in reversed(raw_history[-4:]):
                            if m.get("role") == "system":
                                continue
                            if m.get("author_is_bot") or m.get("role") == "assistant" or m.get("author_id") != message.author.id:
                                break
                            m_addr = m.get("addressed_bot")
                            if m_addr and m_addr.lower() != client.user.display_name.lower():
                                break
                            if m.get("role") == "user" and isinstance(m.get("content"), str):
                                user_typed_lines = []
                                for raw_ln in m["content"].split("\n"):
                                    ln_s = raw_ln.strip()
                                    if ln_s.startswith((
                                        "[Replying to ",
                                        "[Attached Media Description:",
                                        "[Contains GIF of:",
                                        "[Contains YouTube Video:",
                                        "[Linked Webpage:"
                                    )):
                                        continue
                                    ln_s = re.sub(r'^\[[^\]]+\]:\s*', '', ln_s).strip()
                                    if ln_s and not ln_s.endswith("uploaded media]"):
                                        user_typed_lines.append(ln_s)
                                if user_typed_lines:
                                    coalesced_user_lines.append(" ".join(user_typed_lines))
                        coalesced_user_lines.reverse()
                        combined_recent_prompt = "\n".join(coalesced_user_lines) if coalesced_user_lines else clean_content

                        # Load pre-computed semantic_tags from _room_state for Hybrid BM25 + Ebbinghaus RAG (#4)
                        scache_turn = load_json_file(shared_image_cache_file, {})
                        room_sem_tags = (
                            scache_turn.get("_room_state", {}).get(str(channel_id), {}).get("semantic_tags", [])
                            if isinstance(scache_turn.get("_room_state", {}).get(str(channel_id)), dict)
                            else []
                        )
                        active_topic_query = f"{combined_recent_prompt} {reply_context_str}".strip()
                        relevant_ltm = select_relevant_memories(
                            load_json_file(long_term_memory_file, []),
                            active_topic_query,
                            max_items=5,
                            semantic_tags=room_sem_tags,
                        )
                        if relevant_ltm:
                            dynamic_system["content"] += (
                                "\n\n### Topically Relevant Server Lore (PASSIVE ARCHIVE — only reference if directly relevant to the user's current message):\n"
                                + "\n".join(f"- {f}" for f in relevant_ltm)
                            )

                        relevant_gen = select_relevant_memories(
                            load_json_file(general_memory_file, []),
                            active_topic_query,
                            max_items=5,
                            semantic_tags=room_sem_tags,
                        )
                        if relevant_gen:
                            dynamic_system["content"] += (
                                "\n\n### Topically Relevant General Knowledge (PASSIVE ARCHIVE — only reference if directly relevant to the user's current message):\n"
                                + "\n".join(f"- {f}" for f in relevant_gen)
                            )

                        user_profiles = load_json_file(user_profiles_file, {})
                        recent_users = set(msg.get("author_name") for msg in raw_history if msg.get("author_name"))
                        for ob_disp in other_bot_display_list:
                            if ob_disp:
                                recent_users.add(ob_disp)
                        for known_user in user_profiles.keys():
                            if str(known_user).lower() in combined_recent_prompt.lower() or str(known_user).lower() in reply_context_str.lower():
                                recent_users.add(str(known_user))

                        active_user_block = build_active_user_context_block(
                            user_profiles=user_profiles,
                            raw_history=raw_history,
                            current_speaker=message.author.display_name,
                            combined_recent_prompt=combined_recent_prompt,
                            reply_context_str=reply_context_str,
                            is_spontaneous=(not is_relevant_trigger),
                            persona_prompt=system_prompt.get("content", ""),
                            other_bot_names=other_bot_display_list,
                            my_bot_name=client.user.display_name,
                            shared_cache_file=shared_image_cache_file,
                            channel_id=channel_id,
                        )
                        if active_user_block:
                            dynamic_system["content"] += active_user_block

                        img_cache = load_json_file(shared_image_cache_file, {})
                        gen_records = [
                            r for r in img_cache.get("_generated_images", [])
                            if isinstance(r, dict) and r.get("channel_id") == channel_id and (time.time() - r.get("timestamp", 0)) < 1800
                        ]
                        self_recently_generated = any(
                            r.get("bot_id") == client.user.id and (time.time() - r.get("timestamp", 0)) < 75
                            for r in gen_records
                        )
                        if self_recently_generated:
                            dynamic_system["content"] += (
                                "\n\n- **NO PHANTOM IMAGE PROMISES (CRITICAL)**: You generated an image recently. "
                                "If you are replying with text only, NEVER say 'here is your...', 'let me fix that...', 'coming right up', or pretend to show/redraw an image! "
                                "If the user corrected your previous image and you DO decide to redraw/fix it for them, you MUST include the full `[IMAGE_GEN: ...]` tag at the end of your message."
                            )

                        api_messages = build_api_messages_from_history(
                            dynamic_system, raw_history, client.user.id, client.user.display_name
                        )

                        if is_contextual_followup:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    f"[System Context Note: {message.author.display_name} is following up directly on what you ({client.user.display_name}) just said or showed in chat: \"{clean_content[:220]}\". "
                                    "Reply naturally in-character to keep the conversation flowing.]"
                                )
                            })
                        elif not is_explicit_trigger:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    f"[System Context Note: Nobody directly pinged you—you are spontaneously chiming in ONLY on {message.author.display_name}'s specific message: \"{clean_content[:220]}\". "
                                    f"React directly to what {message.author.display_name} just said in 1-2 short, natural sentences. "
                                    "Do NOT bring up unrelated topics, old memes, or stock example quotes from your bio!]"
                                )
                            })

                        # Phase 3 — Image-to-Image Style Remixing Detection
                        remix_source_desc = image_desc
                        if not remix_source_desc and reply_context_str:
                            ref_desc_m = IMAGE_DESC_EXTRACT_REGEX.search(reply_context_str)
                            if ref_desc_m:
                                remix_source_desc = f"[Attached Media Description: {ref_desc_m.group(1).strip()}]"

                        is_image_remix_req = bool(
                            ch_enable_image_gen
                            and not directed_at_other_bot
                            and remix_source_desc
                            and IMAGE_REMIX_REQ_REGEX.search(combined_recent_prompt or "")
                        )
                        is_explicit_meme_req = bool(
                            ch_enable_image_gen
                            and not directed_at_other_bot
                            and EXPLICIT_MEME_REQ_REGEX.search(combined_recent_prompt)
                        )
                        is_explicit_img_req = bool(
                            ch_enable_image_gen
                            and not is_explicit_meme_req
                            and not directed_at_other_bot
                            and (EXPLICIT_IMAGE_REQ_REGEX.search(combined_recent_prompt) or is_image_remix_req)
                        )
                        is_explicit_voice_req = bool(
                            ch_enable_voice
                            and not directed_at_other_bot
                            and (EXPLICIT_VOICE_REQ_REGEX.search(combined_recent_prompt) or bool(voice_transcript_text))
                        )
                        is_explicit_gif_req = bool(
                            ch_enable_gifs
                            and not is_explicit_img_req
                            and not is_explicit_meme_req
                            and not directed_at_other_bot
                            and EXPLICIT_GIF_REQ_REGEX.search(combined_recent_prompt)
                        )

                        if directed_at_other_bot:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    "System Directive: The user's media request is addressed to another bot while mentioning you as the subject. "
                                    "Respond with a short, witty in-character remark, and DO NOT output [IMAGE_GEN:], [MEME_GEN:], or [VOICE_MSG:] so the requested bot can handle the generation."
                                )
                            })
                        elif is_explicit_meme_req:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    "System Directive: The user is asking you to generate a meme! "
                                    "You MUST append `[MEME_GEN: vivid visual description of the scene without text | <witty setup line> | <witty punchline>]` at the end of your response. "
                                    "Both captions MUST be real, original joke lines tied to the topic (NEVER write literal words like 'TOP CAPTION' or 'ABSOLUTELY UNACCEPTABLE'). Do NOT output `[GIF:]` or `[IMAGE_GEN:]` in the same message."
                                )
                            })
                        elif is_image_remix_req:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    f"System Directive (Image-to-Image Remix): The user is asking you to remix/redraw the attached or replied-to visual ({remix_source_desc})! "
                                    "You MUST append `[IMAGE_GEN: detailed 30-50 word visual description]` at the end of your response that preserves the core subject/composition of the source image while applying the user's requested style, twist, or modification. "
                                    "Do NOT output `[GIF:]` in the same message."
                                )
                            })
                        elif is_explicit_img_req:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    "System Directive: The user is explicitly asking you to generate an image. "
                                    "You MUST append a creative, faithful `[IMAGE_GEN: detailed visual description]` tag (WITHOUT meme text captions) at the end of your response that includes every subject, item, and detail the user requested. "
                                    "Do NOT output `[GIF:]` or `[MEME_GEN:]` in the same message."
                                )
                            })
                        elif is_explicit_voice_req:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    "System Directive: The user is explicitly asking you for a voice clip. "
                                    "Include normal chat text AND a complete `[VOICE_MSG: ...]` block at the end of your response with Role, Scene, Guidance, and spoken dialogue."
                                )
                            })
                        elif is_explicit_gif_req:
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    "System Directive: The user is asking for a GIF. "
                                    "Append `[GIF: concise 2-4 word search query]` at the end of your response."
                                )
                            })

                        if PERSONAL_FACT_STATEMENT_REGEX.search(combined_recent_prompt or ""):
                            api_messages.append({
                                "role": "system",
                                "content": (
                                    f"System Memory Reminder: {message.author.display_name} just shared a personal detail, preference, project, or situation! "
                                    f"Append `[LEARN_USER: {message.author.display_name} | specific 8-20 word objective fact]` (or `[LEARN: ...]` for a group/bot situation) at the very end of your response so you remember it."
                                )
                            })

                        # Inject 0ms Deterministic AST Math / Unit / Python Syntax Grounding Tools
                        grounding_note = evaluate_deterministic_grounding_tools(combined_recent_prompt)
                        if grounding_note:
                            api_messages.append({"role": "system", "content": grounding_note})

                        text_only_messages = copy.deepcopy(api_messages)

                        if image_bytes and ch_enable_vision:
                            b64_image = base64.b64encode(image_bytes).decode("utf-8")
                            for i in range(len(api_messages) - 1, -1, -1):
                                if api_messages[i].get("role") == "user":
                                    api_messages[i]["content"] = [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"{api_messages[i]['content']}\n"
                                                "[System Visual Note: Focus directly on the attached image/animation above! If it shows a 2x2 storyboard with small 1-2-3-4 corner badges, it is a chronological sequence of keyframes from a single video or animated GIF. Treat it as one continuous animation/video clip, not a 4-panel comic.]"
                                            )
                                        },
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                                    ]
                                    break

                        # Upgrade #12 — Situational Sampling Modulation (Tech=0.45/0.85, Banter=0.85/0.92, Standard=base)
                        turn_temp, turn_top_p = compute_situational_sampling_params(
                            combined_recent_prompt,
                            base_temp=current_temp,
                            base_top_p=top_p,
                            is_meme_or_roast=is_explicit_meme_req,
                        )

                        payload = {
                            "model": active_model,
                            "messages": api_messages,
                            "temperature": turn_temp,
                            "top_p": turn_top_p,
                            "frequency_penalty": 0.42,
                            "presence_penalty": 0.30,
                            "max_tokens": output_length
                        }

                        endpoint_label = api_url.rsplit("/", 1)[-1] if api_url else "completions"
                        primary_circuit_open = is_model_circuit_open(f"{endpoint_label}:{active_model}")
                        res = None
                        if primary_circuit_open:
                            turn_glitch_tracker.append(f"circuit breaker open on {active_model}")
                            print(f"[{client.user.display_name}] [CIRCUIT BREAKER] Fast-failing over from '{active_model}' to EWMA-ranked backup model...")
                        else:
                            res = await post_with_backoff(
                                api_url,
                                headers={"Authorization": f"Bearer {api_key}"},
                                json_payload=payload,
                                timeout=(8, 22),
                                max_retries=2,
                                bot_name=client.user.display_name,
                                glitch_tracker=turn_glitch_tracker,
                                inject_glitch_on_retry=True
                            )

                        reply = ""
                        raw_reply = ""
                        bot_thought_text = None
                        intentional_silence = False
                        if res is not None and res.status_code == 200:
                            raw_reply = res.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                            if not is_explicit_trigger and INTENTIONAL_NO_RESPONSE_REGEX.search(raw_reply):
                                intentional_silence = True
                            reply = ERROR_REPLY_STRIP_REGEX.sub('', raw_reply).strip()

                        if intentional_silence:
                            mark_turn_wave_complete(shared_image_cache_file, message.id, client.user.id)
                            print(f"[{client.user.display_name}] [INFO] Model chose intentional silence on non-explicit turn; skipping reply.")
                            return

                        should_try_fallback = (
                            not reply
                            and not intentional_silence
                            and (res is None or res.status_code in (200, 429, 500, 502, 503, 504))
                        )

                        if should_try_fallback and image_bytes and ch_enable_vision:
                            for fb_v_model in rank_models_by_ewma_latency(FALLBACK_VISION_MODELS, endpoint_label):
                                if fb_v_model == active_model:
                                    continue
                                print(f"[{client.user.display_name}] [VISION FALLBACK] Trying '{fb_v_model}'...")
                                fb_v_payload = copy.deepcopy(payload)
                                fb_v_payload["model"] = fb_v_model
                                glitch_reason = "; ".join(turn_glitch_tracker[-2:]) if turn_glitch_tracker else "primary model timeout"
                                fb_v_payload["messages"].append({
                                    "role": "system",
                                    "content": (
                                        f"[System Glitch Context: Your primary vision model ({active_model}) just glitched ({glitch_reason}), "
                                        f"so you recovered on backup model ({fb_v_model}). You may briefly joke or grumble in-character about "
                                        "your lag spike or brain-freeze while answering, but NEVER output raw '*(API Error...)*' text.]"
                                    )
                                })
                                fb_v_res = await post_with_backoff(
                                    api_url,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    json_payload=fb_v_payload,
                                    timeout=(6, 18),
                                    max_retries=1,
                                    bot_name=client.user.display_name,
                                    glitch_tracker=turn_glitch_tracker
                                )
                                if fb_v_res is not None and fb_v_res.status_code == 200:
                                    raw_reply = fb_v_res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                                    reply = ERROR_REPLY_STRIP_REGEX.sub('', raw_reply or "").strip()
                                    if reply:
                                        res = fb_v_res
                                        break

                        if not reply and should_try_fallback:
                            for fb_model in rank_models_by_ewma_latency(FALLBACK_TEXT_MODELS, endpoint_label):
                                if fb_model == active_model:
                                    continue
                                print(f"[{client.user.display_name}] [TEXT FALLBACK] Trying '{fb_model}'...")
                                fb_payload = copy.deepcopy(payload)
                                fb_payload["model"] = fb_model
                                fb_payload["messages"] = copy.deepcopy(text_only_messages)
                                glitch_reason = "; ".join(turn_glitch_tracker[-2:]) if turn_glitch_tracker else "primary model timeout"
                                fb_payload["messages"].append({
                                    "role": "system",
                                    "content": (
                                        f"[System Glitch Context: Your primary neural model ({active_model}) just glitched ({glitch_reason}), "
                                        f"so you recovered on backup model ({fb_model}). You may briefly joke or grumble in-character about "
                                        "your lag spike or brain-freeze while answering, but NEVER output raw '*(API Error...)*' text.]"
                                    )
                                })
                                fb_res = await post_with_backoff(
                                    api_url,
                                    headers={"Authorization": f"Bearer {api_key}"},
                                    json_payload=fb_payload,
                                    timeout=(6, 18),
                                    max_retries=1,
                                    bot_name=client.user.display_name,
                                    glitch_tracker=turn_glitch_tracker
                                )
                                if fb_res is not None and fb_res.status_code == 200:
                                    raw_reply = fb_res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                                    reply = ERROR_REPLY_STRIP_REGEX.sub('', raw_reply or "").strip()
                                    if reply:
                                        res = fb_res
                                        break

                        if not reply:
                            is_error_reply = True
                            if res is not None and res.status_code != 200:
                                final_reply = f"*(API Error {res.status_code} - Provider busy. Try again in a moment.)*"
                            else:
                                final_reply = "*(System: DeepInfra dropped the text request after multiple retries.)*"
                        else:
                            bot_thought_text, reply = extract_and_strip_thought_tag(reply)
                            raw_voice_content, reply = extract_and_strip_voice_msg(reply)
                            if directed_at_other_bot:
                                raw_voice_content = None

                            learned_notifications = []
                            allow_turn_learning = (is_relevant_trigger or len(clean_content.split()) >= 4)
                            # Cap explicit [LEARN_USER:] extractions to at most 2 per turn so learning is detailed but never excessive
                            learn_user_matches = list(LEARN_USER_EXTRACT_REGEX.findall(reply))[:2] if allow_turn_learning else []
                            if allow_turn_learning:
                                implicit_self_facts = extract_implicit_user_facts_from_message(
                                    message.author.display_name,
                                    combined_recent_prompt,
                                    user_profiles=user_profiles,
                                    persona_prompt=system_prompt.get("content", ""),
                                    my_bot_name=client.user.display_name,
                                    is_speaker_bot=is_author_bot,
                                    bot_reply_text=reply
                                )
                                for imp_u, imp_f in implicit_self_facts:
                                    if not any(imp_f.lower() in existing_f.lower() or existing_f.lower() in imp_f.lower() for _, existing_f in learn_user_matches):
                                        learn_user_matches.append((imp_u, imp_f))

                            if learn_user_matches:
                                verify_task = asyncio.create_task(
                                    verify_and_resolve_user_facts(
                                        candidate_facts=learn_user_matches,
                                        speaker_name=message.author.display_name,
                                        recent_user_prompt=combined_recent_prompt,
                                        raw_history=raw_history,
                                        user_profiles=user_profiles,
                                        recent_users=recent_users,
                                        other_bot_names=other_bot_names_set,
                                        my_bot_name=client.user.display_name,
                                        api_key=api_key,
                                        text_model=text_model,
                                        api_url=api_url
                                    )
                                )
                                done_v, _ = await asyncio.wait({verify_task}, timeout=0.15)
                                if verify_task in done_v:
                                    try:
                                        verified_user_facts = verify_task.result()
                                        if verified_user_facts:
                                            def _save_user_facts_fast(uprof_data):
                                                for canonical_uname, clean_fact in verified_user_facts:
                                                    if (
                                                        canonical_uname
                                                        and clean_fact
                                                        and canonical_uname.lower() != client.user.display_name.lower()
                                                        and _extract_base_username(canonical_uname) != _extract_base_username(client.user.display_name)
                                                    ):
                                                        if canonical_uname not in uprof_data or not isinstance(uprof_data[canonical_uname], list):
                                                            uprof_data[canonical_uname] = []
                                                        if consolidate_and_append_fact(clean_fact, uprof_data[canonical_uname], max_items=35):
                                                            from subconscious_engine import format_fact_for_display
                                                            disp_fact = format_fact_for_display(clean_fact, include_badge=False)
                                                            note_txt = f"learned about {canonical_uname}: {disp_fact}" if disp_fact else f"learned about {canonical_uname}"
                                                            if note_txt not in learned_notifications:
                                                                learned_notifications.append(note_txt)
                                                return uprof_data
                                            update_json_file(user_profiles_file, {}, _save_user_facts_fast)
                                    except Exception as vf_err:
                                        print(f"[{client.user.display_name}] [WARN] Fast-path user-fact save error: {vf_err}")
                                else:
                                    async def _finish_bg_verify(vt=verify_task):
                                        try:
                                            verified_bg = await vt
                                            if verified_bg:
                                                def _save_user_facts_bg(uprof_data):
                                                    for canonical_uname, clean_fact in verified_bg:
                                                        if (
                                                            canonical_uname
                                                            and clean_fact
                                                            and canonical_uname.lower() != client.user.display_name.lower()
                                                            and _extract_base_username(canonical_uname) != _extract_base_username(client.user.display_name)
                                                        ):
                                                            if canonical_uname not in uprof_data or not isinstance(uprof_data[canonical_uname], list):
                                                                uprof_data[canonical_uname] = []
                                                            consolidate_and_append_fact(clean_fact, uprof_data[canonical_uname], max_items=35)
                                                    return uprof_data
                                                update_json_file(user_profiles_file, {}, _save_user_facts_bg)
                                        except Exception as bg_err:
                                            print(f"[{client.user.display_name}] [WARN] Background user-fact verification error: {bg_err}")
                                    asyncio.create_task(_finish_bg_verify())

                            allow_fact_learning = (allow_turn_learning and len(clean_content.split()) >= 3)
                            raw_gen_matches = LEARN_GENERAL_EXTRACT_REGEX.findall(reply)[:1] if allow_fact_learning else []
                            raw_srv_matches = LEARN_EXTRACT_REGEX.findall(reply)[:1] if allow_fact_learning else []
                            routed_srv_facts, routed_gen_facts = route_server_vs_general_facts(
                                raw_server_facts=raw_srv_matches,
                                raw_general_facts=raw_gen_matches,
                                recent_users=recent_users | set(user_profiles.keys()),
                                other_bot_names=other_bot_names_set,
                                my_bot_name=client.user.display_name
                            )

                            if routed_gen_facts:
                                def _save_general_facts(gen_data):
                                    if not isinstance(gen_data, list):
                                        gen_data = []
                                    for clean_gfact in routed_gen_facts:
                                        if consolidate_and_append_fact(clean_gfact, gen_data, max_items=80):
                                            learned_notifications.append(f"learned general fact: {clean_gfact}")
                                    return gen_data
                                update_json_file(general_memory_file, [], _save_general_facts)

                            if routed_srv_facts:
                                def _save_ltm_facts(ltm_data):
                                    if not isinstance(ltm_data, list):
                                        ltm_data = []
                                    for clean_fact in routed_srv_facts:
                                        if consolidate_and_append_fact(clean_fact, ltm_data, max_items=60):
                                            learned_notifications.append(f"learned server fact: {clean_fact}")
                                    return ltm_data
                                update_json_file(long_term_memory_file, [], _save_ltm_facts)

                            if ch_enable_image_gen and not directed_at_other_bot:
                                was_meme_intent = bool(is_explicit_meme_req)
                                meme_gen_match = MEME_GEN_REGEX.search(reply) or UNCLOSED_MEME_GEN_REGEX.search(reply)
                                if meme_gen_match:
                                    if is_explicit_img_req:
                                        # User asked for a clean image, not a captioned meme
                                        img_prompt, _, _ = parse_meme_gen_tag(meme_gen_match.group(1))
                                        meme_top_text = ""
                                        meme_bottom_text = ""
                                    else:
                                        was_meme_intent = True
                                        img_prompt, meme_top_text, meme_bottom_text = parse_meme_gen_tag(meme_gen_match.group(1))
                                else:
                                    img_gen_match = (
                                        IMAGE_GEN_REGEX.search(reply)
                                        or UNCLOSED_IMAGE_GEN_REGEX.search(reply)
                                        or FALLBACK_IMAGE_NOTE_REGEX.search(reply)
                                    )
                                    if img_gen_match:
                                        raw_img_tag = img_gen_match.group(1).strip()
                                        if "|" in raw_img_tag and not is_explicit_img_req:
                                            was_meme_intent = True
                                            img_prompt, meme_top_text, meme_bottom_text = parse_meme_gen_tag(raw_img_tag)
                                        else:
                                            img_prompt, _, _ = parse_meme_gen_tag(raw_img_tag)

                                other_bot_just_generated_on_msg = any(
                                    r.get("trigger_msg_id") == message.id and r.get("bot_id") != client.user.id
                                    for r in gen_records
                                )
                                is_topic_change_req = bool(TOPIC_CHANGE_REQUEST_REGEX.search(combined_recent_prompt or ""))
                                is_short_reaction_msg = bool(BARE_FILLER_UTTERANCE_REGEX.match(clean_content or ""))
                                bot_promised_visual_in_text = bool(BOT_IMAGE_PROMISE_REGEX.search(reply))

                                if img_prompt and not (is_explicit_img_req or is_explicit_meme_req or bot_promised_visual_in_text):
                                    claimed_unsolicited_img = False
                                    if not (other_bot_just_generated_on_msg or self_recently_generated or is_topic_change_req or is_short_reaction_msg):
                                        def _claim_unsolicited_img_slot(c):
                                            nonlocal claimed_unsolicited_img
                                            img_claims = c.get("_claimed_img_gen_msgs", {})
                                            if not isinstance(img_claims, dict):
                                                img_claims = {}
                                            existing_bot = img_claims.get(str(message.id))
                                            if existing_bot is None or existing_bot == client.user.id:
                                                img_claims[str(message.id)] = client.user.id
                                                if len(img_claims) > 50:
                                                    for old_k in list(img_claims.keys())[:-50]:
                                                        img_claims.pop(old_k, None)
                                                c["_claimed_img_gen_msgs"] = img_claims
                                                claimed_unsolicited_img = True
                                            return c
                                        update_json_file(shared_image_cache_file, {}, _claim_unsolicited_img_slot)

                                    if not claimed_unsolicited_img:
                                        print(f"[{client.user.display_name}] [INFO] Suppressing unsolicited image/meme (cooldown/concurrent-bot-claim/topic-pivot guard).")
                                        img_prompt = None
                                        meme_top_text = ""
                                        meme_bottom_text = ""
                                        was_meme_intent = False

                                if not img_prompt and is_explicit_meme_req:
                                    print(f"[{client.user.display_name}] [INFO] Synthesizing missing [MEME_GEN:] spec for explicit meme request...")
                                    was_meme_intent = True
                                    img_prompt, meme_top_text, meme_bottom_text = await synthesize_missing_meme_spec(
                                        reply, combined_recent_prompt, gen_records, client.user.display_name, persona_summary, api_key, text_model, api_url
                                    )
                                elif not img_prompt and (
                                    is_explicit_img_req
                                    or (
                                        not is_topic_change_req
                                        and not is_short_reaction_msg
                                        and bot_promised_visual_in_text
                                    )
                                ):
                                    print(f"[{client.user.display_name}] [INFO] Synthesizing missing [IMAGE_GEN:] prompt for explicit image/remix request...")
                                    img_prompt = await synthesize_missing_image_prompt(
                                        reply, combined_recent_prompt, gen_records, client.user.display_name, persona_summary, api_key, text_model, api_url, raw_history=raw_history, source_image_desc=remix_source_desc
                                    )

                                if img_prompt:
                                    img_prompt = await ensure_unique_image_prompt(
                                        img_prompt, gen_records, raw_history, client.user.display_name, persona_summary, combined_recent_prompt, api_key, text_model, api_url, reply_text=reply, source_image_desc=remix_source_desc
                                    )
                                    if was_meme_intent or meme_top_text or meme_bottom_text:
                                        meme_top_text, meme_bottom_text = await ensure_unique_meme_captions(
                                            img_prompt, meme_top_text, meme_bottom_text, reply, combined_recent_prompt,
                                            gen_records, client.user.display_name, persona_summary, api_key, text_model, api_url
                                        )

                            reply = LEARN_USER_STRIP_REGEX.sub('', reply).strip()
                            reply = LEARN_GENERAL_STRIP_REGEX.sub('', reply).strip()
                            reply = LEARN_STRIP_REGEX.sub('', reply).strip()
                            reply = INTERNAL_REACT_STRIP_REGEX.sub('', reply).strip()
                            reply = EMOJI_REACT_STRIP_REGEX.sub('', reply).strip()
                            reply = IMAGE_DESC_STRIP_REGEX.sub('', reply).strip()
                            reply = MEME_GEN_REGEX.sub('', reply).strip()
                            reply = UNCLOSED_MEME_GEN_REGEX.sub('', reply).strip()
                            reply = IMAGE_GEN_REGEX.sub('', reply).strip()
                            reply = UNCLOSED_IMAGE_GEN_REGEX.sub('', reply).strip()
                            reply = FALLBACK_IMAGE_NOTE_REGEX.sub('', reply).strip()

                            if ch_enable_gifs and not img_prompt and not directed_at_other_bot:
                                gif_match = GIF_EXTRACT_REGEX.search(reply) or UNCLOSED_GIF_EXTRACT_REGEX.search(reply)
                                if gif_match:
                                    candidate_gif_term = gif_match.group(1).strip()
                                    now_ts = time.time()
                                    gif_cooldown_ok = (now_ts - channel_last_gif_time.get(channel_id, 0)) >= 120.0
                                    if is_explicit_gif_req or (gif_cooldown_ok and random.random() < 0.35):
                                        gif_search_term = candidate_gif_term
                                        channel_last_gif_time[channel_id] = now_ts
                                    else:
                                        print(f"[{client.user.display_name}] [INFO] Suppressed unsolicited GIF ('{candidate_gif_term}') to keep GIF usage low.")
                            reply = GIF_STRIP_REGEX.sub('', reply).strip()
                            reply = clean_and_compact_reply(reply)

                            if not reply and raw_voice_content:
                                reply = derive_chat_text_from_voice(raw_voice_content)

                            if ch_enable_voice and not raw_voice_content and is_explicit_voice_req and reply:
                                raw_voice_content = reply

                            img_task = (
                                generate_image_attachment(
                                    img_prompt, image_gen_model, api_key, client.user.id,
                                    client.user.display_name, channel_id, message.id, shared_image_cache_file,
                                    meme_top_text=meme_top_text, meme_bottom_text=meme_bottom_text
                                )
                                if (ch_enable_image_gen and img_prompt)
                                else asyncio.sleep(0, result=(None, None))
                            )
                            voice_task = (
                                generate_voice_attachment(
                                    raw_voice_content, voice_model, tts_voice, api_key,
                                    client.user.display_name, extracted_voice_role, extracted_voice_style
                                )
                                if (ch_enable_voice and raw_voice_content)
                                else asyncio.sleep(0, result=(None, None, None))
                            )
                            gif_task = (
                                fetch_online_gif(gif_search_term, giphy_key)
                                if (ch_enable_gifs and gif_search_term and not img_prompt)
                                else asyncio.sleep(0, result="")
                            )

                            (image_file, img_err), (voice_file, voice_text, voice_err), fetched_gif = await asyncio.gather(
                                img_task, voice_task, gif_task
                            )

                            if fetched_gif and not image_file:
                                final_gif = fetched_gif
                            if img_err:
                                reply = f"{reply}\n{img_err}".strip()
                            if voice_err:
                                reply = f"{reply}\n{voice_err}".strip()

                            if not reply and not final_gif and not image_file and not voice_file:
                                if not is_explicit_trigger:
                                    print(f"[{client.user.display_name}] [INFO] Blank output after tag stripping on non-explicit turn; staying silent.")
                                    return
                                if learned_notifications:
                                    reply = "Got it, I'll keep that in mind!"
                                else:
                                    is_error_reply = True
                                    reply = "*(Brain empty. The API returned a blank response.)*"

                            if learned_notifications and not is_error_reply:
                                notes_str = " | ".join(f"{client.user.display_name} {n}" for n in learned_notifications)
                                reply = f"{reply}\n*({notes_str})*".strip()

                            final_reply = reply

                    except Exception as err:
                        import traceback
                        print(f"[{client.user.display_name}] [ERROR] Response error: {err}")
                        traceback.print_exc()
                        if channel_id in channel_followup_state:
                            channel_followup_state[channel_id]["active"] = False
                        sys_error = "*(System: I encountered a network timeout or connection error. Check the console.)*"

                if sys_error:
                    sent_err_msg = None
                    try:
                        sent_err_msg = await message.channel.send(sys_error)
                    except Exception:
                        pass
                    err_sum = extract_glitch_summary(sys_error) or "network timeout or connection error"
                    upsert_chat_memory_entry(
                        chat_memory_file,
                        channel_id,
                        {
                            "role": "system",
                            "content": (
                                f"[System Glitch Event: {client.user.display_name} experienced a glitch ({err_sum}) while trying to reply to {message.author.display_name}. "
                                f"Next time {client.user.display_name} or another bot speaks, they can joke or react in-character to this glitch, without repeating raw '*(System:...)*' text.]"
                            ),
                            "message_id": sent_err_msg.id if sent_err_msg else f"sys_glitch_{message.id}_{client.user.id}",
                            "author_id": client.user.id,
                            "author_name": client.user.display_name,
                            "is_glitch_event": True
                        },
                        max_history_length
                    )
                else:
                    files_to_send = []
                    is_meme_output = bool(meme_top_text.strip() or meme_bottom_text.strip())
                    media_view = None
                    if image_file:
                        fname = "generated_meme.png" if is_meme_output else "generated_image.png"
                        files_to_send.append(discord.File(fp=image_file, filename=fname))
                        if img_prompt:
                            media_view = create_generated_media_view(
                                img_prompt=img_prompt,
                                meme_top_text=meme_top_text,
                                meme_bottom_text=meme_bottom_text,
                                image_gen_model=image_gen_model,
                                api_key=api_key,
                                bot_id=client.user.id,
                                bot_name=client.user.display_name,
                                channel_id=channel_id,
                                trigger_msg_id=message.id,
                                shared_image_cache_file=shared_image_cache_file,
                            )
                    if voice_file:
                        files_to_send.append(discord.File(fp=voice_file, filename="voice_message.mp3"))

                    sent_msg = None
                    send_kwargs = {" view": media_view} if media_view is not None else {}
                    try:
                        if final_gif:
                            if media_view is not None:
                                sent_msg = await message.channel.send(
                                    content=f"{final_reply}\n{final_gif}" if final_reply else final_gif,
                                    files=files_to_send,
                                    view=media_view,
                                )
                            else:
                                sent_msg = await message.channel.send(
                                    content=f"{final_reply}\n{final_gif}" if final_reply else final_gif,
                                    files=files_to_send,
                                )
                        else:
                            chunks = [final_reply[i:i + 1900] for i in range(0, len(final_reply), 1900)] if final_reply else []
                            if chunks:
                                for i, chunk in enumerate(chunks):
                                    if i == len(chunks) - 1:
                                        if media_view is not None:
                                            sent_msg = await message.channel.send(chunk.strip(), files=files_to_send, view=media_view)
                                        else:
                                            sent_msg = await message.channel.send(chunk.strip(), files=files_to_send)
                                    elif chunk.strip():
                                        sent_msg = await message.channel.send(chunk.strip())
                            elif files_to_send:
                                if media_view is not None:
                                    sent_msg = await message.channel.send(files=files_to_send, view=media_view)
                                else:
                                    sent_msg = await message.channel.send(files=files_to_send)
                    except discord.errors.HTTPException as e:
                        print(f"[{client.user.display_name}] [ERROR] Discord upload failed: {e}")
                        image_file = None
                        voice_file = None
                        sent_msg = await message.channel.send(
                            f"{final_reply}\n*(System: Failed to upload media file. It may be corrupted or too large.)*"
                        )

                    max_hist_msg_id = max(
                        (int(m["message_id"]) for m in raw_history if isinstance(m.get("message_id"), int)),
                        default=message.id
                    )
                    channel_last_replied_msg_id[channel_id] = max(
                        channel_last_replied_msg_id.get(channel_id, 0),
                        message.id,
                        max_hist_msg_id
                    )

                    if sent_msg and img_prompt and image_file:
                        caption_desc = f" with meme captions '{meme_top_text} / {meme_bottom_text}'" if is_meme_output else ""
                        def _precache_bot_img(c):
                            c[sent_msg.id] = f"[Attached Media Description: Generated {'meme' if is_meme_output else 'image'} by {client.user.display_name} showing: {img_prompt}{caption_desc}]"
                            return c
                        update_json_file(shared_image_cache_file, {}, _precache_bot_img)

                    if not is_error_reply:
                        entry_msg_id = sent_msg.id if sent_msg else f"resp_{message.id}_{client.user.id}"
                        clean_mem_reply = ERROR_REPLY_STRIP_REGEX.sub('', final_reply or '').strip()
                        clean_mem_reply = re.sub(r'\*\([^)]*learned (?:server fact|general fact|about).*?\)\*', '', clean_mem_reply, flags=re.IGNORECASE).strip()

                        if not is_author_bot:
                            prev_turns_used = (
                                channel_followup_state.get(channel_id, {}).get("turns_used", 0)
                                if (is_contextual_followup and not is_explicit_trigger)
                                else 0
                            )
                            new_turns_used = 0 if is_explicit_trigger else (prev_turns_used + 1)

                            effective_max_followup = (max_followup_turns + 1) if (max_followup_turns > 0 and speaker_affinity >= 45) else max_followup_turns
                            if effective_max_followup <= 0 or new_turns_used >= effective_max_followup:
                                channel_followup_state[channel_id] = {
                                    "timestamp": time.time(),
                                    "turns_used": new_turns_used,
                                    "active": False,
                                    "bot_msg_id": entry_msg_id,
                                    "user_id": message.author.id
                                }
                            else:
                                channel_followup_state[channel_id] = {
                                    "timestamp": time.time(),
                                    "turns_used": new_turns_used,
                                    "active": True,
                                    "bot_msg_id": entry_msg_id,
                                    "user_id": message.author.id
                                }
                                def _claim_followup_owner_after_reply(c):
                                    f_map = c.get("_active_followup_bot", {})
                                    if not isinstance(f_map, dict):
                                        f_map = {}
                                    msg_owners = c.get("_msg_addressed_bots", {})
                                    cur_list = msg_owners.get(str(message.id), []) if isinstance(msg_owners, dict) else []
                                    if isinstance(cur_list, list) and len(cur_list) > 1 and client.user.id in cur_list:
                                        f_map[channel_id] = cur_list
                                    else:
                                        f_map[channel_id] = client.user.id
                                    c["_active_followup_bot"] = f_map
                                    return c
                                update_json_file(shared_image_cache_file, {}, _claim_followup_owner_after_reply)

                        memory_text = clean_mem_reply if clean_mem_reply else "[Sent media]"
                        if img_prompt and image_file:
                            caption_desc = f" with meme captions '{meme_top_text} / {meme_bottom_text}'" if is_meme_output else ""
                            memory_text = f"{memory_text}\n[Attached Media Description: Generated {'meme' if is_meme_output else 'image'} by {client.user.display_name} showing: {img_prompt}{caption_desc}]".strip()

                        memory_entry = {
                            "role": "assistant",
                            "content": memory_text,
                            "author_id": client.user.id,
                            "author_name": client.user.display_name,
                            "message_id": entry_msg_id,
                            "author_is_bot": True
                        }
                        if img_prompt and image_file:
                            if is_meme_output:
                                memory_entry["meme_spec"] = f"{img_prompt} | {meme_top_text} | {meme_bottom_text}"
                            else:
                                memory_entry["img_prompt"] = img_prompt
                        if gif_search_term and final_gif:
                            memory_entry["gif_query"] = gif_search_term
                        if voice_text and voice_file:
                            memory_entry["voice_text"] = voice_text

                        upsert_chat_memory_entry(chat_memory_file, channel_id, memory_entry, max_history_length)

                        media_glitches = [
                            extract_glitch_summary(e) for e in (img_err, voice_err) if e
                        ]
                        media_glitches = [g for g in media_glitches if g]
                        if media_glitches:
                            upsert_chat_memory_entry(
                                chat_memory_file,
                                channel_id,
                                {
                                    "role": "system",
                                    "content": (
                                        f"[System Glitch Event: {client.user.display_name}'s media generator glitched ({'; '.join(media_glitches)}) "
                                        f"on the reply to {message.author.display_name}. {client.user.display_name} or other bots can react to this in-character if asked.]"
                                    ),
                                    "message_id": f"media_glitch_{entry_msg_id}",
                                    "author_id": client.user.id,
                                    "author_name": client.user.display_name,
                                    "is_glitch_event": True
                                },
                                max_history_length
                            )
                    else:
                        err_summary = extract_glitch_summary(final_reply) or "API timeout / empty response"
                        upsert_chat_memory_entry(
                            chat_memory_file,
                            channel_id,
                            {
                                "role": "system",
                                "content": (
                                    f"[System Glitch Event: {client.user.display_name} glitched out ({err_summary}) while trying to reply to {message.author.display_name}. "
                                    f"Next time {client.user.display_name} or other bots speak, they may joke, grumble, or roast this malfunction in-character, "
                                    "but NEVER output raw '*(API Error...)*' strings.]"
                                ),
                                "message_id": sent_msg.id if sent_msg else f"err_glitch_{message.id}_{client.user.id}",
                                "author_id": client.user.id,
                                "author_name": client.user.display_name,
                                "is_glitch_event": True
                            },
                            max_history_length
                        )

                    # Signal Wave 1 completion so any Wave 2 Co-Star bots wake up immediately!
                    mark_turn_wave_complete(shared_image_cache_file, message.id, client.user.id)

                    # Phase 1B — Non-Blocking Unified Subconscious Cognitive Engine (0ms typing latency impact!)
                    if not is_error_reply and clean_mem_reply:
                        async def _bg_subconscious_wrapper():
                            try:
                                await asyncio.sleep(0.10)
                                await run_subconscious_cognitive_turn(
                                    channel_id=channel_id,
                                    trigger_message_id=message.id,
                                    speaker_name=message.author.display_name,
                                    user_message=combined_recent_prompt,
                                    bot_reply_text=clean_mem_reply,
                                    bot_thought_text=bot_thought_text,
                                    my_bot_name=client.user.display_name,
                                    is_speaker_bot=is_author_bot,
                                    raw_history=raw_history,
                                    user_profiles_file=user_profiles_file,
                                    long_term_memory_file=long_term_memory_file,
                                    general_knowledge_file=general_memory_file,
                                    shared_cache_file=shared_image_cache_file,
                                    other_bot_names=other_bot_display_list,
                                    api_key=api_key,
                                    api_url=api_url,
                                )
                                record_ipc_heartbeat()
                                await sync_bot_custom_status(channel_id)
                            except Exception as sub_e:
                                print(f"[{client.user.display_name}] [SUBCONSCIOUS WARN] {sub_e}")

                        asyncio.create_task(_bg_subconscious_wrapper())

                    if react_task:
                        try:
                            reacted_emoji = await react_task
                        except Exception:
                            pass

                    if reacted_emoji:
                        update_message_reaction_ledger(
                            chat_memory_file=chat_memory_file,
                            user_profiles_file=user_profiles_file,
                            channel_id=channel_id,
                            message_id=message.id,
                            emoji_str=str(reacted_emoji),
                            reactor_name=client.user.display_name,
                            is_add=True,
                            reactor_is_bot=True,
                            my_bot_name=client.user.display_name,
                        )
        elif react_task:
            # Even when the bot did not send a text reply, record its silent emoji reaction in the In-Place Reaction Ledger!
            try:
                reacted_emoji = await react_task
                if reacted_emoji:
                    update_message_reaction_ledger(
                        chat_memory_file=chat_memory_file,
                        user_profiles_file=user_profiles_file,
                        channel_id=channel_id,
                        message_id=message.id,
                        emoji_str=str(reacted_emoji),
                        reactor_name=client.user.display_name,
                        is_add=True,
                        reactor_is_bot=True,
                        my_bot_name=client.user.display_name,
                    )
            except Exception:
                pass

    @client.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
        if not client.user or payload.user_id == client.user.id:
            return
        guild = client.get_guild(payload.guild_id) if getattr(payload, "guild_id", None) else None
        member = payload.member or (guild.get_member(payload.user_id) if guild else None)
        if not member:
            return
        emoji_str = str(payload.emoji)
        update_message_reaction_ledger(
            chat_memory_file=chat_memory_file,
            user_profiles_file=user_profiles_file,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            emoji_str=emoji_str,
            reactor_name=member.display_name,
            is_add=True,
            reactor_is_bot=bool(getattr(member, "bot", False)),
            my_bot_name=client.user.display_name,
        )

    @client.event
    async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent):
        if not client.user or payload.user_id == client.user.id:
            return
        guild = client.get_guild(payload.guild_id) if getattr(payload, "guild_id", None) else None
        member = guild.get_member(payload.user_id) if guild else None
        if not member:
            return
        emoji_str = str(payload.emoji)
        update_message_reaction_ledger(
            chat_memory_file=chat_memory_file,
            user_profiles_file=user_profiles_file,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            emoji_str=emoji_str,
            reactor_name=member.display_name,
            is_add=False,
            reactor_is_bot=bool(getattr(member, "bot", False)),
            my_bot_name=client.user.display_name,
        )

    @client.event
    async def on_member_join(member: discord.Member):
        if getattr(member, "bot", False) or not getattr(member, "guild", None):
            return
        conf_all = load_json_file(channel_config_file, {})
        g_key = f"guild_welcome_{member.guild.id}"
        g_conf = conf_all.get(g_key, {})
        if not isinstance(g_conf, dict) or not g_conf.get("enabled", False):
            return
        target_ch_id = g_conf.get("channel_id")
        target_ch = client.get_channel(int(target_ch_id)) if target_ch_id else member.guild.system_channel
        if target_ch:
            await trigger_welcome_greeting(member, target_ch, is_test=False)

    client.run(token)
