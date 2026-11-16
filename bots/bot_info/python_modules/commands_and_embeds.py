import re
import time
import asyncio
import discord

from core_system import load_json_file, update_json_file
from bot_discovery import get_other_active_bots, _term_boundary_pattern
from memory_and_context import (
    find_canonical_username, consolidate_and_append_fact,
    _extract_base_username, _extract_saved_rapport_state_full,
    _get_disposition_tier_label, _derive_relationship_archetype,
    RAPPORT_ENTRY_REGEX
)
from subconscious_engine import (
    sanitize_or_reject_sensitive_pii,
    pack_salience_fact,
    unpack_salience_fact,
    format_fact_for_display,
)

# ==============================================================================
# COMMAND UX #1: DAMERAU-LEVENSHTEIN (<= 2) FUZZY TYPO FORGIVENESS
# ==============================================================================
CANONICAL_BANG_COMMANDS = (
    "!whoami",
    "!hub",
    "!relationships",
    "!socialgraph",
    "!bonds",
    "!standing",
    "!chapters",
    "!episodes",
    "!recap",
    "!storysofar",
    "!forgetme",
    "!clearmemory",
    "!help",
)

MULTI_BOT_BROADCAST_REGEX = re.compile(
    r"(?:^|\s)!(?:bots|allbots)\s+(status|whoami|relationships|socialgraph|bonds|help)\b",
    re.IGNORECASE,
)


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Optimal String Alignment / Damerau-Levenshtein edit distance (insertions,
    deletions, substitutions, and adjacent transpositions) in O(N*M) for command tokens.
    """
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 3:
        return abs(len1 - len2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[len1][len2]


def normalize_command_typos(raw_text: str) -> str:
    """
    Scans any `!<word>` command token in `raw_text` and auto-corrects typos within
    Damerau-Levenshtein distance <= 2 (or <= 1 for short commands like `!help`).
    Example: `!whoam` -> `!whoami`, `!relatioships` -> `!relationships`.
    """
    if not raw_text or "!" not in raw_text:
        return raw_text

    def _replace_bang(match: re.Match) -> str:
        token = match.group(0).lower()
        if token in CANONICAL_BANG_COMMANDS or token in ("!bots", "!allbots"):
            return token
        best_cmd = token
        best_dist = 99
        max_allowed = 1 if len(token) <= 6 else 2
        for canon in CANONICAL_BANG_COMMANDS:
            dist = damerau_levenshtein_distance(token, canon)
            if dist <= max_allowed and dist < best_dist:
                best_dist = dist
                best_cmd = canon
        return best_cmd

    return re.sub(r"![a-zA-Z]{4,16}\b", _replace_bang, raw_text)


def is_command_addressed_to_bot(
    message: discord.Message,
    client: discord.Client,
    keywords: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Command UX #1 & #5:
    Allows triggering commands via:
    1. Explicit `@BotName` mention
    2. Direct Discord reply to one of this bot's messages
    3. Bot keyword/name prefix (`steward !whoami`, `kitty !chapters`)
    4. Multi-bot broadcast (`!bots status`, `!bots whoami`, `!bots relationships`)
    Returns `(is_addressed, normalized_text_lower)`.
    """
    raw_content = message.content or ""
    norm_content = normalize_command_typos(raw_content)
    content_lower = norm_content.lower().strip()

    # Support plain-text `/hub` or `!hub` in chat (in case Discord hasn't refreshed the client's slash popup yet)
    if re.search(r"(?:^|\s)[/!]hub\b", content_lower):
        content_lower = re.sub(r"(?:^|\s)[/!]hub\b", " !whoami", content_lower).strip()
        if client.user and client.user.mentioned_in(message):
            return True, content_lower

    # Check multi-bot broadcast commands (`!bots status`, `!bots whoami`)
    bcast_m = MULTI_BOT_BROADCAST_REGEX.search(content_lower)
    if bcast_m:
        sub = bcast_m.group(1).lower()
        mapped = f"admin status" if sub == "status" else f"!{sub}"
        return True, f"{content_lower} {mapped}"

    if client.user and client.user.mentioned_in(message):
        return True, content_lower

    # Check direct reply to this bot when message contains a `!` or `admin ` command
    has_cmd_token = bool(
        any(cmd in content_lower for cmd in CANONICAL_BANG_COMMANDS)
        or re.search(r"\badmin\s+(?:status|enable|disable|welcome|reactonly|chances|random|temp|features|memory|learn|learngeneral|learnuser|forget|webui|dashboard|help)\b", content_lower)
    )
    if not has_cmd_token:
        return False, content_lower

    if message.reference and getattr(message.reference, "resolved", None):
        ref_author = getattr(message.reference.resolved, "author", None)
        if ref_author and client.user and getattr(ref_author, "id", None) == client.user.id:
            return True, content_lower

    # Check bot display_name or keyword alias in the message
    bot_terms = set()
    if client.user and getattr(client.user, "display_name", None):
        disp_low = client.user.display_name.strip().lower()
        bot_terms.add(disp_low)
        base_low = re.sub(r"\s+v?\d+(?:\.\d+)*$", "", disp_low).strip()
        if base_low:
            bot_terms.add(base_low)
    for kw in (keywords or []):
        if kw and kw.strip():
            bot_terms.add(kw.strip().lower())

    for term in sorted(bot_terms, key=len, reverse=True):
        if len(term) >= 2 and re.search(rf"\b{_term_boundary_pattern(term)}", content_lower):
            return True, content_lower

    return False, content_lower


# ==============================================================================
# EMBED BUILDERS (WITH 1-100 SALIENCE BADGES & CORE/BOUNDARY PINNING)
# ==============================================================================
def build_affinity_progress_bar(score: int, width: int = 10) -> str:
    """
    Renders a visual progress bar (e.g. `[████████░░] +65/100`) for Affinity scores
    spanning -100 to +100.
    """
    clamped = max(-100, min(100, int(score)))
    normalized = (clamped + 100) / 200.0
    filled = int(round(normalized * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    sign_str = f"+{clamped}" if clamped > 0 else str(clamped)
    return f"`[{bar}] {sign_str}/100`"


def get_disposition_embed_color(score: int) -> int:
    """Returns a Discord Embed hex color matching the 5 Disposition Tiers."""
    if score >= 45:
        return 0x2ECC71  # Emerald Green (Trusted Inner Circle / Bonded Companion)
    if score >= 15:
        return 0x3498DB  # Sky Blue (Thawing & Reluctantly Amused / Friendly Ally)
    if score > -15:
        return 0x95A5A6  # Neutral Slate (Aloof & Guarded / Neutral Acquaintance)
    if score > -45:
        return 0xE67E22  # Amber Orange (Prickly & Skeptical / Comedic Foil)
    return 0xE74C3C      # Crimson Red (Cold & Cutting / Arch-Nemesis)


def format_relative_last_seen(last_seen_ts: float | None, now_ts: float | None = None) -> str:
    """Formats a human-readable relative timestamp for Embed footers/fields."""
    if not isinstance(last_seen_ts, (int, float)) or last_seen_ts <= 0:
        return "Active now / Not tracked yet"
    now_val = now_ts if now_ts is not None else time.time()
    delta = max(0.0, now_val - float(last_seen_ts))
    if delta < 90:
        return "Just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{delta / 3600.0:.1f}h ago"
    return f"{delta / 86400.0:.1f}d ago"


def build_whoami_embed(
    bot_display_name: str,
    target_canon: str,
    facts: list[str],
    is_target_bot: bool,
    persona_prompt: str,
    last_seen_ts: float | None = None,
) -> discord.Embed:
    """
    Builds a color-coded Discord Embed card for `@BotName !whoami [target]`,
    highlighting `[Core • S:XX]` and `[Boundary • S:XX]` facts at the top and
    formatting `[S:XX|CAT|DATE|R:N]` prefixes cleanly via `format_fact_for_display`.
    """
    sc, tr, cat, _ = _extract_saved_rapport_state_full(
        facts, persona_prompt=persona_prompt, target_name=target_canon
    )
    tier_lbl = _get_disposition_tier_label(sc)
    arch_lbl = _derive_relationship_archetype(
        sc, tr, is_fellow_bot=is_target_bot, had_past_friction=(sc < 0)
    )
    embed_color = get_disposition_embed_color(sc)
    bar_str = build_affinity_progress_bar(sc)
    entity_badge = "🤖 Fellow Bot Co-Star" if is_target_bot else "👤 Human Regular"

    embed = discord.Embed(
        title=f"🗂️ {bot_display_name}'s Dossier: {target_canon}",
        description=f"**{entity_badge}** — *{arch_lbl}*",
        color=embed_color,
    )
    embed.add_field(
        name="Disposition & Affinity",
        value=f"**{tier_lbl}**\n{bar_str} • Trend: `{tr}`",
        inline=False,
    )
    if cat:
        embed.add_field(
            name="Recent Relationship Catalyst",
            value=f"*{cat}*",
            inline=False,
        )
    if last_seen_ts:
        embed.add_field(
            name="Last Seen",
            value=format_relative_last_seen(last_seen_ts),
            inline=True,
        )

    non_rapport_facts = [
        str(f).strip()
        for f in (facts or [])
        if str(f).strip() and not RAPPORT_ENTRY_REGEX.match(str(f).strip())
    ]
    if non_rapport_facts:
        # Sort so Core & Boundary (S >= 90) facts appear first, followed by highest salience & recency
        pinned_core = []
        general_facts = []
        for raw_f in non_rapport_facts:
            info = unpack_salience_fact(raw_f)
            formatted_line = f"• {format_fact_for_display(raw_f, include_badge=True)}"
            if info["salience"] >= 90 or info["category"] in ("CORE", "BOUNDARY"):
                pinned_core.append(formatted_line)
            else:
                general_facts.append(formatted_line)

        ordered_lines = pinned_core + general_facts[-14:]
        facts_lines = "\n".join(ordered_lines[-15:])
        if len(facts_lines) > 1000:
            facts_lines = facts_lines[:995] + "..."
        embed.add_field(
            name=f"Logged Dossier Facts ({len(non_rapport_facts)})",
            value=facts_lines,
            inline=False,
        )
    else:
        embed.add_field(
            name="Logged Dossier Facts",
            value="*No additional personal facts logged yet.*",
            inline=False,
        )
    return embed


def build_social_graph_embed(
    bot_display_name: str,
    user_profiles: dict,
    other_bot_names: set[str],
    persona_prompt: str,
) -> discord.Embed:
    """
    Builds a complete Server Social Graph Embed for `@BotName !relationships` / `!socialgraph` / `!bonds`.
    """
    entries = []
    seen_bases = {_extract_base_username(bot_display_name)}

    all_candidates = list(user_profiles.keys())
    for ob_n in other_bot_names:
        if ob_n and ob_n not in all_candidates:
            all_candidates.append(ob_n)

    for uname in all_candidates:
        if not isinstance(uname, str) or uname.startswith("_"):
            continue
        b_name = _extract_base_username(uname)
        if not b_name or b_name in seen_bases:
            continue
        seen_bases.add(b_name)
        facts = user_profiles.get(uname, [])
        if not isinstance(facts, list):
            facts = []
        is_bot = any(_extract_base_username(uname) == _extract_base_username(bn) for bn in other_bot_names)
        sc, tr, cat, _ = _extract_saved_rapport_state_full(
            facts, persona_prompt=persona_prompt, target_name=uname
        )
        if sc == 0 and not facts and not is_bot:
            continue
        arch = _derive_relationship_archetype(sc, tr, is_fellow_bot=is_bot, had_past_friction=(sc < 0))
        entries.append((sc, uname, is_bot, tr, cat, arch))

    entries.sort(key=lambda x: x[0], reverse=True)

    embed = discord.Embed(
        title=f"🌐 {bot_display_name}'s Server Social Graph & Standings",
        description="Live relationship tiers, archetypes, and catalysts across humans & fellow bots.",
        color=0x5865F2,
    )

    if not entries:
        embed.add_field(
            name="No Relationships Logged Yet",
            value="*Chat with me in the server to build rapport (or rivalry)!*",
            inline=False,
        )
        return embed

    inner_circle = [e for e in entries if e[0] >= 45]
    warming = [e for e in entries if 15 <= e[0] < 45]
    neutral = [e for e in entries if -14 <= e[0] < 15]
    rivals = [e for e in entries if e[0] <= -15]

    def _fmt_group(group_list: list) -> str:
        lines = []
        for sc, uname, is_bot, tr, cat, arch in group_list[:6]:
            icon = "🤖" if is_bot else "👤"
            bar = build_affinity_progress_bar(sc, width=8)
            cat_snip = f" — *{cat[:65]}*" if cat else ""
            lines.append(f"{icon} **{uname}** ({bar} • *{arch}*){cat_snip}")
        return "\n".join(lines) if lines else "*None*"

    if inner_circle:
        embed.add_field(name="👑 Trusted Inner Circle & Bonded Companions", value=_fmt_group(inner_circle), inline=False)
    if warming:
        embed.add_field(name="🤝 Warming Friends & Friendly Allies", value=_fmt_group(warming), inline=False)
    if neutral:
        embed.add_field(name="🛡️ Aloof / Neutral Acquaintances", value=_fmt_group(neutral), inline=False)
    if rivals:
        embed.add_field(name="⚔️ Prickly Rivals, Comedic Foils & Nemeses", value=_fmt_group(rivals), inline=False)

    return embed


def get_merged_episodic_chapters(
    shared_image_cache_file: str,
    chat_memory_file: str,
    channel_id: int,
) -> list:
    """
    Retrieves rolling Episodic Session Chapters from both `shared_image_cache.json`
    (`_episodic_chapters[str(channel_id)]`, populated by the Subconscious 30B Engine)
    and legacy `chat_memory.json` (`chapters_<channel_id>`).
    """
    merged = []
    seen_summaries = set()
    if shared_image_cache_file:
        scache = load_json_file(shared_image_cache_file, {})
        ep_map = scache.get("_episodic_chapters", {})
        if isinstance(ep_map, dict):
            ch_list = ep_map.get(str(channel_id)) or ep_map.get(channel_id) or []
            if isinstance(ch_list, list):
                for item in ch_list:
                    s_txt = str(item.get("summary", item) if isinstance(item, dict) else item).strip()
                    if s_txt and s_txt.lower() not in seen_summaries:
                        seen_summaries.add(s_txt.lower())
                        merged.append(item)
    if chat_memory_file:
        cmem = load_json_file(chat_memory_file, {})
        legacy_list = cmem.get(f"chapters_{channel_id}", [])
        if isinstance(legacy_list, list):
            for item in legacy_list:
                s_txt = str(item.get("summary", item) if isinstance(item, dict) else item).strip()
                if s_txt and s_txt.lower() not in seen_summaries:
                    seen_summaries.add(s_txt.lower())
                    merged.append(item)
    return merged[-12:]


def build_episodic_chapters_embed(
    bot_display_name: str,
    channel_name: str,
    chapters_list: list,
) -> discord.Embed:
    """
    Builds a Discord Embed displaying rolling Episodic Session Chapters for `@BotName !chapters` / `!recap`.
    """
    embed = discord.Embed(
        title=f"📖 Episodic Session Chapters — #{channel_name}",
        description=f"Multi-session storylines and notable room events remembered by **{bot_display_name}**.",
        color=0x9B59B6,
    )
    if not chapters_list:
        embed.add_field(
            name="No Archived Chapters Yet",
            value="*Chapters are automatically synthesized by the Subconscious Engine as conversation sessions unfold.*",
            inline=False,
        )
        return embed

    for idx, ch_item in enumerate(chapters_list[-8:], start=1):
        if isinstance(ch_item, dict):
            title_lbl = ch_item.get("title") or ch_item.get("timestamp_label") or f"Chapter {idx}"
            dt_str = ch_item.get("date", "")
            if dt_str and dt_str not in title_lbl:
                title_lbl = f"{title_lbl} ({dt_str})"
            summary_txt = ch_item.get("summary") or str(ch_item)
        else:
            title_lbl = f"Episode {idx}"
            summary_txt = str(ch_item)
        embed.add_field(
            name=f"🔖 {title_lbl}",
            value=summary_txt[:450],
            inline=False,
        )
    return embed


# ==============================================================================
# CROWN JEWEL #4 & COMMAND UX #2-#4: INTERACTIVE DISCORD UI VIEWS & MODALS
# (WITH STRICT PER-USER OWNERSHIP ISOLATION & ZERO-PII FIREWALL)
# ==============================================================================
_HAS_DISCORD_UI = hasattr(discord, "ui") and hasattr(getattr(discord, "ui", None), "View")

if _HAS_DISCORD_UI:

    class AddFactModal(discord.ui.Modal, title="➕ Add Personal Fact or Boundary"):
        fact_input = discord.ui.TextInput(
            label="Fact or Nickname Boundary (Your Profile Only)",
            placeholder="e.g., Prefers not to be called sweetie | Studying Computer Science",
            min_length=4,
            max_length=160,
            required=True,
        )

        def __init__(self, user_profiles_file: str, caller_display_name: str):
            super().__init__()
            self.user_profiles_file = user_profiles_file
            self.caller_display_name = caller_display_name

        async def on_submit(self, interaction: discord.Interaction):
            # STRICT OWNERSHIP INVARIANT: Always bind to interaction.user.display_name!
            caller_name = getattr(interaction.user, "display_name", None) or self.caller_display_name
            raw_text = str(self.fact_input.value or "").strip()
            safe_fact = sanitize_or_reject_sensitive_pii(raw_text)
            if not safe_fact:
                await interaction.response.send_message(
                    "🛡️ **Zero-PII Security Firewall**: That entry was blocked because it contained sensitive personal data "
                    "(such as a last name/surname, street address, phone number, email, financial number, or key). "
                    "*First names, preferences, projects, and boundaries are welcome!*",
                    ephemeral=True,
                )
                return

            packed = pack_salience_fact(safe_fact)
            saved_holder = {"added": False, "canon": caller_name}

            def _mutate(uprof: dict) -> dict:
                canon = find_canonical_username(caller_name, uprof, {caller_name})
                saved_holder["canon"] = canon
                uprof.setdefault(canon, [])
                saved_holder["added"] = consolidate_and_append_fact(packed, uprof[canon], max_items=35)
                return uprof

            update_json_file(self.user_profiles_file, {}, _mutate)
            disp_clean = format_fact_for_display(packed, include_badge=True)
            await interaction.response.send_message(
                f"✅ Saved to **{saved_holder['canon']}**'s personal dossier: `{disp_clean}`",
                ephemeral=True,
            )


    class EphemeralFactDeleteSelect(discord.ui.Select):
        def __init__(self, user_profiles_file: str, caller_name: str, facts_list: list[str]):
            self.user_profiles_file = user_profiles_file
            self.caller_name = caller_name
            options = []
            non_rapport = [
                str(f).strip()
                for f in (facts_list or [])
                if str(f).strip() and not RAPPORT_ENTRY_REGEX.match(str(f).strip())
            ]
            for idx, raw_f in enumerate(non_rapport[-20:]):
                clean_lbl = format_fact_for_display(raw_f, include_badge=True)
                options.append(
                    discord.SelectOption(
                        label=clean_lbl[:95],
                        value= unpack_salience_fact(raw_f)["text"][:95],
                        description="Click to delete this fact from your profile",
                    )
                )
            if not options:
                options.append(
                    discord.SelectOption(
                        label="No personal facts logged yet",
                        value="__NONE__",
                        description="Use the Add Fact button below to add one!",
                    )
                )
            super().__init__(
                placeholder="🗑️ Select one of YOUR facts to delete...",
                min_values=1,
                max_values=1,
                options=options[:25],
            )

        async def callback(self, interaction: discord.Interaction):
            caller = getattr(interaction.user, "display_name", None) or self.caller_name
            selected_val = self.values[0]
            if selected_val == "__NONE__":
                await interaction.response.send_message("ℹ️ You don't have any logged facts to delete yet.", ephemeral=True)
                return

            removed_holder = {"count": 0}

            def _del_fact(uprof: dict) -> dict:
                canon = find_canonical_username(caller, uprof, {caller})
                existing = uprof.get(canon, [])
                if not isinstance(existing, list):
                    return uprof
                kept = []
                for f in existing:
                    plain_txt = unpack_salience_fact(str(f))["text"]
                    if selected_val.lower() in plain_txt.lower() and not RAPPORT_ENTRY_REGEX.match(str(f).strip()):
                        removed_holder["count"] += 1
                    else:
                        kept.append(f)
                uprof[canon] = kept
                return uprof

            update_json_file(self.user_profiles_file, {}, _del_fact)
            await interaction.response.send_message(
                f"🗑️ Removed `{selected_val}` from **{caller}**'s profile.",
                ephemeral=True,
            )


    class EphemeralFactManageView(discord.ui.View):
        """
        Command UX #3 — Strict Caller-Isolated Personal Fact Manager:
        Even if opened while viewing someone else's public `!whoami` card, this view
        strictly loads and mutates ONLY `interaction.user.display_name`'s own profile!
        """
        def __init__(self, user_profiles_file: str, caller_name: str):
            super().__init__(timeout=180)
            self.user_profiles_file = user_profiles_file
            self.caller_name = caller_name
            uprof = load_json_file(user_profiles_file, {})
            canon = find_canonical_username(caller_name, uprof, {caller_name})
            caller_facts = uprof.get(canon, []) if isinstance(uprof.get(canon), list) else []
            self.add_item(EphemeralFactDeleteSelect(user_profiles_file, caller_name, caller_facts))

        @discord.ui.button(label="➕ Add Fact / Boundary", style=discord.ButtonStyle.success, row=1)
        async def add_fact_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            caller = getattr(interaction.user, "display_name", None) or self.caller_name
            await interaction.response.send_modal(AddFactModal(self.user_profiles_file, caller))

        @discord.ui.button(label="🧹 Wipe All My Facts", style=discord.ButtonStyle.danger, row=1)
        async def wipe_all_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            caller = getattr(interaction.user, "display_name", None) or self.caller_name

            def _wipe(uprof: dict) -> dict:
                canon = find_canonical_username(caller, uprof, {caller})
                existing = uprof.get(canon, [])
                if isinstance(existing, list):
                    # Preserve Rapport standing while wiping personal facts
                    uprof[canon] = [f for f in existing if RAPPORT_ENTRY_REGEX.match(str(f).strip())]
                return uprof

            update_json_file(self.user_profiles_file, {}, _wipe)
            await interaction.response.send_message(
                f"🧹 Cleared all personal facts from **{caller}**'s profile!",
                ephemeral=True,
            )


    class TuneChannelModal(discord.ui.Modal, title="⚙️ Tune Channel Parameters"):
        temp_input = discord.ui.TextInput(label="Temperature (0.1 - 1.5)", placeholder="0.78", required=False, max_length=6)
        resp_input = discord.ui.TextInput(label="Spontaneous Respond Chance (0.0 - 1.0)", placeholder="0.05", required=False, max_length=6)
        emoji_input = discord.ui.TextInput(label="Emoji Reaction Chance (0.0 - 1.0)", placeholder="0.45", required=False, max_length=6)
        rand_input = discord.ui.TextInput(label="Random Post Interval (minutes, 0=off)", placeholder="0", required=False, max_length=6)

        def __init__(self, channel_config_file: str, channel_id: int, bot_name: str):
            super().__init__()
            self.channel_config_file = channel_config_file
            self.channel_id = channel_id
            self.bot_name = bot_name

        async def on_submit(self, interaction: discord.Interaction):
            if not getattr(getattr(interaction.user, "guild_permissions", None), "administrator", False):
                await interaction.response.send_message("🚫 Administrator permissions required.", ephemeral=True)
                return

            updates = {}
            try:
                if self.temp_input.value.strip():
                    updates["temperature"] = max(0.1, min(1.5, float(self.temp_input.value.strip())))
                if self.resp_input.value.strip():
                    updates["resp_chance"] = max(0.0, min(1.0, float(self.resp_input.value.strip())))
                if self.emoji_input.value.strip():
                    updates["emoji_chance"] = max(0.0, min(1.0, float(self.emoji_input.value.strip())))
                if self.rand_input.value.strip():
                    updates["random_interval"] = max(0.0, float(self.rand_input.value.strip()))
            except ValueError:
                await interaction.response.send_message("⚠️ Invalid number format.", ephemeral=True)
                return

            if updates:
                def _apply(c: dict) -> dict:
                    c.setdefault(self.channel_id, {}).update(updates)
                    return c
                update_json_file(self.channel_config_file, {}, _apply)

            await interaction.response.send_message(
                f"✅ Updated **{self.bot_name}** channel parameters: `{updates}`",
                ephemeral=True,
            )


    class AdminControlPanelView(discord.ui.View):
        """Command UX #4 — Interactive 1-Click Admin Channel Control Panel."""
        def __init__(self, channel_config_file: str, channel_id: int, bot_name: str):
            super().__init__(timeout=180)
            self.channel_config_file = channel_config_file
            self.channel_id = channel_id
            self.bot_name = bot_name

        async def _toggle_key(self, interaction: discord.Interaction, key: str, default_val: bool = True):
            if not getattr(getattr(interaction.user, "guild_permissions", None), "administrator", False):
                await interaction.response.send_message("🚫 Only Server Administrators can use the Admin Panel.", ephemeral=True)
                return
            holder = {"new_val": not default_val}

            def _mut(c: dict) -> dict:
                ch_d = c.setdefault(self.channel_id, {})
                cur = bool(ch_d.get(key, default_val))
                holder["new_val"] = not cur
                ch_d[key] = holder["new_val"]
                return c

            update_json_file(self.channel_config_file, {}, _mut)
            await interaction.response.send_message(
                f"⚙️ **{self.bot_name}** `{key}` is now set to `{holder['new_val']}` in <#{self.channel_id}>.",
                ephemeral=True,
            )

        @discord.ui.button(label="⏯️ Toggle Bot Enabled", style=discord.ButtonStyle.primary, row=0)
        async def btn_enable(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._toggle_key(interaction, "enabled", True)

        @discord.ui.button(label="🤐 Toggle React-Only", style=discord.ButtonStyle.secondary, row=0)
        async def btn_reactonly(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._toggle_key(interaction, "react_only", False)

        @discord.ui.button(label="🎨 Toggle Image/Meme Gen", style=discord.ButtonStyle.secondary, row=0)
        async def btn_img(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._toggle_key(interaction, "enable_image", True)

        @discord.ui.button(label="🔊 Toggle Voice TTS", style=discord.ButtonStyle.secondary, row=1)
        async def btn_voice(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._toggle_key(interaction, "enable_voice", True)

        @discord.ui.button(label="🎚️ Tune Sliders / Chances", style=discord.ButtonStyle.success, row=1)
        async def btn_tune(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not getattr(getattr(interaction.user, "guild_permissions", None), "administrator", False):
                await interaction.response.send_message("🚫 Only Server Administrators can tune channel parameters.", ephemeral=True)
                return
            await interaction.response.send_modal(
                TuneChannelModal(self.channel_config_file, self.channel_id, self.bot_name)
            )

        @discord.ui.button(label="🖥️ Web-UI Dashboard Link", style=discord.ButtonStyle.primary, row=1)
        async def btn_webui_link(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not getattr(getattr(interaction.user, "guild_permissions", None), "administrator", False):
                await interaction.response.send_message("🚫 Only Server Administrators can access the Web-UI Control Plane.", ephemeral=True)
                return
            import os as _os
            tok = _os.getenv("WEBUI_ADMIN_TOKEN", "").strip()
            pub_url = _os.getenv("WEBUI_PUBLIC_URL", "").strip().rstrip("/")
            local_link = f"http://127.0.0.1:8765/?token={tok}" if tok else "http://127.0.0.1:8765"
            pub_line = f"\n• **Remote / Tunnel Admin Link:** {pub_url}/?token={tok}" if pub_url else ""
            await interaction.response.send_message(
                f"🔒 **Private Admin Web-UI Control Plane Access (Do NOT share this link!):**\n"
                f"• **Local / SSH Port-Forward Link:** {local_link}{pub_line}\n"
                f"• **Admin Token:** `{tok or 'Localhost Auto-Auth'}`\n"
                "*Tip: Run `./manage_bots.sh --host 0.0.0.0` for LAN access, or set `WEBUI_PUBLIC_URL` in `.env` if using Tailscale / Cloudflare Tunnel!*",
                ephemeral=True,
            )


    class DossierMemberSelect(discord.ui.Select):
        def __init__(self, hub_view: "DossierHubView", candidates: list[str], other_bot_names: set[str]):
            self.hub_view = hub_view
            options = []
            for uname in candidates[:25]:
                is_b = any(_extract_base_username(uname) == _extract_base_username(bn) for bn in other_bot_names)
                options.append(
                    discord.SelectOption(
                        label=uname[:95],
                        value=uname[:95],
                        emoji="🤖" if is_b else "👤",
                        description="Fellow Bot Co-Star" if is_b else "Server Member",
                    )
                )
            if not options:
                options.append(discord.SelectOption(label=hub_view.current_target[:95], value=hub_view.current_target[:95]))
            super().__init__(
                placeholder="🔍 Switch Dossier Target (Member or Fellow Bot)...",
                min_values=1,
                max_values=1,
                options=options,
                row=0,
            )

        async def callback(self, interaction: discord.Interaction):
            self.hub_view.current_target = self.values[0]
            embed = self.hub_view.build_Target_dossier_embed(self.hub_view.current_target)
            await interaction.response.edit_message(embed=embed, view=self.hub_view)


    class DossierHubView(discord.ui.View):
        """
        Command UX #2 — Interactive Multi-Tab Dossier Hub (`[🗂️ Dossier]`, `[🌐 Social Graph]`,
        `[📖 Chapters]`, `[✏️ Manage My Facts]`, `[⚙️ Admin Panel]`) + Member/Bot Selector Dropdown.
        """
        def __init__(
            self,
            *,
            bot_display_name: str,
            current_target: str,
            persona_prompt: str,
            user_profiles_file: str,
            shared_image_cache_file: str,
            chat_memory_file: str,
            channel_config_file: str,
            channel_id: int,
            channel_name: str,
            other_bot_names: set[str],
        ):
            super().__init__(timeout=240)
            self.bot_display_name = bot_display_name
            self.current_target = current_target
            self.persona_prompt = persona_prompt
            self.user_profiles_file = user_profiles_file
            self.shared_image_cache_file = shared_image_cache_file
            self.chat_memory_file = chat_memory_file
            self.channel_config_file = channel_config_file
            self.channel_id = channel_id
            self.channel_name = channel_name
            self.other_bot_names = set(other_bot_names or set())

            uprof = load_json_file(user_profiles_file, {})
            cands = [k for k in uprof.keys() if isinstance(k, str) and not k.startswith("_")]
            for ob in self.other_bot_names:
                if ob and ob not in cands:
                    cands.append(ob)
            if current_target and current_target not in cands:
                cands.insert(0, current_target)
            if cands:
                self.add_item(DossierMemberSelect(self, cands, self.other_bot_names))

        def build_Target_dossier_embed(self, target_name: str) -> discord.Embed:
            uprof = load_json_file(self.user_profiles_file, {})
            canon = find_canonical_username(target_name, uprof, {target_name} | self.other_bot_names)
            facts = uprof.get(canon, []) if isinstance(uprof.get(canon), list) else []
            is_b = any(_extract_base_username(canon) == _extract_base_username(bn) for bn in self.other_bot_names)
            scache = load_json_file(self.shared_image_cache_file, {})
            last_seen_map = scache.get("_user_last_seen_ts", {}) if isinstance(scache.get("_user_last_seen_ts"), dict) else {}
            last_ts = last_seen_map.get(canon.lower()) or last_seen_map.get(_extract_base_username(canon))
            return build_whoami_embed(
                bot_display_name=self.bot_display_name,
                target_canon=canon,
                facts=facts,
                is_target_bot=is_b,
                persona_prompt=self.persona_prompt,
                last_seen_ts=last_ts,
            )

        @discord.ui.button(label="🗂️ Dossier", style=discord.ButtonStyle.primary, row=1)
        async def tab_dossier(self, interaction: discord.Interaction, button: discord.ui.Button):
            embed = self.build_Target_dossier_embed(self.current_target)
            await interaction.response.edit_message(embed=embed, view=self)

        @discord.ui.button(label="🌐 Social Graph", style=discord.ButtonStyle.secondary, row=1)
        async def tab_social(self, interaction: discord.Interaction, button: discord.ui.Button):
            uprof = load_json_file(self.user_profiles_file, {})
            embed = build_social_graph_embed(
                bot_display_name=self.bot_display_name,
                user_profiles=uprof,
                other_bot_names=self.other_bot_names,
                persona_prompt=self.persona_prompt,
            )
            await interaction.response.edit_message(embed=embed, view=self)

        @discord.ui.button(label="📖 Chapters", style=discord.ButtonStyle.secondary, row=1)
        async def tab_chapters(self, interaction: discord.Interaction, button: discord.ui.Button):
            chapters = get_merged_episodic_chapters(
                self.shared_image_cache_file, self.chat_memory_file, self.channel_id
            )
            embed = build_episodic_chapters_embed(
                bot_display_name=self.bot_display_name,
                channel_name=self.channel_name,
                chapters_list=chapters,
            )
            await interaction.response.edit_message(embed=embed, view=self)

        @discord.ui.button(label="✏️ Manage My Facts", style=discord.ButtonStyle.success, row=2)
        async def btn_manage_own_facts(self, interaction: discord.Interaction, button: discord.ui.Button):
            # STRICT CALLER ISOLATION: Always binds to interaction.user.display_name!
            caller_name = getattr(interaction.user, "display_name", "User")
            view = EphemeralFactManageView(self.user_profiles_file, caller_name)
            await interaction.response.send_message(
                f"🔒 **Private Fact Manager for {caller_name}** (You can only view, add, or delete facts on your own profile):",
                view=view,
                ephemeral=True,
            )

        @discord.ui.button(label="⚙️ Admin Panel", style=discord.ButtonStyle.secondary, row=2)
        async def btn_admin_panel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not getattr(getattr(interaction.user, "guild_permissions", None), "administrator", False):
                await interaction.response.send_message(
                    "🚫 Only verified Server Administrators can open the Admin Control Panel.",
                    ephemeral=True,
                )
                return
            view = AdminControlPanelView(self.channel_config_file, self.channel_id, self.bot_display_name)
            await interaction.response.send_message(
                f"⚙️ **{self.bot_display_name} Admin Control Panel** (<#{self.channel_id}>):",
                view=view,
                ephemeral=True,
            )


    class GeneratedMediaView(discord.ui.View):
        """
        Crown Jewel #4 — Interactive Micro-Controls attached to generated images & memes:
        - `[🎨 View Prompt]`: sends the exact FLUX visual prompt & meme captions ephemerally.
        - `[🔄 Redraw]`: generates a fresh seed variation using `generate_image_attachment`.
        """
        def __init__(
            self,
            *,
            img_prompt: str,
            meme_top_text: str = "",
            meme_bottom_text: str = "",
            image_gen_model: str = "black-forest-labs/FLUX-1-schnell",
            api_key: str = "",
            bot_id: int = 0,
            bot_name: str = "Bot",
            channel_id: int = 0,
            trigger_msg_id: int = 0,
            shared_image_cache_file: str = "",
        ):
            super().__init__(timeout=300)
            self.img_prompt = img_prompt
            self.meme_top_text = meme_top_text
            self.meme_bottom_text = meme_bottom_text
            self.image_gen_model = image_gen_model
            self.api_key = api_key
            self.bot_id = bot_id
            self.bot_name = bot_name
            self.channel_id = channel_id
            self.trigger_msg_id = trigger_msg_id
            self.shared_image_cache_file = shared_image_cache_file
            self._redraw_count = 0

        @discord.ui.button(label="🎨 View Prompt", style=discord.ButtonStyle.secondary)
        async def view_prompt_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            cap_info = (
                f"\n**Top Caption:** `{self.meme_top_text}`\n**Bottom Caption:** `{self.meme_bottom_text}`"
                if (self.meme_top_text or self.meme_bottom_text)
                else ""
            )
            await interaction.response.send_message(
                f"🎨 **{self.bot_name} Visual Prompt:**\n> {self.img_prompt}{cap_info}",
                ephemeral=True,
            )

        @discord.ui.button(label="🔄 Redraw", style=discord.ButtonStyle.primary)
        async def redraw_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            if self._redraw_count >= 3:
                await interaction.response.send_message("⏳ Redraw limit (3) reached for this image.", ephemeral=True)
                return
            self._redraw_count += 1
            await interaction.response.defer()
            from ai_services import generate_image_attachment

            new_file, err = await generate_image_attachment(
                self.img_prompt,
                self.image_gen_model,
                self.api_key,
                self.bot_id,
                self.bot_name,
                self.channel_id,
                self.trigger_msg_id,
                self.shared_image_cache_file,
                meme_top_text=self.meme_top_text,
                meme_bottom_text=self.meme_bottom_text,
            )
            if new_file:
                is_meme = bool(self.meme_top_text.strip() or self.meme_bottom_text.strip())
                fname = "redrawn_meme.png" if is_meme else "redrawn_image.png"
                await interaction.followup.send(
                    content=f"🔄 **Redrawn by {self.bot_name}** (requested by {interaction.user.display_name}):",
                    file=discord.File(fp=new_file, filename=fname),
                    view=self,
                )
            else:
                await interaction.followup.send(
                    content=err or "*(Redraw timed out; try again in a moment.)*",
                    ephemeral=True,
                )


def create_generated_media_view(
    *,
    img_prompt: str,
    meme_top_text: str = "",
    meme_bottom_text: str = "",
    image_gen_model: str = "black-forest-labs/FLUX-1-schnell",
    api_key: str = "",
    bot_id: int = 0,
    bot_name: str = "Bot",
    channel_id: int = 0,
    trigger_msg_id: int = 0,
    shared_image_cache_file: str = "",
):
    """Factory helper returning a `GeneratedMediaView` instance when `discord.ui` is available."""
    if not _HAS_DISCORD_UI or not img_prompt:
        return None
    try:
        return GeneratedMediaView(
            img_prompt=img_prompt,
            meme_top_text=meme_top_text,
            meme_bottom_text=meme_bottom_text,
            image_gen_model=image_gen_model,
            api_key=api_key,
            bot_id=bot_id,
            bot_name=bot_name,
            channel_id=channel_id,
            trigger_msg_id=trigger_msg_id,
            shared_image_cache_file=shared_image_cache_file,
        )
    except Exception:
        return None


async def handle_bot_commands(
    message: discord.Message,
    client: discord.Client,
    is_admin: bool,
    channel_id: int,
    system_prompt: dict,
    chat_memory_file: str,
    shared_image_cache_file: str,
    long_term_memory_file: str,
    user_profiles_file: str,
    channel_config_file: str,
    general_memory_file: str,
    channel_followup_state: dict[int, dict],
    keywords: list | None = None,
    chance_to_respond: float = 0.05,
    random_emoji_chance: float = 0.45,
    default_random_interval: int = 0,
    temperature: float = 0.78,
    enable_vision: bool = True,
    enable_image_gen: bool = False,
    enable_voice: bool = False,
    enable_gifs: bool = True,
    welcome_test_callback=None,
) -> bool:
    """
    Processes all user (`!help`, `!whoami`, `!relationships`, `!chapters`, `!forgetme`)
    and admin (`admin *`, `!clearmemory`, `!bots status`) commands when triggered via
    `@mention`, direct reply to the bot, keyword prefix (`steward !whoami`), or `!bots` broadcast,
    with automatic Damerau-Levenshtein (<=2) typo forgiveness.
    """
    is_addressed, content_lower = is_command_addressed_to_bot(message, client, keywords)
    if not is_addressed:
        return False

    norm_full_text = normalize_command_typos(message.content or "")
    other_b_meta = get_other_active_bots(
        shared_image_cache_file,
        client.user.id,
        getattr(message.guild, "members", None) if message.guild else None,
    )
    other_b_names = {ob.get("display_name", "") for ob in other_b_meta if ob.get("display_name")}

    def _make_hub_view(target_name: str):
        if not _HAS_DISCORD_UI:
            return None
        try:
            return DossierHubView(
                bot_display_name=client.user.display_name,
                current_target=target_name,
                persona_prompt=system_prompt.get("content", ""),
                user_profiles_file=user_profiles_file,
                shared_image_cache_file=shared_image_cache_file,
                chat_memory_file=chat_memory_file,
                channel_config_file=channel_config_file,
                channel_id=channel_id,
                channel_name=getattr(message.channel, "name", str(channel_id)),
                other_bot_names=other_b_names,
            )
        except Exception:
            return None

    # ==========================================================================
    # USER COMMANDS
    # ==========================================================================
    if "!help" in content_lower or "admin help" in content_lower:
        help_msg = (
            f"🛠️ **{client.user.display_name} Command & Interactive Hub Reference**\n"
            "*Tip: Trigger commands via `@mention`, replying to me, using my keyword (`"
            + (keywords[0] if keywords else client.user.display_name)
            + " !whoami`), or `!bots status`! Typos like `!whoam` are auto-corrected.*\n"
            "**User Commands:**\n"
            f"• `@{client.user.display_name} !whoami [name]` — Open the interactive Dossier Hub, Affinity bar & `[Core]`/`[Boundary]` cards.\n"
            f"• `@{client.user.display_name} !relationships` (`!socialgraph` / `!bonds`) — View my complete server social graph & standings.\n"
            f"• `@{client.user.display_name} !chapters` (`!recap`) — View archived episodic storyline chapters for this channel.\n"
            f"• `@{client.user.display_name} !forgetme <keyword|all>` — Delete a saved fact (or `all`) from **your own** profile.\n"
            f"• `!bots status` / `!bots whoami` — Broadcast status or dossier request across all active bots.\n"
            "**Admin Channel Controls:**\n"
            f"• `@{client.user.display_name} admin status` — View current channel config, telemetry & memory stats.\n"
            f"• `@{client.user.display_name} admin enable` / `admin disable` — Toggle me on/off in this channel.\n"
            f"• `@{client.user.display_name} admin welcome <on|off|status|test>` — Configure or test in-character server-join welcomes.\n"
            f"• `@{client.user.display_name} admin reactonly on` / `off` — Toggle silent emoji-reaction-only mode.\n"
            f"• `@{client.user.display_name} admin chances <resp 0.0-1.0> <emoji 0.0-1.0>` — Set random chime-in & reaction probabilities.\n"
            f"• `@{client.user.display_name} admin random <minutes>` — Set periodic unprompted post timer (`0` = off).\n"
            f"• `@{client.user.display_name} admin features <vision|gifs|image|voice> <on|off>` — Toggle media capabilities.\n"
            f"• `@{client.user.display_name} admin temp <0.1-1.5>` — Adjust creativity temperature.\n"
            "**Admin Memory Management:**\n"
            f"• `@{client.user.display_name} admin memory` — Inspect learned server lore, general facts & user profiles.\n"
            f"• `@{client.user.display_name} admin learn <fact>` / `admin learngeneral <fact>` / `admin learnuser <user> | <fact>`\n"
            f"• `@{client.user.display_name} admin forget <keyword>` / `!clearmemory`"
        )
        await message.channel.send(help_msg)
        return True

    if "!whoami" in content_lower:
        uprof = load_json_file(user_profiles_file, {})
        after_whoami = re.split(r"!whoami", norm_full_text, flags=re.IGNORECASE)[1].strip() if "!whoami" in norm_full_text.lower() else ""
        after_whoami = re.sub(r"<@!?\d+>", "", after_whoami).strip().lstrip("@").strip()
        target_lookup = after_whoami if after_whoami else message.author.display_name
        canon = find_canonical_username(
            target_lookup, uprof, {target_lookup, message.author.display_name} | other_b_names
        )
        facts = uprof.get(canon, [])
        is_target_bot = any(_extract_base_username(canon) == _extract_base_username(bn) for bn in other_b_names)
        scache = load_json_file(shared_image_cache_file, {})
        last_seen_map = scache.get("_user_last_seen_ts", {}) if isinstance(scache.get("_user_last_seen_ts"), dict) else {}
        last_seen_ts = (
            last_seen_map.get(canon.lower())
            or last_seen_map.get(_extract_base_username(canon))
            or (uprof.get("_last_seen", {}) if isinstance(uprof.get("_last_seen"), dict) else {}).get(canon)
        )

        embed = build_whoami_embed(
            bot_display_name=client.user.display_name,
            target_canon=canon,
            facts=facts if isinstance(facts, list) else [],
            is_target_bot=is_target_bot,
            persona_prompt=system_prompt.get("content", ""),
            last_seen_ts=last_seen_ts,
        )
        hub_view = _make_hub_view(canon)
        if hub_view is not None:
            await message.channel.send(embed=embed, view=hub_view)
        else:
            await message.channel.send(embed=embed)
        return True

    if any(cmd in content_lower for cmd in ("!relationships", "!socialgraph", "!bonds", "!standing")):
        uprof = load_json_file(user_profiles_file, {})
        embed = build_social_graph_embed(
            bot_display_name=client.user.display_name,
            user_profiles=uprof,
            other_bot_names=other_b_names,
            persona_prompt=system_prompt.get("content", ""),
        )
        hub_view = _make_hub_view(message.author.display_name)
        if hub_view is not None:
            await message.channel.send(embed=embed, view=hub_view)
        else:
            await message.channel.send(embed=embed)
        return True

    if any(cmd in content_lower for cmd in ("!chapters", "!episodes", "!recap", "!storysofar")):
        chapters_list = get_merged_episodic_chapters(
            shared_image_cache_file, chat_memory_file, channel_id
        )
        embed = build_episodic_chapters_embed(
            bot_display_name=client.user.display_name,
            channel_name=getattr(message.channel, "name", str(channel_id)),
            chapters_list=chapters_list,
        )
        hub_view = _make_hub_view(message.author.display_name)
        if hub_view is not None:
            await message.channel.send(embed=embed, view=hub_view)
        else:
            await message.channel.send(embed=embed)
        return True

    if "!forgetme" in content_lower:
        # STRICT OWNERSHIP INVARIANT: `!forgetme` ONLY ever modifies `message.author.display_name`!
        target_kw = norm_full_text.lower().split("!forgetme", 1)[1].strip() if "!forgetme" in norm_full_text.lower() else ""
        if not target_kw:
            await message.channel.send(f"💡 Usage: `@{client.user.display_name} !forgetme <word from that fact | all>`")
            return True

        removed_list = []
        caller_disp = message.author.display_name

        def _forget_self(uprof_dict):
            canon = find_canonical_username(caller_disp, uprof_dict, {caller_disp})
            existing = uprof_dict.get(canon, [])
            if not isinstance(existing, list) or not existing:
                return uprof_dict
            if target_kw == "all":
                for f in existing:
                    if not RAPPORT_ENTRY_REGEX.match(str(f).strip()):
                        removed_list.append(format_fact_for_display(str(f), include_badge=False))
                uprof_dict[canon] = [f for f in existing if RAPPORT_ENTRY_REGEX.match(str(f).strip())]
            else:
                kept = []
                for f in existing:
                    plain_f = format_fact_for_display(str(f), include_badge=False)
                    if target_kw in plain_f.lower() and not RAPPORT_ENTRY_REGEX.match(str(f).strip()):
                        removed_list.append(plain_f)
                    else:
                        kept.append(f)
                uprof_dict[canon] = kept
            return uprof_dict

        update_json_file(user_profiles_file, {}, _forget_self)
        if removed_list:
            removed_preview = "; ".join(f"`{r}`" for r in removed_list[:5])
            await message.channel.send(
                f"🗑️ Removed **{len(removed_list)}** fact(s) from **{caller_disp}**'s own profile: {removed_preview}"
            )
        else:
            await message.channel.send(
                f"🔍 No saved facts in **{caller_disp}**'s profile matched `{target_kw}`."
            )
        return True

    # ==========================================================================
    # ADMIN COMMANDS
    # ==========================================================================
    if not is_admin:
        return False

    if "admin welcome" in content_lower:
        after_w = content_lower.split("admin welcome", 1)[1].strip()
        sub_cmd = after_w.split()[0] if after_w.split() else "status"
        guild_id_str = f"guild_welcome_{message.guild.id}" if message.guild else f"guild_welcome_{channel_id}"

        if sub_cmd in ("on", "enable", "true"):
            def _w_on(c):
                c[guild_id_str] = {"enabled": True, "channel_id": channel_id}
                c.setdefault(channel_id, {})["welcome_greeter"] = True
                return c
            update_json_file(channel_config_file, {}, _w_on)
            await message.channel.send(
                f"👋 **Welcome Greeter ENABLED** for **{client.user.display_name}** in <#{channel_id}>! "
                f"(Use `@{client.user.display_name} admin welcome test` to preview.)"
            )
            return True
        elif sub_cmd in ("off", "disable", "false"):
            def _w_off(c):
                c[guild_id_str] = {"enabled": False, "channel_id": channel_id}
                c.setdefault(channel_id, {})["welcome_greeter"] = False
                return c
            update_json_file(channel_config_file, {}, _w_off)
            await message.channel.send(f"🔕 **Welcome Greeter DISABLED** for **{client.user.display_name}**.")
            return True
        elif sub_cmd == "test":
            if callable(welcome_test_callback):
                await welcome_test_callback(message.author, message.channel, is_test=True)
            else:
                await message.channel.send(f"*(Welcome test callback not wired on {client.user.display_name}.)*")
            return True
        else:
            conf_all = load_json_file(channel_config_file, {})
            g_conf = conf_all.get(guild_id_str, {})
            w_en = g_conf.get("enabled", conf_all.get(channel_id, {}).get("welcome_greeter", False))
            w_ch = g_conf.get("channel_id", channel_id)
            await message.channel.send(
                f"👋 **{client.user.display_name} Welcome Greeter Status**: Enabled=`{w_en}` | Target Channel=<#{w_ch}>"
            )
            return True

    if "admin enable" in content_lower:
        def _enable(c):
            c.setdefault(channel_id, {})["enabled"] = True
            return c
        update_json_file(channel_config_file, {}, _enable)
        await message.channel.send(f"✅ **{client.user.display_name}** enabled in this channel.")
        return True

    if "admin disable" in content_lower:
        def _disable(c):
            c.setdefault(channel_id, {})["enabled"] = False
            return c
        update_json_file(channel_config_file, {}, _disable)
        await message.channel.send(f"🚫 **{client.user.display_name}** disabled in this channel.")
        return True

    if "admin reactonly on" in content_lower:
        def _react_on(c):
            c.setdefault(channel_id, {})["react_only"] = True
            return c
        update_json_file(channel_config_file, {}, _react_on)
        await message.channel.send(f"🤐 **React-Only Mode ENABLED for {client.user.display_name}.**")
        return True

    if "admin reactonly off" in content_lower:
        def _react_off(c):
            c.setdefault(channel_id, {})["react_only"] = False
            return c
        update_json_file(channel_config_file, {}, _react_off)
        await message.channel.send(f"🗣️ **React-Only Mode DISABLED for {client.user.display_name}.**")
        return True

    if "admin chances" in content_lower:
        nums = re.findall(r"[\d\.]+", content_lower.split("admin chances")[1])
        if len(nums) >= 2:
            r_val = max(0.0, min(1.0, float(nums[0])))
            e_val = max(0.0, min(1.0, float(nums[1])))
            def _chances(c):
                c.setdefault(channel_id, {})["resp_chance"] = r_val
                c[channel_id]["emoji_chance"] = e_val
                return c
            update_json_file(channel_config_file, {}, _chances)
            await message.channel.send(f"🎲 Chances updated for **{client.user.display_name}**: Respond=`{r_val}`, Emoji=`{e_val}`")
        return True

    if "admin random" in content_lower:
        nums = re.findall(r"[\d\.]+", content_lower.split("admin random")[1])
        if len(nums) >= 1:
            int_val = max(0.0, float(nums[0]))
            def _rand(c):
                c.setdefault(channel_id, {})["random_interval"] = int_val
                return c
            update_json_file(channel_config_file, {}, _rand)
            await message.channel.send(f"⏱️ Random post interval set to `{int_val}` minutes.")
        return True

    if "admin temp" in content_lower:
        nums = re.findall(r"[\d\.]+", content_lower.split("admin temp")[1])
        if len(nums) >= 1:
            t_val = max(0.1, min(1.5, float(nums[0])))
            def _temp(c):
                c.setdefault(channel_id, {})["temperature"] = t_val
                return c
            update_json_file(channel_config_file, {}, _temp)
            await message.channel.send(f"🌡️ Temperature for **{client.user.display_name}** in this channel set to `{t_val}`.")
        return True

    if "admin features" in content_lower:
        after_feat = content_lower.split("admin features", 1)[1].strip()
        parts = after_feat.split()
        if len(parts) >= 2 and parts[0] in ("vision", "gifs", "image", "voice") and parts[1] in ("on", "off", "true", "false"):
            feat_key = f"enable_{parts[0]}"
            feat_val = parts[1] in ("on", "true")
            def _feat(c):
                c.setdefault(channel_id, {})[feat_key] = feat_val
                return c
            update_json_file(channel_config_file, {}, _feat)
            await message.channel.send(f"⚙️ Feature `{parts[0]}` set to `{feat_val}` for **{client.user.display_name}** in this channel.")
        return True

    if "admin webui" in content_lower or "admin dashboard" in content_lower:
        import os as _os
        scache_w = load_json_file(shared_image_cache_file, {})
        w_info = scache_w.get("_webui_control_plane_info", {}) if isinstance(scache_w.get("_webui_control_plane_info"), dict) else {}
        tok = _os.getenv("WEBUI_ADMIN_TOKEN", "").strip()
        local_u = w_info.get("local_url", "http://127.0.0.1:8765")
        lan_u = w_info.get("lan_url", local_u)
        pub_u = w_info.get("public_url") or _os.getenv("WEBUI_PUBLIC_URL", "").strip().rstrip("/")
        q_sfx = f"/?token={tok}" if tok else ""
        dm_text = (
            f"🔒 **{client.user.display_name} — Admin Web-UI Control Plane Access**\n"
            f"• **Localhost / VS Code SSH Link:** {local_u}{q_sfx}\n"
            f"• **Home LAN Link (`--host 0.0.0.0`):** {lan_u}{q_sfx}\n"
            + (f"• **Remote Tunnel Link (`WEBUI_PUBLIC_URL`):** {pub_u}{q_sfx}\n" if pub_u else "")
            + f"• **Admin Access Token (`WEBUI_ADMIN_TOKEN`):** `{tok or 'Not set (Localhost mode)'}`\n"
            "*Keep this link private—anyone with the token can manage bots, memory salience, and channel configs!*"
        )
        try:
            await message.author.send(dm_text)
            await message.channel.send(
                f"📬 Sent your private **Web-UI Control Plane** magic login link via DM, **{message.author.display_name}**! "
                "*(Or click `[⚙️ Admin Panel]` -> `[🖥️ Web-UI Dashboard Link]` on any `!whoami` card for an in-channel ephemeral link.)*"
            )
        except Exception:
            await message.channel.send(
                f"🔒 Could not DM you (DMs closed). Open `!whoami` -> `[⚙️ Admin Panel]` -> `[🖥️ Web-UI Dashboard Link]` to view your private link ephemerally!"
            )
        return True

    if "admin status" in content_lower:
        conf_file_data = load_json_file(channel_config_file, {})
        conf_all = conf_file_data.get(channel_id, {})
        guild_id_str = f"guild_welcome_{message.guild.id}" if message.guild else f"guild_welcome_{channel_id}"
        w_en = conf_file_data.get(guild_id_str, {}).get("enabled", conf_all.get("welcome_greeter", False))
        ltm = load_json_file(long_term_memory_file, [])
        gen_mem = load_json_file(general_memory_file, [])
        uprof = load_json_file(user_profiles_file, {})
        cmem_all = load_json_file(chat_memory_file, {})
        cmem = cmem_all.get(channel_id, [])
        chapters_cnt = len(get_merged_episodic_chapters(shared_image_cache_file, chat_memory_file, channel_id))
        status_msg = (
            f"📊 **{client.user.display_name} Channel Status**\n"
            f"• Enabled: `{conf_all.get('enabled', True)}` | React-Only: `{conf_all.get('react_only', False)}` | Welcome Greeter: `{w_en}`\n"
            f"• Respond Chance: `{conf_all.get('resp_chance', chance_to_respond)}` | Emoji Chance: `{conf_all.get('emoji_chance', random_emoji_chance)}` | Random Interval: `{conf_all.get('random_interval', default_random_interval)}m`\n"
            f"• Temperature: `{conf_all.get('temperature', temperature)}` | Keywords: `{', '.join(keywords or [])}`\n"
            f"• Active Media: Vision=`{conf_all.get('enable_vision', enable_vision)}`, Image/MemeGen=`{conf_all.get('enable_image', enable_image_gen)}`, Voice=`{conf_all.get('enable_voice', enable_voice)}`, GIFs=`{conf_all.get('enable_gifs', enable_gifs)}`\n"
            f"• Memory: `{len(cmem)}` recent msgs | `{chapters_cnt}` episodic chapters | `{len(ltm)}` server facts | `{len(gen_mem)}` general facts | `{len(uprof)}` user profiles"
        )
        await message.channel.send(status_msg)
        return True

    if "admin memory" in content_lower:
        ltm = load_json_file(long_term_memory_file, [])
        gen_mem = load_json_file(general_memory_file, [])
        uprof = load_json_file(user_profiles_file, {})
        ltm_str = "\n".join(f"• {format_fact_for_display(f, include_badge=True)}" for f in ltm[-8:]) if ltm else "None"
        gen_str = "\n".join(f"• {format_fact_for_display(f, include_badge=True)}" for f in gen_mem[-8:]) if gen_mem else "None"
        uprof_str = "\n".join(
            f"• **{u}**: {'; '.join(format_fact_for_display(str(x), include_badge=True) for x in facts[-5:])}"
            for u, facts in uprof.items()
            if not str(u).startswith("_") and isinstance(facts, list) and facts
        ) if uprof else "None"
        mem_msg = (
            f"🧠 **{client.user.display_name} Memory Snapshot**\n"
            f"**Server Lore:**\n{ltm_str}\n"
            f"**Shared General Facts:**\n{gen_str}\n"
            f"**User Profiles:**\n{uprof_str}"
        )
        await message.channel.send(mem_msg[:1950])
        return True

    if "admin learnuser" in content_lower:
        raw_arg = re.split(r"admin\s+learnuser", message.content, flags=re.IGNORECASE)[1].strip()
        if "|" in raw_arg:
            u_part, f_part = raw_arg.split("|", 1)
            u_clean = u_part.strip().lstrip("@").strip()
            f_clean = f_part.strip().rstrip(".")
            safe_f = sanitize_or_reject_sensitive_pii(f_clean)
            if not safe_f:
                await message.channel.send("🛡️ **Zero-PII Firewall**: Fact blocked due to sensitive PII (last name, address, phone, email, or card).")
                return True
            if u_clean and safe_f:
                def _manual_uprof(uprof_data):
                    canon = find_canonical_username(u_clean, uprof_data, {u_clean})
                    uprof_data.setdefault(canon, [])
                    consolidate_and_append_fact(safe_f, uprof_data[canon], max_items=35)
                    return uprof_data
                update_json_file(user_profiles_file, {}, _manual_uprof)
                await message.channel.send(f"📝 Saved fact for **{u_clean}**: `{format_fact_for_display(safe_f, include_badge=True)}`")
        return True

    if "admin learngeneral" in content_lower:
        raw_fact = re.split(r"admin\s+learngeneral", message.content, flags=re.IGNORECASE)[1].strip().rstrip(".")
        safe_f = sanitize_or_reject_sensitive_pii(raw_fact)
        if not safe_f:
            await message.channel.send("🛡️ **Zero-PII Firewall**: Fact blocked due to sensitive PII.")
            return True
        if safe_f:
            def _manual_gen(gen_list):
                if not isinstance(gen_list, list):
                    gen_list = []
                consolidate_and_append_fact(safe_f, gen_list, max_items=80)
                return gen_list
            update_json_file(general_memory_file, [], _manual_gen)
            await message.channel.send(f"🌐 Saved shared general fact: `{format_fact_for_display(safe_f, include_badge=True)}`")
        return True

    if "admin learn" in content_lower:
        raw_fact = re.split(r"admin\s+learn", message.content, flags=re.IGNORECASE)[1].strip().rstrip(".")
        safe_f = sanitize_or_reject_sensitive_pii(raw_fact)
        if not safe_f:
            await message.channel.send("🛡️ **Zero-PII Firewall**: Fact blocked due to sensitive PII.")
            return True
        if safe_f:
            def _manual_ltm(ltm_list):
                if not isinstance(ltm_list, list):
                    ltm_list = []
                consolidate_and_append_fact(safe_f, ltm_list, max_items=60)
                return ltm_list
            update_json_file(long_term_memory_file, [], _manual_ltm)
            await message.channel.send(f"📝 Saved server fact: `{format_fact_for_display(safe_f, include_badge=True)}`")
        return True

    if "admin forget" in content_lower:
        target_kw = message.content.lower().split("admin forget", 1)[1].strip()
        if target_kw:
            removed_count = 0
            def _forget_ltm(ltm_list):
                nonlocal removed_count
                kept = [f for f in ltm_list if target_kw not in str(f).lower()]
                removed_count += (len(ltm_list) - len(kept))
                return kept
            update_json_file(long_term_memory_file, [], _forget_ltm)
            update_json_file(general_memory_file, [], _forget_ltm)

            def _forget_uprof(uprof_dict):
                nonlocal removed_count
                for u in list(uprof_dict.keys()):
                    if isinstance(uprof_dict[u], list):
                        before = len(uprof_dict[u])
                        uprof_dict[u] = [f for f in uprof_dict[u] if target_kw not in str(f).lower()]
                        removed_count += (before - len(uprof_dict[u]))
                return uprof_dict
            update_json_file(user_profiles_file, {}, _forget_uprof)
            await message.channel.send(f"🗑️ Removed `{removed_count}` memory entries matching `{target_kw}`.")
        return True

    if "!clearmemory" in content_lower:
        def _clear(cm):
            cm[channel_id] = []
            return cm
        update_json_file(chat_memory_file, {}, _clear)

        def _clear_ch_cache(c):
            gen_list = c.get("_generated_images", [])
            if isinstance(gen_list, list):
                c["_generated_images"] = [
                    r for r in gen_list
                    if isinstance(r, dict) and r.get("channel_id") != channel_id
                ]
            f_map = c.get("_active_followup_bot", {})
            if isinstance(f_map, dict):
                f_map.pop(channel_id, None)
                f_map.pop(str(channel_id), None)
            return c
        update_json_file(shared_image_cache_file, {}, _clear_ch_cache)

        channel_followup_state[channel_id] = {
            "timestamp": 0.0,
            "turns_used": 999,
            "active": False,
            "bot_msg_id": "",
        }
        await message.channel.send("🧹 **Short-term channel memory & recent media cache cleared.**")
        return True

    return False

def register_hub_slash_command(
    client: discord.Client,
    *,
    system_prompt: dict,
    chat_memory_file: str,
    shared_image_cache_file: str,
    user_profiles_file: str,
    channel_config_file: str,
):
    """
    Registers a single, clutter-free `/hub` Slash Command (`discord.app_commands.CommandTree`)
    that opens the interactive `DossierHubView` card (Dossier, Social Graph, Episodic Chapters,
    Manage My Facts, and Admin Panel—all in one place!) with an optional `target` member/bot name
    and `private` (ephemeral) toggle.
    """
    app_commands = getattr(discord, "app_commands", None)
    if app_commands is None or not hasattr(app_commands, "CommandTree"):
        return None

    tree = getattr(client, "_hub_cmd_tree", None)
    if tree is not None:
        return tree

    try:
        tree = app_commands.CommandTree(client)
        client._hub_cmd_tree = tree
    except Exception:
        return None

    @tree.command(
        name="hub",
        description="Open the interactive Dossier, Social Graph, Chapters & Admin Control Hub card.",
    )
    @app_commands.describe(
        target="Optional: Name of a server member or fellow bot to inspect (defaults to you)",
        private="Optional: Show the interactive card only to you (default: False)",
    )
    async def slash_hub_cmd(
        interaction: discord.Interaction,
        target: str = "",
        private: bool = False,
    ):
        try:
            await interaction.response.defer(ephemeral=bool(private))
        except Exception:
            pass

        try:
            bot_disp = getattr(client.user, "display_name", "Bot")
            caller_disp = getattr(interaction.user, "display_name", "User")
            ch_id = getattr(interaction.channel, "id", 0) or 0
            ch_name = getattr(interaction.channel, "name", str(ch_id))

            guild_members = getattr(interaction.guild, "members", None) if getattr(interaction, "guild", None) else None
            other_b_meta = get_other_active_bots(
                shared_image_cache_file,
                getattr(client.user, "id", 0),
                guild_members,
            )
            other_b_names = {ob.get("display_name", "") for ob in other_b_meta if ob.get("display_name")}

            uprof = load_json_file(user_profiles_file, {})
            clean_target = re.sub(r"<@!?\d+>", "", (target or "")).strip().lstrip("@").strip()
            target_lookup = clean_target if clean_target else caller_disp
            canon = find_canonical_username(
                target_lookup, uprof, {target_lookup, caller_disp} | other_b_names
            )
            facts = uprof.get(canon, [])
            is_target_bot = any(_extract_base_username(canon) == _extract_base_username(bn) for bn in other_b_names)

            scache = load_json_file(shared_image_cache_file, {})
            last_seen_map = scache.get("_user_last_seen_ts", {}) if isinstance(scache.get("_user_last_seen_ts"), dict) else {}
            last_seen_ts = (
                last_seen_map.get(canon.lower())
                or last_seen_map.get(_extract_base_username(canon))
                or (uprof.get("_last_seen", {}) if isinstance(uprof.get("_last_seen"), dict) else {}).get(canon)
            )

            embed = build_whoami_embed(
                bot_display_name=bot_disp,
                target_canon=canon,
                facts=facts if isinstance(facts, list) else [],
                is_target_bot=is_target_bot,
                persona_prompt=system_prompt.get("content", ""),
                last_seen_ts=last_seen_ts,
            )

            view = None
            if _HAS_DISCORD_UI:
                try:
                    view = DossierHubView(
                        bot_display_name=bot_disp,
                        current_target=canon,
                        persona_prompt=system_prompt.get("content", ""),
                        user_profiles_file=user_profiles_file,
                        shared_image_cache_file=shared_image_cache_file,
                        chat_memory_file=chat_memory_file,
                        channel_config_file=channel_config_file,
                        channel_id=ch_id,
                        channel_name=ch_name,
                        other_bot_names=other_b_names,
                    )
                except Exception:
                    view = None

            if view is not None:
                await interaction.followup.send(embed=embed, view=view, ephemeral=bool(private))
            else:
                await interaction.followup.send(embed=embed, ephemeral=bool(private))
        except Exception as e:
            print(f"[{getattr(client.user, 'display_name', 'Bot')}] [SLASH ERROR] /hub failed: {e}")
            try:
                await interaction.followup.send(f"⚠️ Error opening Hub card: `{e}`", ephemeral=True)
            except Exception:
                pass

    return tree
