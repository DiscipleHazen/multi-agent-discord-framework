import re
import time
import discord

from core_system import load_json_file, update_json_file
from bot_discovery import get_other_active_bots
from memory_and_context import (
    find_canonical_username, consolidate_and_append_fact,
    _extract_base_username, _extract_saved_rapport_state_full,
    _get_disposition_tier_label, _derive_relationship_archetype,
    RAPPORT_ENTRY_REGEX
)


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
    """
    Returns a Discord Embed hex color matching the 5 Disposition Tiers.
    """
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
    """
    Formats a human-readable relative timestamp for Embed footers/fields.
    """
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
    last_seen_ts: float | None = None
) -> discord.Embed:
    """
    Builds a color-coded Discord Embed card for `@BotName !whoami [target]`.
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
        color=embed_color
    )
    embed.add_field(
        name="Disposition & Affinity",
        value=f"**{tier_lbl}**\n{bar_str} • Trend: `{tr}`",
        inline=False
    )
    if cat:
        embed.add_field(
            name="Recent Relationship Catalyst",
            value=f"*{cat}*",
            inline=False
        )
    if last_seen_ts:
        embed.add_field(
            name="Last Seen",
            value=format_relative_last_seen(last_seen_ts),
            inline=True
        )

    non_rapport_facts = [
        str(f).strip() for f in (facts or [])
        if str(f).strip() and not RAPPORT_ENTRY_REGEX.match(str(f).strip())
    ]
    if non_rapport_facts:
        facts_lines = "\n".join(f"• {f}" for f in non_rapport_facts[-15:])
        if len(facts_lines) > 1000:
            facts_lines = facts_lines[:995] + "..."
        embed.add_field(
            name=f"Logged Dossier Facts ({len(non_rapport_facts)})",
            value=facts_lines,
            inline=False
        )
    else:
        embed.add_field(
            name="Logged Dossier Facts",
            value="*No additional personal facts logged yet.*",
            inline=False
        )
    return embed


def build_social_graph_embed(
    bot_display_name: str,
    user_profiles: dict,
    other_bot_names: set[str],
    persona_prompt: str
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
        # Skip completely empty 0-score unknown entries without facts
        if sc == 0 and not facts and not is_bot:
            continue
        arch = _derive_relationship_archetype(sc, tr, is_fellow_bot=is_bot, had_past_friction=(sc < 0))
        entries.append((sc, uname, is_bot, tr, cat, arch))

    entries.sort(key=lambda x: x[0], reverse=True)

    embed = discord.Embed(
        title=f"🌐 {bot_display_name}'s Server Social Graph & Standings",
        description="Live relationship tiers, archetypes, and catalysts across humans & fellow bots.",
        color=0x5865F2
    )

    if not entries:
        embed.add_field(
            name="No Relationships Logged Yet",
            value="*Chat with me in the server to build rapport (or rivalry)!*",
            inline=False
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


def build_episodic_chapters_embed(
    bot_display_name: str,
    channel_name: str,
    chapters_list: list
) -> discord.Embed:
    """
    Builds a Discord Embed displaying rolling Episodic Session Chapters for `@BotName !chapters` / `!recap`.
    """
    embed = discord.Embed(
        title=f"📖 Episodic Session Chapters — #{channel_name}",
        description=f"Multi-session storylines and notable room events remembered by **{bot_display_name}**.",
        color=0x9B59B6
    )
    if not chapters_list:
        embed.add_field(
            name="No Archived Chapters Yet",
            value="*Chapters are automatically archived when a lively conversation session concludes.*",
            inline=False
        )
        return embed

    for idx, ch_item in enumerate(chapters_list[-8:], start=1):
        if isinstance(ch_item, dict):
            title_lbl = ch_item.get("timestamp_label") or f"Chapter {idx}"
            summary_txt = ch_item.get("summary") or str(ch_item)
        else:
            title_lbl = f"Episode {idx}"
            summary_txt = str(ch_item)
        embed.add_field(
            name=f"🔖 {title_lbl}",
            value=summary_txt[:450],
            inline=False
        )
    return embed


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
    welcome_test_callback=None
) -> bool:
    """
    Processes all user (`!help`, `!whoami`, `!relationships`, `!chapters`, `!forgetme`)
    and admin (`admin *`, `!clearmemory`) commands when `client.user` is mentioned.
    Returns True if a command was matched and handled (so `on_message` can return early),
    or False if normal message routing should continue.
    """
    if not client.user.mentioned_in(message):
        return False

    content_lower = (message.content or "").lower()

    # ==========================================================================
    # USER COMMANDS
    # ==========================================================================
    if "!help" in content_lower or "admin help" in content_lower:
        help_msg = (
            f"🛠️ **{client.user.display_name} Command Reference**\n"
            "**User Commands:**\n"
            f"• `@{client.user.display_name} !whoami [name]` — View my rich dossier & relationship card for you (or another member/bot).\n"
            f"• `@{client.user.display_name} !relationships` (`!socialgraph` / `!bonds`) — View my complete server social graph & standings.\n"
            f"• `@{client.user.display_name} !chapters` (`!recap`) — View archived episodic session chapters for this channel.\n"
            f"• `@{client.user.display_name} !forgetme <keyword|all>` — Delete a wrongly saved fact (or `all`) from your own profile.\n"
            f"• `@{client.user.display_name} !help` — Show this command guide.\n"
            "**Admin Channel Controls:**\n"
            f"• `@{client.user.display_name} admin status` — View current channel config & memory stats.\n"
            f"• `@{client.user.display_name} admin enable` / `admin disable` — Toggle me on/off in this channel.\n"
            f"• `@{client.user.display_name} admin welcome <on|off|status|test>` — Configure or test in-character server-join welcomes in this channel.\n"
            f"• `@{client.user.display_name} admin reactonly on` / `off` — Toggle silent emoji-reaction-only mode.\n"
            f"• `@{client.user.display_name} admin chances <resp 0.0-1.0> <emoji 0.0-1.0>` — Set random chime-in & reaction probabilities.\n"
            f"• `@{client.user.display_name} admin random <minutes>` — Set periodic unprompted post timer (`0` = off).\n"
            f"• `@{client.user.display_name} admin features <vision|gifs|image|voice> <on|off>` — Toggle media capabilities in this channel.\n"
            f"• `@{client.user.display_name} admin temp <0.1-1.5>` — Adjust creativity temperature in this channel.\n"
            "**Admin Memory Management:**\n"
            f"• `@{client.user.display_name} admin memory` — Inspect learned server lore, general facts & user profiles.\n"
            f"• `@{client.user.display_name} admin learn <fact>` — Manually add a server lore fact.\n"
            f"• `@{client.user.display_name} admin learngeneral <fact>` — Manually add a shared general knowledge fact.\n"
            f"• `@{client.user.display_name} admin learnuser <username> | <fact>` — Manually add a fact to a user's profile.\n"
            f"• `@{client.user.display_name} admin forget <keyword>` — Delete any saved memory matching `<keyword>`.\n"
            f"• `@{client.user.display_name} !clearmemory` — Wipe short-term conversation history in this channel."
        )
        await message.channel.send(help_msg)
        return True

    if "!whoami" in content_lower:
        uprof = load_json_file(user_profiles_file, {})
        after_whoami = re.split(r'!whoami', message.content, flags=re.IGNORECASE)[1].strip()
        after_whoami = re.sub(r'<@!?\d+>', '', after_whoami).strip().lstrip("@").strip()
        target_lookup = after_whoami if after_whoami else message.author.display_name
        other_b_meta = get_other_active_bots(
            shared_image_cache_file, client.user.id, getattr(message.guild, "members", None) if message.guild else None
        )
        other_b_names = {ob.get("display_name", "") for ob in other_b_meta if ob.get("display_name")}
        canon = find_canonical_username(target_lookup, uprof, {target_lookup, message.author.display_name} | other_b_names)
        facts = uprof.get(canon, [])
        is_target_bot = any(_extract_base_username(canon) == _extract_base_username(bn) for bn in other_b_names)
        last_seen_map = uprof.get("_last_seen", {}) if isinstance(uprof.get("_last_seen"), dict) else {}
        last_seen_ts = last_seen_map.get(canon) or last_seen_map.get(_extract_base_username(canon))

        embed = build_whoami_embed(
            bot_display_name=client.user.display_name,
            target_canon=canon,
            facts=facts if isinstance(facts, list) else [],
            is_target_bot=is_target_bot,
            persona_prompt=system_prompt.get("content", ""),
            last_seen_ts=last_seen_ts
        )
        await message.channel.send(embed=embed)
        return True

    if any(cmd in content_lower for cmd in ("!relationships", "!socialgraph", "!bonds", "!standing")):
        uprof = load_json_file(user_profiles_file, {})
        other_b_meta = get_other_active_bots(
            shared_image_cache_file, client.user.id, getattr(message.guild, "members", None) if message.guild else None
        )
        other_b_names = {ob.get("display_name", "") for ob in other_b_meta if ob.get("display_name")}
        embed = build_social_graph_embed(
            bot_display_name=client.user.display_name,
            user_profiles=uprof,
            other_bot_names=other_b_names,
            persona_prompt=system_prompt.get("content", "")
        )
        await message.channel.send(embed=embed)
        return True

    if any(cmd in content_lower for cmd in ("!chapters", "!episodes", "!recap", "!storysofar")):
        cmem = load_json_file(chat_memory_file, {})
        ch_key = f"chapters_{channel_id}"
        chapters_list = cmem.get(ch_key, [])
        if not isinstance(chapters_list, list):
            chapters_list = []
        embed = build_episodic_chapters_embed(
            bot_display_name=client.user.display_name,
            channel_name=getattr(message.channel, "name", str(channel_id)),
            chapters_list=chapters_list
        )
        await message.channel.send(embed=embed)
        return True

    if "!forgetme" in content_lower:
        target_kw = message.content.lower().split("!forgetme", 1)[1].strip()
        if not target_kw:
            await message.channel.send(f"💡 Usage: `@{client.user.display_name} !forgetme <word from that fact>`")
            return True

        removed_list = []

        def _forget_self(uprof_dict):
            canon = find_canonical_username(message.author.display_name, uprof_dict, {message.author.display_name})
            existing = uprof_dict.get(canon, [])
            if not isinstance(existing, list) or not existing:
                return uprof_dict
            if target_kw == "all":
                removed_list.extend(existing)
                uprof_dict[canon] = []
            else:
                kept = []
                for f in existing:
                    if target_kw in f.lower():
                        removed_list.append(f)
                    else:
                        kept.append(f)
                uprof_dict[canon] = kept
            return uprof_dict

        update_json_file(user_profiles_file, {}, _forget_self)
        if removed_list:
            removed_preview = "; ".join(f"`{r}`" for r in removed_list[:5])
            await message.channel.send(f"🗑️ Removed **{len(removed_list)}** fact(s) from **{message.author.display_name}**'s profile: {removed_preview}")
        else:
            await message.channel.send(f"🔍 No saved facts in **{message.author.display_name}**'s profile matched `{target_kw}`.")
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
        chapters_cnt = len(cmem_all.get(f"chapters_{channel_id}", []))
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
        ltm_str = "\n".join(f"• {f}" for f in ltm[-8:]) if ltm else "None"
        gen_str = "\n".join(f"• {f}" for f in gen_mem[-8:]) if gen_mem else "None"
        uprof_str = "\n".join(
            f"• **{u}**: {'; '.join(str(x) for x in facts[-5:])}"
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
        raw_arg = re.split(r'admin\s+learnuser', message.content, flags=re.IGNORECASE)[1].strip()
        if "|" in raw_arg:
            u_part, f_part = raw_arg.split("|", 1)
            u_clean = u_part.strip().lstrip("@").strip()
            f_clean = f_part.strip().rstrip(".")
            if u_clean and f_clean:
                def _manual_uprof(uprof_data):
                    canon = find_canonical_username(u_clean, uprof_data, {u_clean})
                    uprof_data.setdefault(canon, [])
                    consolidate_and_append_fact(f_clean, uprof_data[canon], max_items=30)
                    return uprof_data
                update_json_file(user_profiles_file, {}, _manual_uprof)
                await message.channel.send(f"📝 Saved fact for **{u_clean}**: `{f_clean}`")
        return True

    if "admin learngeneral" in content_lower:
        raw_fact = re.split(r'admin\s+learngeneral', message.content, flags=re.IGNORECASE)[1].strip().rstrip(".")
        if raw_fact:
            def _manual_gen(gen_list):
                if not isinstance(gen_list, list):
                    gen_list = []
                consolidate_and_append_fact(raw_fact, gen_list, max_items=80)
                return gen_list
            update_json_file(general_memory_file, [], _manual_gen)
            await message.channel.send(f"🌐 Saved shared general fact: `{raw_fact}`")
        return True

    if "admin learn" in content_lower:
        raw_fact = re.split(r'admin\s+learn', message.content, flags=re.IGNORECASE)[1].strip().rstrip(".")
        if raw_fact:
            def _manual_ltm(ltm_list):
                if not isinstance(ltm_list, list):
                    ltm_list = []
                consolidate_and_append_fact(raw_fact, ltm_list, max_items=60)
                return ltm_list
            update_json_file(long_term_memory_file, [], _manual_ltm)
            await message.channel.send(f"📝 Saved server fact: `{raw_fact}`")
        return True

    if "admin forget" in content_lower:
        target_kw = message.content.lower().split("admin forget", 1)[1].strip()
        if target_kw:
            removed_count = 0
            def _forget_ltm(ltm_list):
                nonlocal removed_count
                kept = [f for f in ltm_list if target_kw not in f.lower()]
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
            "bot_msg_id": ""
        }
        await message.channel.send("🧹 **Short-term channel memory & recent media cache cleared.**")
        return True

    return False
