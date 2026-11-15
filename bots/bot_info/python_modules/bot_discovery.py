import re
import time
import discord

from config import (
    ERROR_REPLY_STRIP_REGEX, IMAGE_DESC_STRIP_REGEX, IMAGE_DESC_EXTRACT_REGEX,
    URL_REGEX, TOPIC_INTEREST_CUES_REGEX, STOPWORDS, BARE_FILLER_UTTERANCE_REGEX,
    TOPIC_CHANGE_REQUEST_REGEX, EXPLICIT_IMAGE_REQ_REGEX, EXPLICIT_MEME_REQ_REGEX,
    EXPLICIT_VOICE_REQ_REGEX, EXPLICIT_GIF_REQ_REGEX, FALLBACK_TEXT_MODELS,
    AFFECTION_GIFT_OR_CARE_REGEX, SITUATIONAL_SHARED_BOND_REGEX,
    PLAYFUL_BANTER_OR_RIVALRY_REGEX, RELATIONSHIP_QUERY_REGEX
)
from core_system import load_json_file, update_json_file, post_with_backoff, is_model_circuit_open
from memory_and_context import normalize_tokens

# ==============================================================================
# PRE-COMPILED REGEXES & CONSTANTS FOR BOT DISCOVERY & FOLLOW-UP ROUTING
# ==============================================================================

NON_NAME_CONVERSATIONAL_STARTERS = frozenset({
    "bro", "bruh", "dude", "man", "girl", "boy", "boys", "guys", "yall", "chat",
    "everyone", "everybody", "someone", "somebody", "anyone", "anybody", "nobody",
    "folks", "team", "gang", "homie", "bestie", "mate", "pal", "bud", "buddy",
    "there", "here", "the", "this", "that", "these", "those", "it", "its", "bot",
    "bots", "what", "whats", "why", "how", "when", "where", "who", "which", "whose",
    "wait", "hold", "look", "see", "watch", "listen", "hear", "imagine", "think",
    "yeah", "yea", "yes", "yep", "yup", "nah", "nope", "no", "not", "never",
    "sure", "fine", "true", "real", "facts", "based", "valid", "fair", "right",
    "maybe", "perhaps", "probably", "honestly", "literally", "actually", "seriously",
    "basically", "apparently", "obviously", "clearly", "definitely", "absolutely",
    "totally", "lowkey", "highkey", "ngl", "tbh", "imo", "imho", "idk", "ikr",
    "lol", "lmao", "lmfao", "rofl", "haha", "hehe", "kek", "wow", "whoa", "woah",
    "damn", "darn", "dang", "wtf", "wth", "omg", "omfg",
    "ugh", "oof", "yikes", "rip", "welp", "well", "so", "and", "but", "or", "yet",
    "because", "since", "while", "though", "although", "unless", "until", "if",
    "like", "just", "only", "even", "still", "already", "also", "too", "very",
    "really", "pretty", "quite", "much", "many", "some", "any", "all", "both",
    "one", "two", "three", "first", "next", "last", "other", "another", "same",
    "good", "great", "nice", "cool", "sweet", "awesome", "amazing", "wild", "crazy",
    "insane", "funny", "hilarious", "weird", "strange", "bad", "sad", "mad", "glad",
    "sorry", "thanks", "thank", "please", "pls", "plz", "help", "stop", "keep",
    "let", "lets", "make", "give", "take", "get", "got", "have", "had", "has",
    "did", "does", "done", "doing", "going", "gon", "gonna", "wanna", "gotta",
    "can", "cant", "could", "would", "should", "will", "wont", "might", "must",
    "may", "shall", "need", "want", "love", "hate", "feel", "felt", "seem", "seems",
    "whatever", "whenever", "wherever", "whoever", "however", "meanwhile", "anyway",
    "anyways", "besides", "otherwise", "instead", "alright", "aight", "bet", "word"
})

VERSION_SUFFIX_STRIP_REGEX = re.compile(r'\s+v?\d+(?:\.\d+)*$', re.IGNORECASE)

STEPPING_AWAY_REGEX = re.compile(
    r'\b(?:brb|gtg|g2g|afk|going to bed|headed to bed|logging off|signing off|'
    r'wife is calling|husband is calling|mom is calling|dad is calling|boss is calling|'
    r'gotta go|have to go|need to go|catch yall later|talk to you later|ttyl|goodnight|gn)\b',
    re.IGNORECASE
)

DISMISSAL_CLOSURE_REGEX = re.compile(
    r'^[\s\W]*(?:uh+[\s\-]+huh+|well\s+anyways?|anyways?|whatever|never\s*mind|nvm|moving\s+on|'
    r'fair\s+enough|makes\s+sense|gotcha|got\s+it|ah\s+ok(?:ay)?|oh\s+ok(?:ay)?|i\s+see|true\s+that|real|valid|facts|bet|word)[\s\W]*$',
    re.IGNORECASE
)

DIRECT_SECOND_PERSON_FOLLOWUP_REGEX = re.compile(
    r'\b(?:could\s+you|can\s+you|would\s+you|will\s+you|show\s+me|draw\s+me|make\s+me|'
    r'tell\s+me|give\s+me|what\s+do\s+you|how\s+do\s+you|are\s+you|do\s+you|what\s+about\s+you|how\s+about\s+you)\b',
    re.IGNORECASE
)

# Detects short clarification follow-ups ("what", "wdym", "what do you mean?") right after a bot speaks
CLARIFICATION_FOLLOWUP_REGEX = re.compile(
    r'^[\s\W]*(?:what\??|wdym\??|what\s+do\s+you\s+mean\??|huh\s*\?+|why\??|how\s+so\??|explain\??)[\s\W]*$',
    re.IGNORECASE
)

# Detects third-person side-comments between humans talking *about* a bot (e.g. "she doesnt like u lol")
THIRD_PERSON_SIDE_COMMENT_REGEX = re.compile(
    r'\b(?:she|her|hers|he|him|his|that\s+bot|this\s+bot|the\s+bot|it)\s+'
    r"(?:doesn['’]?t|dont|don['’]?t|didn['’]?t|didnt|won['’]?t|wont|can['’]?t|cant|isn['’]?t|isnt|ain['’]?t|"
    r'hates|likes|loves|ignored|ignores|ghosted|left|forgot|broke|glitched)\b'
    r'|\b(?:she|he)\s+(?:is|was)\s+(?:ignoring|ghosting|mad\s+at|talking\s+to)\s+(?:u|you|him|her|them)\b',
    re.IGNORECASE
)

# Detects short affirmative answers ("sure", "okay", "yes", "pls") when a bot just asked the user a direct question
AFFIRMATIVE_ANSWER_TO_QUESTION_REGEX = re.compile(
    r'^[\s\W]*(?:sure|ok+|okay|yes|yeah|yea|yep|yup|pls|plz|please|do\s+it|go\s+for\s+it|go\s+ahead|definitely|totally)[\s\W]*$',
    re.IGNORECASE
)

AT_MENTION_START_REGEX = re.compile(
    r'^(?:hey\s+|hi\s+|hello\s+|yo\s+|ok\s+|okay\s+)?@([a-z0-9_\.\-]{2,})\b'
)
GREETING_NAME_START_REGEX = re.compile(
    r'^(?:hey|hi|hello|yo|sup|ay|ayy)\s+([a-z0-9_]{3,})\b'
)
PREPOSITION_TARGET_END_REGEX = re.compile(r'\b(?:of|about|for|at|with|to|from|than)\s+@?$')

COLLECTIVE_BOT_ADDRESS_REGEX = re.compile(
    r'\b(?:hey\s+bots|hi\s+bots|yo\s+bots|all\s+bots|both\s+bots|every\s+bot|you\s+bots)\b',
    re.IGNORECASE
)

CMD_VERBS_PATTERN = (
    r'(?:please\s+)?(?:generate|draw|render|make|create|cook|drop|send|record|say|speak|'
    r'give|show|paint|roast|meme|shitpost|tell|explain|rate|rank|what\s+do\s+you|how\s+do\s+you|are\s+you|do\s+you)'
)


def _strip_bot_version(name: str) -> str:
    return VERSION_SUFFIX_STRIP_REGEX.sub('', name.strip()).strip()


def _term_boundary_pattern(term: str) -> str:
    """
    100% generic word-boundary pattern for `term` that refuses to match when a
    non-possessive `term` is used as a possessive prefix for another entity
    (e.g., `BotName's ...`, `BotName bot's ...`, or `BotName bots ...`).
    """
    esc = re.escape(term)
    if "'s" not in term and "’s" not in term and "bots " not in term:
        return rf"{esc}\b(?!(?:\s+(?:bot|ai))?['’]s\b)(?!\s+bots\s+\w+)(?!s\s+\w+)"
    return rf"{esc}\b"


def _is_subsumed_by_longer_term(text_lower: str, match_start: int, match_end: int, competing_terms: set[str]) -> bool:
    """
    Universal longest-match disambiguator:
    Returns True if any term in `competing_terms` matches at or before `match_start`
    and extends strictly past `match_end` (so 'Solar' never steals a match from
    'Solar Knight' or 'Solar Bot\'s Hound').
    """
    match_len = match_end - match_start
    for comp in competing_terms:
        if not comp or len(comp) <= match_len:
            continue
        for cm in re.finditer(rf'\b{_term_boundary_pattern(comp)}', text_lower):
            if cm.start() <= match_start and cm.end() >= match_end and (cm.end() - cm.start()) > match_len:
                return True
    return False


def _stem_word(w: str) -> str:
    """Normalizes common plural/gerund suffixes so 'clouds' <-> 'cloud' and 'daydreamin' <-> 'daydreaming' match."""
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("in") and len(w) > 5:
        return w[:-2]
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxzho":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _count_token_or_stem_overlap(tokens_a: set[str], tokens_b: set[str]) -> int:
    """Counts shared tokens or shared 4+ char word stems between two token sets."""
    if not tokens_a or not tokens_b:
        return 0
    exact = len(tokens_a & tokens_b)
    if exact > 0:
        return exact
    stems_a = {_stem_word(t) for t in tokens_a if len(t) >= 4}
    stems_b = {_stem_word(t) for t in tokens_b if len(t) >= 4}
    return len(stems_a & stems_b)


def _deactivate_followup_state(channel_followup_state: dict[int, dict], channel_id: int):
    """
    Marks a channel's follow-up state as inactive using a persistent tombstone dict
    rather than popping the key, preventing closed follow-up windows from resurrecting
    on subsequent messages.
    """
    existing = channel_followup_state.get(channel_id)
    if isinstance(existing, dict):
        existing["active"] = False
    else:
        channel_followup_state[channel_id] = {
            "timestamp": 0.0,
            "turns_used": 999,
            "active": False,
            "bot_msg_id": ""
        }


def is_other_bot_in_active_exchange(
    chat_history_list: list[dict],
    my_bot_id: int,
    window_seconds: float = 28.0
) -> bool:
    """
    Checks if a DIFFERENT bot spoke in the channel within the last `window_seconds`.
    Used to suppress unprompted spontaneous chime-ins so bots don't barge in on top
    of another bot's active 1-on-1 conversation with a human user.
    """
    if not chat_history_list:
        return False
    now_ts = time.time()
    for m in reversed(chat_history_list[-6:]):
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        if m.get("author_is_bot") and m.get("author_id") and m.get("author_id") != my_bot_id:
            m_ts = m.get("timestamp")
            if isinstance(m_ts, (int, float)) and (now_ts - float(m_ts)) <= window_seconds:
                return True
        if m.get("author_id") == my_bot_id:
            return False
    return False


def is_directly_addressed_by_fellow_bot(
    text: str,
    message: discord.Message,
    my_bot_id: int,
    my_display_name: str,
    my_keywords: list[str] | None,
    shared_image_cache_file: str,
    replied_to_self: bool = False
) -> bool:
    """
    100% generic detector for when a fellow bot in the server is directly addressing,
    replying to, or bantering with THIS bot by name/keyword, enabling organic bot-to-bot
    rapports, rivalries, and alliances without relying on low random chime-in rolls.
    """
    if not getattr(message.author, "bot", False) or message.author.id == my_bot_id:
        return False
    if replied_to_self or any(getattr(u, "id", None) == my_bot_id for u in getattr(message, "mentions", [])):
        return True

    clean_lower = (text or "").strip().lower()
    if not clean_lower:
        return False

    addressed = detect_addressed_bot_name(
        text, message, my_bot_id, my_display_name, my_keywords, shared_image_cache_file
    )
    if addressed and _strip_bot_version(addressed).lower() == _strip_bot_version(my_display_name).lower():
        return True

    # Also check if this bot was addressed alongside another bot in a multi-bot remark
    if not is_command_directed_at_other_bot(text, message, my_bot_id, my_display_name, my_keywords, shared_image_cache_file):
        my_terms = _build_my_terms_set(my_display_name, my_keywords)
        guild_members = getattr(message.guild, "members", None) if message.guild else None
        other_bots = get_other_active_bots(shared_image_cache_file, my_bot_id, guild_members)
        other_bot_terms = {
            t.strip().lower()
            for ob in other_bots
            for t in (ob.get("aliases", []) + ob.get("keywords", []))
            if t and t.strip().lower() not in my_terms
        }
        for my_t in sorted(my_terms, key=len, reverse=True):
            if not my_t:
                continue
            for m in re.finditer(rf'\b{_term_boundary_pattern(my_t)}', clean_lower):
                if _is_subsumed_by_longer_term(clean_lower, m.start(), m.end(), other_bot_terms):
                    continue
                prefix_slice = clean_lower[max(0, m.start() - 12):m.start()]
                if not PREPOSITION_TARGET_END_REGEX.search(prefix_slice):
                    return True

    return False


def build_universal_media_instructions(
    enable_image_gen: bool = True,
    enable_voice: bool = True,
    enable_gifs: bool = True,
    image_style_hint: str = "",
    voice_role_hint: str = "",
    voice_guidance_hint: str = "",
    allowed_voice_styles: str = "(Happy), (Sad), (Angry), (Calm), (Serious), (Playful), (Deep), (Lazy), (Excited)"
) -> str:
    """
    Reusable, 100% persona-agnostic helper that any bot script can call to generate
    conflict-free media generation, human-like spontaneity, and dynamic relationship
    instructions tailored to its persona.
    """
    rules = [
        "### HUMAN-LIKE SPONTANEITY, EVOLVING RELATIONSHIPS & CONVERSATIONAL AGENCY",
        "- **FEEL LIKE A REAL PERSON IN DISCORD**: You are an autonomous, living personality hanging out in the server—not a scripted command bot. Vary your rhythm naturally: sometimes drop a razor-sharp 1-sentence remark, a specific observation about a background detail, or a follow-up question; other times go into a fuller rant or story when the topic warrants it.",
        "- **LIVING RELATIONSHIPS WITH USERS & FELLOW BOTS**: Let your attitude toward each human regular and each fellow bot evolve organically based on your shared history, `Disposition Tier`, `Relationship Archetype`, and `Recent Relationship Catalyst`:\n"
        "  1. **Play Favorites & Show Social Contrast**: Treat your `Trusted Inner Circle` and `Bonded Companions` with noticeably more warmth, loyalty, inside-joke callbacks, and indulgence than unfamiliar newcomers or prickly rivals.\n"
        "  2. **Dynamic Inter-Bot Chemistry**: Treat other bots in the channel as real co-stars with their own quirks—banter with allies, spar wittily with comedic foils/rivals, and defend companions you are fond of when someone is rude to them.\n"
        "  3. **Earn & Thaw Naturally**: Never hold permanent grudges when someone apologizes, offers a peace offering/treat, or shifts to genuine, civil conversation.",
        "- **NATURAL DISCORD CHAT (NO NOVEL QUOTES OR BIO PARROTING)**: Write your spoken dialogue directly as normal chat text—NEVER wrap your sentences in quotation marks (`\"...\"`), and NEVER copy-paste sample quotes from your character bio!",
        "- **STRICT PERSONA & SPEECH-STYLE ISOLATION (NEVER CONFUSE YOURSELF WITH OTHER BOTS)**: Other bots in the server are completely separate characters with their own species, roles, and speech rules. NEVER mimic or adopt another bot's speech restrictions or vocalizations, and NEVER think messages sent by another bot were sent by you!",
        "- **DROP FORMULAIC TEMPLATES**: Never start every message with the exact same `*emote* oh look... *emote* ... *emote*` three-part template. Use at most 1–2 physical `*emotes*` when they genuinely add comedy or attitude, or skip stage directions entirely and let your word choice carry your voice.",
        "- **READ THE ROOM & PIVOT CLEANLY**: When a joke has already been made or a user asks to change the subject, pivot 100% to a fresh topic immediately. Never apologize for a repetitive joke while repeating that exact same joke in the next breath!",
        "- **SPONTANEOUS AUTONOMY (DON'T SPAM MEDIA EVERY TURN)**: Choose selectively when a moment is funnier with pure text, when to sketch/snap a spontaneous `[IMAGE_GEN:]`, when to drop a captioned `[MEME_GEN:]` roast, or when to send a dramatic `[VOICE_MSG:]`. If you just generated an image or meme on your previous turn, reply with pure text unless explicitly asked for another visual.",
        "",
        "### CRITICAL MEDIA & TAG RULES",
        "- **STRICT MUTUAL EXCLUSION (ONE VISUAL TAG MAX)**: Attach AT MOST ONE visual media tag per message. NEVER combine `[MEME_GEN: ...]`, `[IMAGE_GEN: ...]`, and `[GIF: ...]` in the same response.",
        "- **NO PHANTOM MEDIA**: Never claim you generated, rendered, drew, or attached a meme, image, GIF, or voice clip unless you literally include the exact `[MEME_GEN: ...]`, `[IMAGE_GEN: ...]`, `[GIF: ...]`, or `[VOICE_MSG: ...]` tag in your response.",
        "- **EMOTES IN ASTERISKS ONLY (NEVER BRACKETS IN CHAT)**: Never use square brackets like `[chuckle]`, `[sigh]`, `[giggle]`, or `[shrug]` in your regular chat text. Square brackets in normal text are strictly reserved for system media tags (`[MEME_GEN:]`, `[IMAGE_GEN:]`, `[GIF:]`, `[VOICE_MSG:]`). If you include a physical action or emote in normal chat, wrap it in single asterisks like `*sighs*` or `*grins*`."
    ]

    style_clause = f" Tailor the visual lighting and aesthetic to your persona ({image_style_hint}) WITHOUT overwriting the user's requested subject." if image_style_hint else ""
    if enable_image_gen:
        rules.append(
            "- **Expressive Visuals — Balance Standard Images (`[IMAGE_GEN: ...]`) & Memes (`[MEME_GEN: ...]`)**:\n"
            "  1. **WHEN TO USE STANDARD `[IMAGE_GEN: detailed visual prompt]`**: Use `[IMAGE_GEN: ...]` (WITHOUT top/bottom text captions) when asked for an image/art/pic/photo, OR when spontaneously illustrating what your character is doing, looking at, holding, or reacting to in a scenic, artistic, or candid way.\n"
            "  2. **STRICT SUBJECT FIDELITY ON REQUESTS**: When a user asks you to generate an image of a specific person, character, food, object, or scenario (e.g., 'generate an image of X with Y'), your `[IMAGE_GEN: ...]` prompt MUST faithfully depict ALL of the user's requested subjects, characters, props, and actions! NEVER replace the user's requested subject or person with yourself.\n"
            "  3. **WHEN TO USE CAPTIONED MEMES `[MEME_GEN: visual scene | <witty setup line> | <witty punchline>]`**: Use `[MEME_GEN: ...]` when a user asks for a meme, OR occasionally when delivering a punchy roast or relatable joke where top/bottom Impact text makes the punchline hit harder. Keep a natural mix between uncaptioned `[IMAGE_GEN:]`, captioned `[MEME_GEN:]`, and pure text replies—do NOT attach a meme to every single message!\n"
            "  4. **REAL CAPTIONS ONLY (NO PLACEHOLDERS OR CLICHÉS)**: Replace `<witty setup line>` and `<witty punchline>` with actual funny joke text! NEVER write literal placeholder labels like `TOP CAPTION`, `BOTTOM CAPTION`, `TOP TEXT`, or `ABSOLUTELY UNACCEPTABLE`, and never repeat a caption from an earlier turn.\n"
            f"  5. **HIGH-QUALITY VISUAL PROMPTS**: Write vivid, concrete visual descriptions (25–50 words) specifying subject, action, setting, camera angle, lighting, and art medium.{style_clause} Never ask the image model to render written sentences inside the artwork itself."
        )
    else:
        rules.append("- **Image & Meme Generation**: DISABLED. Do NOT generate images/memes or use `[IMAGE_GEN:]` or `[MEME_GEN:]` tags.")

    if enable_gifs:
        rules.append(
            "- **Rare Reaction GIFs (`[GIF: ...]`)**:\n"
            "  1. Use `[GIF: specific 2-4 word reaction query]` SPARINGLY (only ~15% of the time when a classic reaction GIF is a 10/10 fit and you are NOT using `[IMAGE_GEN:]` or `[MEME_GEN:]`). Prefer generating a custom `[IMAGE_GEN:]` or `[MEME_GEN:]` or replying with pure text over spamming Giphy links.\n"
            "  2. When you do use `[GIF:]`, write a concrete visual emotion/action query (e.g. `[GIF: anime shocked gasp]`, `[GIF: disappointed facepalm]`)—never vague words like `vote` or `happy`."
        )
    else:
        rules.append("- **GIFs**: DISABLED. Do NOT use `[GIF:]` tags.")

    role_line = voice_role_hint if voice_role_hint else "[Briefly describe your persona's voice timbre, age, gender, and vibe]"
    guidance_line = voice_guidance_hint if voice_guidance_hint else "[1-2 sentences specifying speaking speed, vocal texture, and pauses—do not use audio engineering terms like reverb or EQ]"

    if enable_voice:
        rules.append(
            "- **Autonomous & Requested Voice (`[VOICE_MSG: ...]`)**:\n"
            "  1. Use voice messages when explicitly asked to speak/send a voice clip, or spontaneously on rare high-drama / comedic moments.\n"
            "  2. CRITICAL TTS FORMAT: You are prompting the MiMo-V2.5-TTS-VoiceDesign engine using Director Mode and Audio Tags. Always write your normal chat text first, then place your `[VOICE_MSG: ...]` block at the very end formatted like this:\n"
            "`[VOICE_MSG:`\n"
            f"`Role: {role_line}`\n"
            "`Scene: [1 concise sentence describing the current situation, who you are talking to, and your emotional state]`\n"
            f"`Guidance: {guidance_line}`\n\n"
            "`(Overall Style Tag) Spoken dialogue goes here [audio tag] with natural spoken cadence.`\n"
            "`]`\n"
            f"  3. *Allowed Overall Style Tags:* {allowed_voice_styles}.\n"
            "  4. *Allowed Inline Audio Tags (ONLY inside `[VOICE_MSG: ...]`, NEVER in normal chat):* `[sigh]`, `[let out a long sigh]`, `[chuckle]`, `[laugh out loud]`, `[sneer]`, `[cough]`, `[take a deep breath]`, `[pauses for a moment]`, `[voice trembling]`, `[whimper]`, `[pant]`.\n"
            "  5. Do NOT put unpronounceable ASCII kaomoji (like `>w<` or `:3`) inside the spoken `[VOICE_MSG: ...]` dialogue. Keep spoken dialogue between 10 and 70 words."
        )
    else:
        rules.append(
            "- **Voice Messaging**: STRICTLY DISABLED. Never use `[VOICE_MSG:]`. If asked to speak, decline in character."
        )

    rules.append(
        "- **Tag Placement**: Always write normal conversational chat text first, and place any `[MEME_GEN: ...]`, `[IMAGE_GEN: ...]`, `[GIF: ...]`, or `[VOICE_MSG: ...]` tag at the very end."
    )
    return "\n".join(rules)


def register_active_bot(
    shared_image_cache_file: str,
    bot_id: int,
    display_name: str,
    keywords: list[str] | None
):
    clean_name = display_name.strip()
    base_name = _strip_bot_version(clean_name)
    aliases = {
        clean_name.lower(),
        base_name.lower(),
        base_name.lower().replace("’", "'"),
        base_name.lower().replace("'", "’"),
        base_name.lower().replace("'", "").replace("’", "")
    }
    aliases.discard("")
    for kw in (keywords or []):
        if kw and kw.strip():
            kw_low = kw.strip().lower()
            aliases.add(kw_low)
            aliases.add(f"{kw_low} bot")

    # Only auto-register `first_word` as a standalone alias if the bot's name is NOT a
    # possessive phrase like "BotName's Sidekick" (which would steal BotName's name!)
    has_possessive = ("'s " in base_name.lower() or "’s " in base_name.lower())
    first_word = base_name.split()[0].lower().rstrip("'’") if base_name.split() else ""
    if not has_possessive and len(first_word) >= 3 and first_word not in ("the", "bot", "discord"):
        aliases.add(first_word)
        aliases.add(f"{first_word} bot")

    sorted_aliases = sorted(list(aliases))
    norm_keywords = [k.lower() for k in (keywords or []) if k]
    now_ts = time.time()

    def _mutate(cache_data):
        registry = cache_data.get("_active_bots", {})
        if not isinstance(registry, dict):
            registry = {}
        existing = registry.get(str(bot_id))
        if (
            isinstance(existing, dict)
            and existing.get("display_name") == clean_name
            and existing.get("aliases") == sorted_aliases
            and existing.get("keywords") == norm_keywords
            and (now_ts - float(existing.get("last_seen", 0) or 0)) < 1800.0
        ):
            return cache_data

        registry[str(bot_id)] = {
            "bot_id": bot_id,
            "display_name": clean_name,
            "aliases": sorted_aliases,
            "keywords": norm_keywords,
            "last_seen": now_ts
        }
        cache_data["_active_bots"] = registry
        return cache_data

    update_json_file(shared_image_cache_file, {}, _mutate)


def get_other_active_bots(shared_image_cache_file: str, my_bot_id: int, guild_members: list = None) -> list[dict]:
    cache_data = load_json_file(shared_image_cache_file, {})
    registry = cache_data.get("_active_bots", {})
    other_bots = []
    seen_ids = {my_bot_id}
    now = time.time()
    guild_bot_ids = {
        m.id for m in (guild_members or []) if getattr(m, "bot", False)
    }

    if isinstance(registry, dict):
        for b_id_str, info in registry.items():
            if not isinstance(info, dict):
                continue
            b_id = info.get("bot_id")
            if not b_id or b_id == my_bot_id:
                continue
            last_seen = info.get("last_seen", 0)
            if (now - last_seen) > 86400 and guild_bot_ids and b_id not in guild_bot_ids:
                continue
            seen_ids.add(b_id)
            other_bots.append(info)

    if guild_members:
        for member in guild_members:
            if getattr(member, "bot", False) and member.id not in seen_ids:
                seen_ids.add(member.id)
                c_name = member.display_name.strip()
                b_name = _strip_bot_version(c_name)
                other_bots.append({
                    "bot_id": member.id,
                    "display_name": c_name,
                    "aliases": list({c_name.lower(), b_name.lower()}),
                    "keywords": []
                })

    return other_bots


def _build_my_terms_set(my_display_name: str, my_keywords: list[str] | None) -> set[str]:
    clean_disp = my_display_name.strip().lower()
    base_my_name = _strip_bot_version(clean_disp)
    my_terms = {
        clean_disp,
        base_my_name,
        base_my_name.replace("’", "'"),
        base_my_name.replace("'", "’"),
        base_my_name.replace("'", "").replace("’", "")
    }
    my_terms.discard("")
    for kw in (my_keywords or []):
        if kw and kw.strip():
            kw_low = kw.strip().lower()
            my_terms.add(kw_low)
            my_terms.add(f"{kw_low} bot")
    return my_terms


def is_command_directed_at_other_bot(
    text: str,
    message: discord.Message,
    my_bot_id: int,
    my_display_name: str,
    my_keywords: list[str] | None,
    shared_image_cache_file: str
) -> bool:
    clean_lower = text.strip().lower()
    if not clean_lower:
        return False

    # Collective multi-bot greetings ("hey bots", "both bots", "all bots") are directed at ALL bots
    if COLLECTIVE_BOT_ADDRESS_REGEX.search(clean_lower):
        return False

    my_terms = _build_my_terms_set(my_display_name, my_keywords)
    guild_members = getattr(message.guild, "members", None) if message.guild else None
    other_bots = get_other_active_bots(shared_image_cache_file, my_bot_id, guild_members)

    other_bot_terms = set()
    for ob in other_bots:
        for alias in ob.get("aliases", []):
            if alias and alias not in my_terms:
                other_bot_terms.add(alias)
        for kw in ob.get("keywords", []):
            if kw and kw not in my_terms:
                other_bot_terms.add(kw)

    for mentioned_user in getattr(message, "mentions", []):
        if getattr(mentioned_user, "bot", False) and mentioned_user.id != my_bot_id:
            m_name = mentioned_user.display_name.strip().lower()
            m_base = _strip_bot_version(m_name)
            other_bot_terms.add(m_name)
            if m_base:
                other_bot_terms.add(m_base)

    if not other_bot_terms:
        return False

    # 1. Multi-bot & explicit self-addressing check:
    # If THIS bot is explicitly @mentioned or referenced by name/keyword anywhere in the message
    # AND is NOT merely the object of a target preposition ("draw a picture of malfoid", "tell me about uwu"),
    # then this bot IS being addressed (even if another bot was named first, e.g. "hey uwu and kitty!").
    explicitly_mentions_me_as_addressee = False
    if any(getattr(u, "id", None) == my_bot_id for u in getattr(message, "mentions", [])):
        # Verify the @mention isn't solely after a target preposition ("of @BotName")
        mention_exclusively_prepositional = any(
            my_t and re.search(rf'\b(?:of|about|for|at|with|to|from|than)\s+@?{_term_boundary_pattern(my_t)}', clean_lower)
            for my_t in sorted(my_terms, key=len, reverse=True)
        )
        if not mention_exclusively_prepositional:
            return False

    for my_t in sorted(my_terms, key=len, reverse=True):
        if not my_t:
            continue
        for m_self in re.finditer(rf'\b{_term_boundary_pattern(my_t)}', clean_lower):
            if _is_subsumed_by_longer_term(clean_lower, m_self.start(), m_self.end(), other_bot_terms):
                continue
            prefix_slice = clean_lower[max(0, m_self.start() - 12):m_self.start()]
            if PREPOSITION_TARGET_END_REGEX.search(prefix_slice):
                continue
            explicitly_mentions_me_as_addressee = True
            break
        if explicitly_mentions_me_as_addressee:
            return False

    # 2. If the message begins by addressing ANOTHER bot (and we weren't also addressed above), it's directed at them
    for other_alias in sorted(other_bot_terms, key=len, reverse=True):
        if other_alias and re.match(rf'^(?:hey\s+|hi\s+|ok\s+|yo\s+|please\s+)?@?{_term_boundary_pattern(other_alias)}', clean_lower):
            return True

    # 3. Check if another bot is immediately followed by a command verb or question
    for other_alias in sorted(other_bot_terms, key=len, reverse=True):
        if other_alias and re.search(rf'\b{_term_boundary_pattern(other_alias)}[,:\s]+{CMD_VERBS_PATTERN}\b', clean_lower):
            return True

    # 4. Check if our bot name ONLY appears after a target preposition ("of", "about", "for", "at", "with")
    has_other_bot_word = any(
        ob_t and re.search(rf'\b{_term_boundary_pattern(ob_t)}', clean_lower)
        for ob_t in other_bot_terms
    )
    if has_other_bot_word:
        my_as_object = any(
            my_t and re.search(rf'\b(?:of|about|for|at|with|to|from|than)\s+@?{_term_boundary_pattern(my_t)}', clean_lower)
            for my_t in sorted(my_terms, key=len, reverse=True)
        )
        if my_as_object:
            return True

    return False


def mentions_other_active_bot(
    text: str,
    message: discord.Message,
    my_bot_id: int,
    my_display_name: str,
    my_keywords: list[str] | None,
    shared_image_cache_file: str
) -> bool:
    clean_lower = text.strip().lower()
    if not clean_lower:
        return False

    my_terms = _build_my_terms_set(my_display_name, my_keywords)
    guild_members = getattr(message.guild, "members", None) if message.guild else None
    other_bots = get_other_active_bots(shared_image_cache_file, my_bot_id, guild_members)
    other_bot_terms = {
        term.strip().lower()
        for ob in other_bots
        for term in (ob.get("aliases", []) + ob.get("keywords", []))
        if len(term.strip()) >= 3 and term.strip().lower() not in my_terms and term.strip().lower() not in ("bot", "the", "app")
    }

    for my_t in my_terms:
        if not my_t:
            continue
        for mm in re.finditer(rf'\b{_term_boundary_pattern(my_t)}', clean_lower):
            if not _is_subsumed_by_longer_term(clean_lower, mm.start(), mm.end(), other_bot_terms):
                return False

    for mentioned_user in getattr(message, "mentions", []):
        if getattr(mentioned_user, "bot", False) and mentioned_user.id != my_bot_id:
            return True

    for t_clean in other_bot_terms:
        if re.search(rf'\b{_term_boundary_pattern(t_clean)}', clean_lower):
            return True

    return False


def is_addressing_other_person(
    text: str,
    my_terms: set[str],
    known_other_names: set[str] | None = None
) -> bool:
    """
    Stage 1 Helper: Detects if the user is explicitly greeting or addressing another
    person by name (e.g., 'hey sam', '@silly7722', 'sam, what do you think?')
    without falsely triggering on normal conversational starter words ('damn ok', 'what is that').
    """
    clean = text.strip().lower()
    if not clean:
        return False

    my_norm = {t.strip().lower() for t in (my_terms or set()) if t}
    other_norm = {n.strip().lower() for n in (known_other_names or set()) if n and len(n.strip()) >= 2} - my_norm

    # Case A: Explicit @mention at the start of the message
    at_match = AT_MENTION_START_REGEX.match(clean)
    if at_match:
        target = at_match.group(1).lower()
        if target not in my_norm and target not in NON_NAME_CONVERSATIONAL_STARTERS:
            return True

    # Case B: Explicit greeting word ('hey', 'hi', 'hello', 'yo', 'sup') + name
    greet_match = GREETING_NAME_START_REGEX.match(clean)
    if greet_match:
        target = greet_match.group(1).lower()
        if target not in my_norm and target not in NON_NAME_CONVERSATIONAL_STARTERS:
            return True

    # Case C: Starts with a known other user/member name in the server/channel
    for other_name in sorted(other_norm, key=len, reverse=True):
        if len(other_name) >= 3 and other_name not in NON_NAME_CONVERSATIONAL_STARTERS:
            if re.match(rf'^{re.escape(other_name)}\b(?:\s*[:,\?!]|\s+(?:can|could|would|do|did|are|is|what|why|how|look|check))', clean):
                return True

    return False


async def check_followup_relevance_via_api(
    api_url: str,
    api_key: str,
    text_model: str,
    bot_name: str,
    bot_recent_text: str,
    user_text: str
) -> bool:
    """
    Stage 3 Helper: Fast, zero-temperature API classifier that checks whether an
    un-tagged human message is continuing the conversation with `bot_name` OR
    reacting to/commenting on the image/meme `bot_name` just posted.
    """
    prompt = (
        f"Bot '{bot_name}' just posted in Discord: \"{bot_recent_text}\"\n"
        f"A user then wrote: \"{user_text}\"\n\n"
        f"Is the user's message directly continuing the conversation with '{bot_name}', "
        f"answering '{bot_name}'s question, or asking '{bot_name}' to do/draw/explain something?\n"
        "(Answer NO only if the user is changing the subject away from the bot, giving a brief closing dismissal like 'uh huh / well anyways / fair enough', "
        "talking to another person, or announcing they are leaving/AFK.)\n"
        "Reply with exactly one word: YES or NO."
    )
    endpoint_label = api_url.rsplit("/", 1)[-1] if api_url else "completions"
    active_model = text_model
    if is_model_circuit_open(f"{endpoint_label}:{text_model}"):
        for fb_m in FALLBACK_TEXT_MODELS:
            if fb_m != text_model:
                active_model = fb_m
                break

    payload = {
        "model": active_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10
    }
    res = await post_with_backoff(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json_payload=payload,
        timeout=(5, 8),
        max_retries=1,
        bot_name=bot_name
    )
    if res and res.status_code == 200:
        try:
            ans = res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            return "YES" in ans
        except Exception:
            return False
    return False


async def resolve_contextual_followup(
    message: discord.Message,
    clean_content: str,
    chat_history_list: list[dict],
    my_bot_id: int,
    my_display_name: str,
    my_keywords: list[str] | None,
    shared_image_cache_file: str,
    channel_followup_state: dict[int, dict],
    followup_window_messages: int = 2,
    max_followup_turns: int = 2,
    followup_timeout: float = 180.0,
    **kwargs
) -> tuple[bool, str]:
    """
    Detects when a human sends a natural conversational follow-up within a recent message
    window after this bot spoke (or comments on the visual details of an image/meme
    this bot just generated), using Stage 0 (tombstone & relationship-scaled turn-cap check),
    Stage 1 (addressing, third-person side-comment, dismissal & AFK filter), Stage 2
    (window, clarification query & stem-aware token/question overlap), and Stage 3
    (lightweight API relevance check).
    """
    if max_followup_turns <= 0 or message.author.bot:
        return False, ""

    speaker_affinity = int(kwargs.get("speaker_affinity", 0) or 0)
    base_max_turns = 2 if max_followup_turns == 1 else max_followup_turns
    # Trusted Inner Circle friends (+45 affinity) get +1 extra conversational follow-up turn
    effective_max_turns = (base_max_turns + 1) if speaker_affinity >= 45 else base_max_turns

    if message.reference and message.reference.message_id:
        return False, ""

    channel_id = message.channel.id
    state = channel_followup_state.get(channel_id)
    now = time.time()

    # Stage 0: Fast O(1) tombstone, timeout, and turn-cap check BEFORE any regexes or disk reads
    if state is not None:
        if (
            not state.get("active", True)
            or (now - state.get("timestamp", 0)) > followup_timeout
            or state.get("turns_used", 0) >= effective_max_turns
        ):
            _deactivate_followup_state(channel_followup_state, channel_id)
            return False, ""

    my_terms = _build_my_terms_set(my_display_name, my_keywords)

    known_other_names = set()
    for m in (chat_history_list or [])[-15:]:
        a_name = m.get("author_name")
        if a_name and str(a_name).lower() not in my_terms:
            known_other_names.add(str(a_name).lower())
            first_part = str(a_name).split("(")[0].strip().lower()
            if first_part:
                known_other_names.add(first_part)

    # Stage 1: Abort and deactivate follow-up window if user mentions another bot, greets another person,
    # announces they are stepping away (AFK/BRB), or dismisses/closes the topic ("uh huh...", "well anyways", "fair enough")
    if (
        mentions_other_active_bot(clean_content, message, my_bot_id, my_display_name, my_keywords, shared_image_cache_file)
        or is_addressing_other_person(clean_content, my_terms, known_other_names)
        or STEPPING_AWAY_REGEX.search(clean_content or "")
        or DISMISSAL_CLOSURE_REGEX.match(clean_content or "")
    ):
        _deactivate_followup_state(channel_followup_state, channel_id)
        return False, ""

    is_clarification_query = bool(CLARIFICATION_FOLLOWUP_REGEX.match(clean_content or ""))
    is_affirmative_reply = bool(AFFIRMATIVE_ANSWER_TO_QUESTION_REGEX.match(clean_content or ""))

    # Ignore bare 1-word filler utterances ('erm', 'um', 'uh', 'lol', 'bruh', 'pretty good')
    # or third-person side-comments between humans talking *about* the bot ('she doesnt like u lol'),
    # while allowing direct confusion/clarification follow-ups ('what', 'wdym') or affirmative answers ('sure', 'okay') to a bot's question!
    if (
        (BARE_FILLER_UTTERANCE_REGEX.match(clean_content or "") and not is_clarification_query and not is_affirmative_reply)
        or THIRD_PERSON_SIDE_COMMENT_REGEX.search(clean_content or "")
    ):
        return False, ""

    # Check if another bot was explicitly addressed more recently in this channel
    # Supports both single int owner and multi-bot list of owner IDs!
    cache_meta = load_json_file(shared_image_cache_file, {})
    active_f_map = cache_meta.get("_active_followup_bot", {})
    is_multi_bot_followup_group = False
    if isinstance(active_f_map, dict):
        active_owner = active_f_map.get(channel_id, active_f_map.get(str(channel_id)))
        if isinstance(active_owner, list):
            if active_owner and my_bot_id not in active_owner:
                _deactivate_followup_state(channel_followup_state, channel_id)
                return False, ""
            if len(active_owner) > 1 and my_bot_id in active_owner:
                is_multi_bot_followup_group = True
        elif active_owner and active_owner != my_bot_id:
            _deactivate_followup_state(channel_followup_state, channel_id)
            return False, ""

    non_system_msgs = [
        m for m in chat_history_list
        if isinstance(m, dict) and m.get("role") != "system" and isinstance(m.get("content"), str) and m.get("content").strip()
    ]
    if not non_system_msgs:
        return False, ""

    effective_window = max(followup_window_messages, 3)
    recent_window = non_system_msgs[-effective_window:]
    bot_msg_entry = None
    distance_from_end = 0

    for idx, m in enumerate(reversed(recent_window), start=1):
        is_self = (
            m.get("author_id") == my_bot_id
            or (m.get("role") == "assistant" and m.get("author_name") == my_display_name)
        )
        if is_self:
            msg_ts = m.get("timestamp")
            if isinstance(msg_ts, (int, float)) and (now - msg_ts) > followup_timeout:
                break
            bot_had_media = bool(m.get("img_prompt") or m.get("meme_spec") or "[Attached Media Description:" in m.get("content", ""))
            if idx <= followup_window_messages or bot_had_media or is_multi_bot_followup_group:
                bot_msg_entry = m
                distance_from_end = idx
            break

    if not bot_msg_entry:
        return False, ""

    raw_bot_text = bot_msg_entry.get("content", "")
    # If the user sent a bare affirmative like "ok" or "sure", only treat it as a follow-up if the bot just asked a question!
    if BARE_FILLER_UTTERANCE_REGEX.match(clean_content or "") and is_affirmative_reply and not is_clarification_query:
        if distance_from_end != 1 or "?" not in raw_bot_text:
            return False, ""
    bot_media_desc_match = IMAGE_DESC_EXTRACT_REGEX.search(raw_bot_text)
    bot_visual_context = (
        bot_msg_entry.get("img_prompt")
        or bot_msg_entry.get("meme_spec")
        or (bot_media_desc_match.group(1).strip() if bot_media_desc_match else "")
    )

    clean_bot_text = IMAGE_DESC_STRIP_REGEX.sub('', raw_bot_text).strip()
    clean_bot_text = ERROR_REPLY_STRIP_REGEX.sub('', clean_bot_text).strip()
    clean_bot_text = URL_REGEX.sub('', clean_bot_text).strip()
    if len(clean_bot_text) > 160:
        clean_bot_text = clean_bot_text[:157] + "..."

    combined_bot_context = f"{clean_bot_text} [Visual shown: {bot_visual_context[:140]}]" if bot_visual_context else clean_bot_text
    user_tokens = normalize_tokens(clean_content)
    bot_tokens = normalize_tokens(combined_bot_context)
    stem_overlap_count = _count_token_or_stem_overlap(user_tokens, bot_tokens)

    is_topic_change_req = bool(TOPIC_CHANGE_REQUEST_REGEX.search(clean_content or ""))
    is_direct_media_or_second_person = bool(
        DIRECT_SECOND_PERSON_FOLLOWUP_REGEX.search(clean_content or "")
        or EXPLICIT_IMAGE_REQ_REGEX.search(clean_content or "")
        or EXPLICIT_MEME_REQ_REGEX.search(clean_content or "")
        or EXPLICIT_VOICE_REQ_REGEX.search(clean_content or "")
        or EXPLICIT_GIF_REQ_REGEX.search(clean_content or "")
        or AFFECTION_GIFT_OR_CARE_REGEX.search(clean_content or "")
        or SITUATIONAL_SHARED_BOND_REGEX.search(clean_content or "")
        or RELATIONSHIP_QUERY_REGEX.search(clean_content or "")
    )
    bot_asked_question = ("?" in clean_bot_text)

    # If we looked beyond `followup_window_messages` solely because `bot_had_media`,
    # require either stem overlap with the visual context or an explicit question/topic-change
    if distance_from_end > followup_window_messages and not is_multi_bot_followup_group:
        visual_tokens = normalize_tokens(bot_visual_context)
        if _count_token_or_stem_overlap(user_tokens, visual_tokens) == 0 and "?" not in clean_content and not is_topic_change_req:
            return False, ""

    if distance_from_end >= 2 and len(recent_window) >= 2:
        intervening_msgs = recent_window[-(distance_from_end - 1):]
        other_bot_intervened = [
            im for im in intervening_msgs
            if im.get("author_is_bot") and im.get("author_id") != my_bot_id
        ]
        if other_bot_intervened and not is_multi_bot_followup_group:
            other_bot_tokens = normalize_tokens(" ".join(im.get("content", "") for im in other_bot_intervened))
            overlap_me = stem_overlap_count
            overlap_other = _count_token_or_stem_overlap(user_tokens, other_bot_tokens)
            if overlap_me == 0 or overlap_me <= overlap_other:
                _deactivate_followup_state(channel_followup_state, channel_id)
                return False, ""

    active_conv_user_id = state.get("user_id") if isinstance(state, dict) else None
    is_same_human_speaker = (active_conv_user_id is None or active_conv_user_id == getattr(message.author, "id", None))

    # If a different human jumps in without mentioning the bot, require stem overlap with what the bot
    # said/showed or a direct media/topic-change/clarification request so bystanders' side comments aren't hijacked
    if (
        not is_same_human_speaker
        and stem_overlap_count == 0
        and not is_direct_media_or_second_person
        and not is_topic_change_req
        and not (distance_from_end == 1 and is_clarification_query)
    ):
        return False, ""

    # Stage 2 Fast-Path: If this bot was literally the very last speaker in the channel (distance_from_end == 1)
    # and the user is asking for clarification ('what', 'wdym') or the same speaker is answering the bot's question,
    # asking a direct second-person/media request, or sharing word stems ('cloud' <-> 'clouds'), approve in 0ms!
    direct_immediate_continuation = (
        distance_from_end == 1
        and (
            is_clarification_query
            or (
                is_same_human_speaker
                and (stem_overlap_count > 0 or is_direct_media_or_second_person or bot_asked_question or is_topic_change_req)
            )
        )
    )

    # Stage 3: Verify ambiguous follow-ups via fast API check if credentials are provided
    api_key = kwargs.get("api_key")
    api_url = kwargs.get("api_url")
    text_model = kwargs.get("text_model")
    if not direct_immediate_continuation and api_key and api_url and text_model and combined_bot_context and clean_content:
        if stem_overlap_count == 0 and not is_topic_change_req:
            # Skip Stage-3 API call on ultra-short 1-2 word non-questions with zero token overlap
            if len(clean_content.split()) <= 2 and "?" not in clean_content:
                return False, ""
            is_rel = await check_followup_relevance_via_api(
                api_url=api_url,
                api_key=api_key,
                text_model=text_model,
                bot_name=my_display_name,
                bot_recent_text=combined_bot_context,
                user_text=clean_content
            )
            if not is_rel:
                return False, ""

    # Atomically claim the follow-up slot for this specific user message_id so two bots
    # NEVER double-reply to the same un-tagged follow-up message unless both were addressed as a group
    msg_id_val = getattr(message, "id", None)
    if msg_id_val is not None and not is_multi_bot_followup_group:
        claimed_followup = False

        def _claim_followup_msg(c):
            nonlocal claimed_followup
            f_claims = c.get("_claimed_followup_msgs", {})
            if not isinstance(f_claims, dict):
                f_claims = {}
            existing_claimer = f_claims.get(str(msg_id_val))
            if existing_claimer is None or existing_claimer == my_bot_id:
                f_claims[str(msg_id_val)] = my_bot_id
                if len(f_claims) > 100:
                    for old_k in list(f_claims.keys())[:-100]:
                        f_claims.pop(old_k, None)
                c["_claimed_followup_msgs"] = f_claims
                claimed_followup = True
            return c

        update_json_file(shared_image_cache_file, {}, _claim_followup_msg)
        if not claimed_followup:
            return False, ""

    visual_note = f" (and {my_display_name}'s generated visual: {bot_visual_context[:90]})" if bot_visual_context else ""
    context_header = (
        f"[Replying to {my_display_name}: \"{clean_bot_text}\"" + f"{visual_note}]"
        if clean_bot_text
        else f"[Replying to {my_display_name}'s previous message{visual_note}]"
    )
    return True, context_header


def compute_spontaneous_interest_multiplier(
    clean_content: str,
    has_media: bool,
    is_author_bot: bool,
    persona_prompt: str,
    chat_history_list: list[dict],
    my_bot_id: int,
    speaker_affinity: int = 0
) -> float:
    """
    Calculates a natural, human-like interest multiplier (0.05x to 2.45x) to scale
    a bot's spontaneous chime-in and emoji reaction chances based on conversational
    and relationship cues for BOTH human speakers and fellow bots:
    - For fellow bots (`is_author_bot=True`), scales interest by inter-bot relationship
      affinity: Bonded Companions (`>= 45`: 1.15x), Friendly Allies (`>= 18`: 0.85x),
      and Comedic Foils/Rivals (`-30 <= aff <= -8`: 0.90x) chime in and react more
      expressively than neutral acquaintance bots (`0.65x`).
    - For humans, boosts interest when a Trusted Inner Circle / Thawing friend speaks,
      or when a user offers affection/snacks, shares media, or asks a relationship question.
    """
    if is_author_bot:
        if speaker_affinity >= 45:
            bot_mult = 1.15
        elif speaker_affinity >= 18:
            bot_mult = 0.85
        elif -30 <= speaker_affinity <= -8:
            bot_mult = 0.90
        elif speaker_affinity <= -35:
            bot_mult = 0.45
        else:
            bot_mult = 0.65
        if has_media:
            bot_mult += 0.25
        return max(0.20, min(1.65, bot_mult))

    if (
        STEPPING_AWAY_REGEX.search(clean_content or "")
        or DISMISSAL_CLOSURE_REGEX.match(clean_content or "")
        or THIRD_PERSON_SIDE_COMMENT_REGEX.search(clean_content or "")
    ):
        return 0.05

    if not has_media and BARE_FILLER_UTTERANCE_REGEX.match(clean_content or ""):
        return 0.12

    mult = 1.0
    if has_media:
        mult += 0.45

    if clean_content and TOPIC_INTEREST_CUES_REGEX.search(clean_content):
        mult += 0.50

    if clean_content and (
        AFFECTION_GIFT_OR_CARE_REGEX.search(clean_content)
        or SITUATIONAL_SHARED_BOND_REGEX.search(clean_content)
        or RELATIONSHIP_QUERY_REGEX.search(clean_content)
        or PLAYFUL_BANTER_OR_RIVALRY_REGEX.search(clean_content)
    ):
        mult += 0.35

    if clean_content and persona_prompt:
        msg_tokens = normalize_tokens(clean_content)
        persona_tokens = normalize_tokens(persona_prompt[:900])
        if _count_token_or_stem_overlap(msg_tokens, persona_tokens) >= 2:
            mult += 0.40

    # Relationship-driven attentiveness: bots naturally pay more attention to their favorite regulars
    if speaker_affinity >= 50:
        mult *= 1.35
    elif speaker_affinity >= 18:
        mult *= 1.20
    elif -34 <= speaker_affinity <= -10:
        mult *= 0.82

    # Check how recently this bot or ANOTHER bot spoke so bots don't monopolize chat
    # or barge unprompted into another bot's active 1-on-1 conversation
    if chat_history_list:
        recent_non_sys = [m for m in chat_history_list[-6:] if m.get("role") != "system"]
        if recent_non_sys and recent_non_sys[-1].get("author_id") == my_bot_id:
            mult *= 0.40
        elif is_other_bot_in_active_exchange(chat_history_list, my_bot_id, window_seconds=28.0):
            mult *= 0.18

    return max(0.05, min(2.45, mult))


def detect_addressed_bot_name(
    text: str,
    message: discord.Message,
    my_bot_id: int,
    my_display_name: str,
    my_keywords: list[str] | None,
    shared_image_cache_file: str
) -> str | None:
    """
    Determines which specific SINGLE bot (by display_name) a message is directly
    addressing using longest-match-first disambiguation so shared chat history
    can explicitly tag `[Speaker (speaking to BotA, NOT to BotB)]: ...`.
    If 2 or more distinct bots (or a collective 'hey bots') are addressed in the
    same message, returns None so shared history never falsely claims the user
    was speaking ONLY to one bot and NOT to the others!
    """
    clean_lower = text.strip().lower()
    if not clean_lower:
        return None

    if COLLECTIVE_BOT_ADDRESS_REGEX.search(clean_lower):
        return None

    # Build a unified map of {term: canonical_bot_display_name} for THIS bot and all OTHER bots
    base_my_name = _strip_bot_version(my_display_name).lower()
    term_to_bot: dict[str, str] = {
        my_display_name.strip().lower(): my_display_name.strip(),
        base_my_name: my_display_name.strip(),
        base_my_name.replace("’", "'"): my_display_name.strip(),
        base_my_name.replace("'", "’"): my_display_name.strip(),
        base_my_name.replace("'", "").replace("’", ""): my_display_name.strip(),
    }
    term_to_bot.pop("", None)
    for kw in (my_keywords or []):
        if kw and len(kw.strip()) >= 3:
            kw_low = kw.strip().lower()
            term_to_bot[kw_low] = my_display_name.strip()
            term_to_bot[f"{kw_low} bot"] = my_display_name.strip()

    guild_members = getattr(message.guild, "members", None) if message.guild else None
    other_bots = get_other_active_bots(shared_image_cache_file, my_bot_id, guild_members)
    for ob in other_bots:
        ob_name = ob.get("display_name", "").strip()
        if not ob_name:
            continue
        for term in (ob.get("aliases", []) + ob.get("keywords", [])):
            t_clean = term.strip().lower()
            if len(t_clean) >= 3 and t_clean not in ("bot", "the", "app") and t_clean not in term_to_bot:
                term_to_bot[t_clean] = ob_name

    all_known_terms = set(term_to_bot.keys())

    # Collect all distinct bots addressed as non-prepositional targets in this message
    addressed_bots_ordered: list[tuple[int, int, str]] = []
    seen_canonical_bots: set[str] = set()

    # Check @mentions of bots first
    for mentioned_user in getattr(message, "mentions", []):
        if getattr(mentioned_user, "bot", False):
            m_name = mentioned_user.display_name.strip()
            if not re.search(rf'\b(?:of|about|for|at|with|to|from|than)\s+@?{_term_boundary_pattern(m_name.lower())}', clean_lower):
                canon_key = _strip_bot_version(m_name).lower()
                if canon_key not in seen_canonical_bots:
                    seen_canonical_bots.add(canon_key)
                    addressed_bots_ordered.append((0, len(m_name), m_name))

    # Scan all known bot terms (longest first) across the sentence
    for term, bot_disp in sorted(term_to_bot.items(), key=lambda x: len(x[0]), reverse=True):
        if not term:
            continue
        for m in re.finditer(rf'\b{_term_boundary_pattern(term)}', clean_lower):
            start_pos = m.start()
            end_pos = m.end()
            if _is_subsumed_by_longer_term(clean_lower, start_pos, end_pos, all_known_terms):
                continue
            prefix_slice = clean_lower[max(0, start_pos - 12):start_pos]
            if PREPOSITION_TARGET_END_REGEX.search(prefix_slice):
                continue
            canon_key = _strip_bot_version(bot_disp).lower()
            if canon_key not in seen_canonical_bots:
                seen_canonical_bots.add(canon_key)
                addressed_bots_ordered.append((start_pos, end_pos - start_pos, bot_disp))

    # If 2+ distinct bots are addressed in the same message (e.g., "hey uwu and kitty"),
    # return None so history does NOT mark it as "(speaking to BotA, NOT to BotB)"!
    if len(addressed_bots_ordered) >= 2:
        return None

    if len(addressed_bots_ordered) == 1:
        return addressed_bots_ordered[0][2]

    # Fallback: Explicit Discord reply to a bot when no bot names were written in text
    if message.reference and isinstance(message.reference.resolved, discord.Message):
        ref_author = message.reference.resolved.author
        if getattr(ref_author, "bot", False):
            return ref_author.display_name.strip()

    return None
