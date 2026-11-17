import re
import ast
import math
from config import (
    VOICE_START_REGEX, MIMO_AUDIO_TAG_MAP, URL_REGEX, IMAGE_GEN_REGEX,
    UNCLOSED_IMAGE_GEN_REGEX, GIF_STRIP_REGEX, TTS_KAOMOJI_STRIP_REGEX,
    BRACKET_EMOTE_REGEX, ASTERISK_EMOTE_EXTRACT_REGEX, ERROR_REPLY_STRIP_REGEX,
    IMAGE_DESC_STRIP_REGEX, MEME_GEN_REGEX, UNCLOSED_MEME_GEN_REGEX,
    LEARN_GENERAL_STRIP_REGEX, LEARN_USER_STRIP_REGEX, LEARN_STRIP_REGEX,
    MEME_PLACEHOLDER_REGEX, INTENTIONAL_NO_RESPONSE_REGEX,
    REACTION_FOOTER_STRIP_REGEX
)

# ==============================================================================
# PRE-COMPILED VOICE, TEXT, CADENCE & EMOTE PARSING REGEXES
# ==============================================================================

EMPHASIS_NON_EMOTE_WORDS = frozenset({
    "not", "never", "really", "very", "so", "actually", "literally",
    "always", "all", "you", "me", "this", "that", "what", "why", "how",
    "everything", "nothing", "everyone", "no", "yes", "now", "here"
})

VALID_MIMO_PREFIXES = (
    "inhale", "take a deep breath", "sigh", "let out a long sigh", "pant",
    "hold one's breath", "nervous", "scared", "excited", "tired", "wronged",
    "coquettish", "guilty", "shocked", "impatient", "trembling", "voice trembling",
    "pitch change", "cracked voice", "nasal voice", "breathiness", "hoarseness",
    "smile", "chuckle", "laugh out loud", "sneer", "sob", "whimper", "choke",
    "wail", "pause", "pauses for a moment", "cough", "softly", "whisper"
)

# Strips any leading `[SpeakerName ...]:` header (including parenthetical pronouns or Fellow Bot badges)
LEADING_BOT_HEADER_REGEX = re.compile(
    r'^\s*\[[^\]\n]{1,140}\]:\s*',
    re.IGNORECASE
)

LEAKED_CHAT_PREFIX_REGEX = re.compile(
    r'(?m)^\s*\[[^\]\n]{1,140}\]:\s*',
    re.IGNORECASE
)

# Detects when the LLM hallucinates a second speaker turn in the middle of its reply
HALLUCINATED_MULTI_TURN_CUTOFF_REGEX = re.compile(
    r'(?:'
    r'\n\s*\[[^\]\n]{2,140}\]:'
    r'|\s+\[[a-zA-Z0-9_\-\.\s]{2,40}\((?:speaking to|Fellow Bot)[^\]\n]+\]:'
    r'|\s+\*?[A-Z][a-zA-Z0-9_\-]{1,25}:\s+[^:\n]{2,80}?\*?\s*:\s+'
    r'|\s+:\s+(?:Oh\b|Clearly\b|Honestly\b|Wait\b|Well\b|Look\b|Please\b|No\b|Yes\b|I\b|You\b|\*[a-zA-Z])'
    r'|\s*\[?\*?\[?No response[\—\-\:][^\n\]]*\]?\*?'
    r')',
    re.IGNORECASE
)

# Strips echoed short user inputs at the very start of the bot's reply (e.g. `erm : *glances up*` or `um : *sighs*`)
ECHOED_INPUT_PREFIX_REGEX = re.compile(
    r'^\s*(?:[a-zA-Z0-9_\-]{1,20}\s*:\s*)+(?=\*|[A-Z"\'“])'
)

LEAKED_CONTEXT_TAG_REGEX = re.compile(
    r'\[(?:Replying to [^\]\n]+|Contains (?:GIF|YouTube Video)[^\]\n]*|'
    r'Reactions on this message:[^\]\n]*|'
    r'Linked Webpage:[^\]\n]*|'
    r'System (?:Visual|Context|Glitch|Time|Memory|Relationship)[^\]\n]*|'
    r'Generated Attached (?:Image|Meme|Voice Audio)[^\]\n]*|'
    r'Sent (?:GIF|Image|Meme|Audio) of[^\]\n]*|'
    r'(?:Disposition|Rapport|Affinity|Relationship(?:\s+Catalyst|\s+Dynamic|\s+Archetype)?|Social\s+Contrast|Protective\s+Loyalty)\s*:[^\]\n]*)\]\n?',
    re.IGNORECASE
)

TRAILING_META_NOTE_REGEX = re.compile(
    r'(?:\s*\((?:Note|OOC|Meta|Author[\'’]?s?\s*Note|System|Internal\s*Thought|Thought|Subtext|Disposition|Rapport|Affinity|Relationship|Catalyst|Archetype)\s*:[^\)]*\)\s*'
    r'|\s*\[(?:Note|OOC|Meta|Internal\s*Thought|Thought|Subtext|Disposition|Rapport|Affinity|Relationship|Catalyst|Archetype)\s*:[^\]]*\]\s*)+$',
    re.IGNORECASE
)

# Matches hidden inner-monologue tags ([THOUGHT: ...] or [SUBTEXT: ...]) so bots can reason before speaking
THOUGHT_START_REGEX = re.compile(r'\[\s*(?:THOUGHT|SUBTEXT|INTERNAL_THOUGHT)\s*:\s*', re.IGNORECASE)
THOUGHT_INLINE_STRIP_REGEX = re.compile(
    r'\[\s*(?:THOUGHT|SUBTEXT|INTERNAL_THOUGHT)\s*:[^\]]*(?:\]|$)',
    re.IGNORECASE
)

# Crown Jewel #6 — Persona-Signature Drift Guard: strips generic AI-assistant stock openings
GENERIC_ASSISTANT_DRIFT_REGEX = re.compile(
    r'^\s*(?:'
    r'(?:certainly|sure\s+thing|absolutely|of\s+course)[!,.]\s+'
    r'(?:here(?:[\'’]s|\s+is)\s+(?:a|the|how|what|your)\b[^.!:\n]*[:.!]|i[\'’]?d\s+be\s+happy\s+to\s+help[^.!:\n]*[:.!]|let\s+me\s+(?:help|break\s+that\s+down|explain)[^.!:\n]*[:.!])\s*'
    r'|i[\'’]?d\s+be\s+(?:more\s+than\s+)?happy\s+to\s+(?:help|explain|assist)[^.!:\n]*[:.!]\s*'
    r'|as\s+an\s+ai(?:\s+language\s+model|\s+assistant)?[,\s]+i\b[^.!?\n]*[.!?]\s*'
    r'|great\s+question!\s+(?:here(?:[\'’]s|\s+is)\s+how[^.!:\n]*[:.!]\s*)?'
    r')',
    re.IGNORECASE
)

# Matches lines formatted like novel manuscripts where all spoken segments are wrapped in quotes around *emotes*
NOVEL_QUOTED_LINE_REGEX = re.compile(
    r'^(?:\s*(?:\*[^*]+\*|"[^"\n]+"|“[^”\n]+”|[?,.;!~]+|(?:>w<|>_<|\^w\^|:3|;3|OwO|UwU|owo|uwu|>\:\())[?,.;!~]*\s*)+$'
)
STRAIGHT_QUOTED_SEGMENT_REGEX = re.compile(r'"([^"\n]+)"')
CURLY_QUOTED_SEGMENT_REGEX = re.compile(r'“([^”\n]+)”')
UNCLOSED_TRAILING_EMOTE_REGEX = re.compile(r'(?<!\*)\*([a-zA-Z][a-zA-Z0-9\s,\'’\-]{2,42})$')

PROMPT_BAKED_TEXT_STRIP_REGEX = re.compile(
    r'(?:,\s*|\s+)(?:with\s+(?:bold\s+|white\s+|impact\s+|top\s+|bottom\s+)*(?:text|caption|captions|words|lettering)\s+'
    r'(?:reading|saying|that\s+says)\s*[\'""“][^\"\'”]+[\'""”])',
    re.IGNORECASE
)

# Detects non-voice media or learning tags accidentally nested inside or after an unclosed [VOICE_MSG: ...] block
RESCUABLE_EMBEDDED_TAG_REGEX = re.compile(
    r'\[\s*(?:IMAGE_GEN|MEME_GEN|GIF|LEARN_USER|LEARN_GENERAL(?:_FACT|_KNOWLEDGE)?|LEARN(?:_SERVER|_LORE|_FACT)?)\s*[|:]\s*[^\]]+\]?',
    re.IGNORECASE | re.DOTALL
)

HTML_TAG_STRIP_REGEX = re.compile(r'<[^>]+>')
SYSTEM_NOTE_STRIP_REGEX = re.compile(r'\*?\s*\(System Note:[^\n]*?\)\s*\*?\n?', re.IGNORECASE)
LEARNED_NOTE_STRIP_REGEX = re.compile(
    r'\*?\s*\([^\n]*?\blearned (?:server fact|general fact|about)\b[^\n]*\)\s*\*?\n?',
    re.IGNORECASE
)
EMPTY_CODEBLOCK_REGEX = re.compile(r'```[a-zA-Z]*\s*```')
EMPTY_BLOCKQUOTE_REGEX = re.compile(r'(?m)^>\s*$')
STRAY_MARKDOWN_LINE_REGEX = re.compile(r'(?m)^\s*(?:\*{1,3}|_{1,3}|~{1,3})\s*$')
TRAILING_ASTERISKS_REGEX = re.compile(r'(?:\s*\*{2,}\s*)+$')
STRUCTURED_LIST_LINE_REGEX = re.compile(r'^(?:\d+\.|[\-\*]\s+|```)')
LEADING_ACTION_EMOTE_LINE_REGEX = re.compile(r'^\*[a-zA-Z][^*\n]{2,55}\*\s+\S+')
DISCORD_CUSTOM_EMOJI_REGEX = re.compile(r'<a?:[a-zA-Z0-9_]+:\d+>')
MULTI_SPACE_REGEX = re.compile(r'\s{2,}')
SPACE_BEFORE_PUNCT_REGEX = re.compile(r'\s+([,.;!?])')
INLINE_ASTERISK_ACTION_REGEX = re.compile(r'\*([^*]{1,40})\*')
INLINE_BRACKET_ACTION_REGEX = re.compile(r'\[([^\]]{1,45})\]')

SCENE_PREFIX_REGEX = re.compile(r'^(?:scene|image|visual|prompt|visual\s+scene)\s*:\s*', re.IGNORECASE)
TOP_CAPTION_PREFIX_REGEX = re.compile(
    r'^(?:short\s+)?(?:top\s*text|top\s*caption|top|caption\s*1|text\s*1|setup)\s*:\s*',
    re.IGNORECASE
)
BOTTOM_CAPTION_PREFIX_REGEX = re.compile(
    r'^(?:short\s+)?(?:bottom\s*text|bottom\s*caption|bottom\s*punchline|bottom|caption\s*2|text\s*2|punchline)\s*:\s*',
    re.IGNORECASE
)

MEME_PLACEHOLDER_SET = frozenset({
    "top text", "bottom text", "top caption", "bottom caption",
    "short top caption", "short bottom punchline", "bottom punchline",
    "short top text", "short bottom text", "caption 1", "caption 2",
    "text 1", "text 2", "setup", "punchline", "insert top text",
    "insert bottom text", "insert caption", "none", "n/a", "null",
    "empty", "no text", "no caption", "blank"
})


def sanitize_api_error_text(raw_err: str, fallback: str = "Provider busy") -> str:
    """
    Cleans raw upstream HTTP/API error messages (stripping HTML tags from Cloudflare
    502/504 pages and replacing inner parentheses) so wrapped error notices like
    `*(Voice generation error 502: ...)*` never leak raw HTML or break ERROR_REPLY_STRIP_REGEX.
    """
    if not raw_err or not isinstance(raw_err, str):
        return fallback
    cleaned = HTML_TAG_STRIP_REGEX.sub(' ', raw_err)
    cleaned = cleaned.replace("(", "[").replace(")", "]").replace("*", "")
    cleaned = MULTI_SPACE_REGEX.sub(' ', cleaned.replace("\n", " ").replace("\r", " ")).strip()
    if not cleaned or len(cleaned) < 2:
        return fallback
    return cleaned[:95].strip()


def _strip_outer_novel_quotes(line: str) -> str:
    """
    Removes novel-style dialogue quotation marks when an LLM wraps its entire
    spoken response (or every spoken segment around *emotes*) in double quotes,
    while preserving legitimate inline quotes inside normal sentences.
    """
    if not line or ('"' not in line and '“' not in line and '”' not in line):
        return line
    stripped = line.strip()
    if NOVEL_QUOTED_LINE_REGEX.match(stripped):
        dequoted = STRAIGHT_QUOTED_SEGMENT_REGEX.sub(r'\1', stripped)
        dequoted = CURLY_QUOTED_SEGMENT_REGEX.sub(r'\1', dequoted)
        return MULTI_SPACE_REGEX.sub(' ', dequoted).strip()
    return line


def _heal_unclosed_trailing_emote(line: str) -> str:
    """Closes a trailing unclosed single-asterisk emote at the end of a line (e.g. `*tilts head` -> `*tilts head*`)."""
    if not line or "*" not in line:
        return line
    single_star_count = len(re.findall(r'(?<!\*)\*(?!\*)', line))
    if single_star_count % 2 == 1:
        if UNCLOSED_TRAILING_EMOTE_REGEX.search(line):
            return f"{line.rstrip()}*"
    return line


def extract_and_strip_thought_tag(text: str) -> tuple[str | None, str]:
    """
    Extracts and strips optional `[THOUGHT: ...]` or `[SUBTEXT: ...]` hidden inner-monologue
    blocks emitted by the LLM before its spoken response, rescuing any accidentally nested
    media or learning tags (`[IMAGE_GEN:]`, `[MEME_GEN:]`, `[GIF:]`, `[LEARN*:]`).
    Returns `(thought_str_or_None, cleaned_reply_text)`.
    """
    if not text:
        return None, ""

    match = THOUGHT_START_REGEX.search(text)
    if not match:
        return None, text

    thoughts_collected = []
    working = text
    max_iters = 3
    for _ in range(max_iters):
        m = THOUGHT_START_REGEX.search(working)
        if not m:
            break
        start_idx = m.start()
        content_start = m.end()
        depth = 1
        i = content_start
        while i < len(working) and depth > 0:
            if working[i] == '[':
                depth += 1
            elif working[i] == ']':
                depth -= 1
            i += 1

        if depth == 0:
            raw_thought = working[content_start:i - 1].strip()
            working = (working[:start_idx] + " " + working[i:]).strip()
        else:
            # Unclosed [THOUGHT: ... — stop at first newline or take up to 180 chars
            nl_pos = working.find("\n", content_start)
            if nl_pos != -1:
                raw_thought = working[content_start:nl_pos].strip()
                working = (working[:start_idx] + " " + working[nl_pos + 1:]).strip()
            else:
                raw_thought = working[content_start:].strip()
                working = working[:start_idx].strip()

        if raw_thought:
            rescued = []
            for m_tag in RESCUABLE_EMBEDDED_TAG_REGEX.finditer(raw_thought):
                t_str = m_tag.group(0).strip()
                if not t_str.endswith("]"):
                    t_str += "]"
                rescued.append(t_str)
            if rescued:
                raw_thought = RESCUABLE_EMBEDDED_TAG_REGEX.sub('', raw_thought).strip()
                working = f"{working} {' '.join(rescued)}".strip()
            if raw_thought:
                thoughts_collected.append(MULTI_SPACE_REGEX.sub(' ', raw_thought))

    final_thought = " | ".join(thoughts_collected)[:240].strip() if thoughts_collected else None
    working = THOUGHT_INLINE_STRIP_REGEX.sub('', working).strip()
    return (final_thought if final_thought else None), working


def extract_and_strip_voice_msg(text: str) -> tuple[str | None, str]:
    """
    Safely extracts the full content of [VOICE_MSG: ...] even when nested
    brackets like [sigh] or [chuckle] are inside the voice script, rescues any
    trailing [IMAGE_GEN:], [MEME_GEN:], [GIF:], or [LEARN*:] tags that were
    swallowed by an unclosed [VOICE_MSG: block, and returns (voice_content, cleaned_reply_text).
    """
    if not text:
        return None, ""

    match = VOICE_START_REGEX.search(text)
    if not match:
        return None, text

    start_idx = match.start()
    content_start = match.end()
    depth = 1
    i = content_start

    while i < len(text) and depth > 0:
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
        i += 1

    if depth == 0:
        voice_content = text[content_start:i - 1].strip()
        cleaned_text = (text[:start_idx] + text[i:]).strip()
    else:
        voice_content = text[content_start:].strip()
        cleaned_text = text[:start_idx].strip()

    # Rescue any media or learning tags ([IMAGE_GEN:], [MEME_GEN:], [GIF:], [LEARN*:])
    # that the LLM accidentally placed inside or after an unclosed [VOICE_MSG: ...] block
    rescued_tags = []
    if voice_content:
        for m_tag in RESCUABLE_EMBEDDED_TAG_REGEX.finditer(voice_content):
            tag_str = m_tag.group(0).strip()
            if not tag_str.endswith("]"):
                tag_str += "]"
            rescued_tags.append(tag_str)
        if rescued_tags:
            voice_content = RESCUABLE_EMBEDDED_TAG_REGEX.sub('', voice_content).strip()
            cleaned_text = f"{cleaned_text} {' '.join(rescued_tags)}".strip()

    cleaned_text = re.sub(r'`+\s*`+', '', cleaned_text).strip()
    cleaned_text = re.sub(r'`+$', '', cleaned_text).strip()
    return (voice_content if voice_content else None), cleaned_text


def sanitize_mimo_inline_tags(speech_text: str) -> str:
    """
    Normalizes inline [action] or *action* tags inside spoken TTS text into
    valid MiMo-V2.5-TTS audio control tags, removing non-audio roleplay actions
    (like *tail flicks* or *adjusts glasses*) so the TTS actor doesn't glitch.
    """
    def _replace_tag(match):
        raw_tag = match.group(1).strip().lower()
        if raw_tag in MIMO_AUDIO_TAG_MAP:
            return f"[{MIMO_AUDIO_TAG_MAP[raw_tag]}]"
        if any(raw_tag.startswith(p) for p in VALID_MIMO_PREFIXES):
            return f"[{raw_tag}]"
        return ""

    speech_text = INLINE_ASTERISK_ACTION_REGEX.sub(r'[\1]', speech_text)
    speech_text = INLINE_BRACKET_ACTION_REGEX.sub(_replace_tag, speech_text)
    return speech_text


def parse_mimotts_voice_block(
    raw_voice_content: str,
    default_role_desc: str,
    default_style: str = "Natural"
) -> tuple[str, str]:
    """
    Separates Director Mode headers (Role, Scene, Guidance) from the spoken dialogue
    for MiMo-V2.5-TTS-VoiceDesign, cleaning out unpronounceable chat kaomoji and URLs.
    Returns (user_prompt_content, assistant_speech_content). If no pronounceable spoken
    words remain after stripping tags, `assistant_speech_content` is returned as `""`.
    """
    if not raw_voice_content or not raw_voice_content.strip():
        return default_role_desc, ""

    cleaned_raw = raw_voice_content.strip().strip('`')
    lines = [line.strip().strip('`') for line in cleaned_raw.split('\n') if line.strip().strip('`')]
    director_instructions = []
    spoken_text_lines = []
    in_header_section = True

    for line in lines:
        clean_line = re.sub(r'^[\*\-\s]+', '', line)
        clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_line)
        lower_line = clean_line.lower()

        if lower_line.startswith(("role:", "character:", "scene:", "guidance:")):
            director_instructions.append(clean_line)
            in_header_section = True
        elif in_header_section and clean_line.startswith("(") and ")" in clean_line:
            in_header_section = False
            spoken_text_lines.append(clean_line)
        elif (
            in_header_section
            and len(director_instructions) > 0
            and not clean_line.startswith("(")
            and len(spoken_text_lines) == 0
            and line.startswith("-")
        ):
            director_instructions.append(clean_line)
        else:
            in_header_section = False
            spoken_text_lines.append(clean_line)

    user_prompt_content = "\n".join(director_instructions).strip()
    assistant_speech_content = " ".join(spoken_text_lines).strip()

    if not assistant_speech_content and director_instructions:
        last_dir = director_instructions[-1]
        if not last_dir.lower().startswith(("role:", "character:", "scene:", "guidance:")):
            assistant_speech_content = last_dir
            director_instructions = director_instructions[:-1]
            user_prompt_content = "\n".join(director_instructions).strip()

    if not user_prompt_content:
        user_prompt_content = default_role_desc
    if not assistant_speech_content and not director_instructions:
        assistant_speech_content = cleaned_raw

    assistant_speech_content = URL_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = LEARN_USER_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = LEARN_GENERAL_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = LEARN_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = REACTION_FOOTER_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = LEARNED_NOTE_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = LEAKED_CONTEXT_TAG_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = THOUGHT_INLINE_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = TRAILING_META_NOTE_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = MEME_GEN_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = UNCLOSED_MEME_GEN_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = IMAGE_GEN_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = UNCLOSED_IMAGE_GEN_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = GIF_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = sanitize_mimo_inline_tags(assistant_speech_content)
    assistant_speech_content = TTS_KAOMOJI_STRIP_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = DISCORD_CUSTOM_EMOJI_REGEX.sub('', assistant_speech_content)
    assistant_speech_content = re.sub(r'[\*_`~]+', '', assistant_speech_content)
    assistant_speech_content = MULTI_SPACE_REGEX.sub(' ', assistant_speech_content).strip()

    # Check if any actual spoken words exist outside of (Style) and [audio] tags
    spoken_words_check = re.sub(r'\([^\)]*\)|\[[^\]]*\]', '', assistant_speech_content).strip()
    if not re.search(r'[a-zA-Z0-9]', spoken_words_check):
        return user_prompt_content, ""

    if assistant_speech_content and not assistant_speech_content.startswith("("):
        assistant_speech_content = f"({default_style}) {assistant_speech_content}"

    if len(assistant_speech_content) > 450:
        clipped = assistant_speech_content[:450]
        last_punct = max(clipped.rfind('.'), clipped.rfind('!'), clipped.rfind('?'))
        if last_punct > 150:
            assistant_speech_content = clipped[:last_punct + 1].strip()
        else:
            assistant_speech_content = clipped.strip()

    return user_prompt_content, assistant_speech_content


def derive_chat_text_from_voice(raw_voice_content: str) -> str:
    """
    If the LLM placed its entire response inside [VOICE_MSG: ...] and left the visible
    chat reply empty, derives a clean, natural Discord text message from the spoken lines.
    """
    _, spoken = parse_mimotts_voice_block(raw_voice_content, "", "Natural")
    if not spoken:
        return ""
    spoken = re.sub(r'^\([^\)]+\)\s*', '', spoken).strip()
    spoken = convert_bracket_emotes_to_asterisks(spoken)
    return clean_and_compact_reply(spoken)


def convert_bracket_emotes_to_asterisks(text: str) -> str:
    """
    Converts any remaining [emote] or [action] tags in normal chat text into *emote*
    after media/system tags have already been stripped.
    """
    def _replace(match):
        inner = match.group(1).strip()
        lower_inner = inner.lower()
        if (
            lower_inner.startswith((
                "http", "image_gen", "meme_gen", "voice_msg", "gif",
                "attached", "contains", "system", "replying", "addressed", "directed",
                "learn", "no response", "linked webpage", "note:", "ooc:", "meta:",
                "disposition:", "rapport:", "affinity:", "relationship:", "catalyst:",
                "archetype:", "social contrast:", "thought:", "subtext:", "internal_thought:"
            ))
            or "speaking to" in lower_inner
            or "fellow bot" in lower_inner
            or "separate character" in lower_inner
        ):
            return ""
        return f"*{inner}*"

    return BRACKET_EMOTE_REGEX.sub(_replace, text)


def extract_used_emotes(text: str) -> list[str]:
    """
    Extracts normalized *emote* phrases from a message so memory/context builders
    can prevent bots from repeating identical physical actions every turn.
    """
    if not text:
        return []
    found = []
    for m in ASTERISK_EMOTE_EXTRACT_REGEX.finditer(text):
        em = MULTI_SPACE_REGEX.sub(' ', m.group(1).strip().lower())
        words = em.split()
        if 1 <= len(words) <= 8:
            if len(words) == 1 and words[0] in EMPHASIS_NON_EMOTE_WORDS:
                continue
            if em not in found:
                found.append(em)
    return found


def _prune_excessive_inline_emotes(text: str, max_emotes: int = 2) -> str:
    """
    Prevents robotic 'stage-direction overload' (e.g., 3-4 separate *emotes* stuffed
    into a single short line) by keeping at most `max_emotes` distinct actions per turn,
    collapsing back-to-back adjacent emotes, and removing duplicate emote stems.
    """
    if not text or "*" not in text:
        return text

    # Collapse back-to-back adjacent action emotes like `*smirks* *rolls eyes*` -> `*smirks*`
    text = re.sub(
        r'(\*[a-zA-Z][^*\n]{2,45}\*)\s+\*[a-zA-Z][^*\n]{2,45}\*',
        r'\1',
        text
    )

    seen_roots = set()
    kept_count = 0

    def _filter_emote(match):
        nonlocal kept_count
        raw_inner = match.group(1).strip()
        words = re.findall(r'[a-z]+', raw_inner.lower())
        if len(words) == 1 and words[0] in EMPHASIS_NON_EMOTE_WORDS:
            return match.group(0)

        root_key = " ".join(words[:2]) if words else raw_inner.lower()
        if root_key in seen_roots or kept_count >= max_emotes:
            return ""
        seen_roots.add(root_key)
        kept_count += 1
        return f"*{raw_inner}*"

    pruned = ASTERISK_EMOTE_EXTRACT_REGEX.sub(_filter_emote, text)
    pruned = MULTI_SPACE_REGEX.sub(' ', pruned).strip()
    pruned = SPACE_BEFORE_PUNCT_REGEX.sub(r'\1', pruned)
    return pruned


def _normalize_echo_compare(text: str) -> str:
    """Normalizes text for case-insensitive, punctuation-tolerant verbatim echo detection."""
    if not text:
        return ""
    s = LEADING_BOT_HEADER_REGEX.sub('', text.strip())
    s = re.sub(r'[\s\W_]+', ' ', s.lower()).strip()
    return s


def strip_verbatim_user_echo(reply: str, recent_user_messages: list[str] | None = None) -> str:
    """
    Prevents a bot from repeating a user's recent message word-for-word (e.g. 'no, fort worth')
    or echoing the user's message as an unquoted prefix before its own emote/comment
    (e.g. 'they should AI slop remaster *giggles* only for...').
    Returns '' if the entire reply is a verbatim echo of a recent user message.
    """
    if not reply or not recent_user_messages:
        return reply
    cleaned_reply = reply.strip()
    norm_reply = _normalize_echo_compare(cleaned_reply)
    if not norm_reply:
        return ""

    for raw_u in recent_user_messages:
        if not raw_u or not isinstance(raw_u, str):
            continue
        # Strip any speaker prefix or metadata lines from the user message
        u_lines = [
            LEADING_BOT_HEADER_REGEX.sub('', ln.strip()).strip()
            for ln in raw_u.split("\n")
            if ln.strip() and not ln.strip().startswith(("[Replying to ", "[Attached Media", "[Contains ", "[Linked Webpage:", "[Reactions on"))
        ]
        u_clean = " ".join(u_lines).strip()
        if not u_clean:
            continue
        norm_u = _normalize_echo_compare(u_clean)
        if not norm_u:
            continue

        # 1. Exact full-message verbatim echo (e.g. Carzen: 'no, fort worth' -> Bot: 'no, fort worth')
        if norm_reply == norm_u and len(norm_u) >= 2:
            return ""

        # 2. Unquoted leading prefix echo (e.g. User: 'they should AI slop remaster' -> Bot: 'they should AI slop remaster *giggles* ...')
        if len(u_clean) >= 8 and cleaned_reply.lower().startswith(u_clean.lower()):
            remainder = cleaned_reply[len(u_clean):].lstrip(" ,;:-—")
            if remainder and (remainder.startswith("*") or len(remainder) >= 12):
                cleaned_reply = remainder
                norm_reply = _normalize_echo_compare(cleaned_reply)

    return cleaned_reply


def clean_and_compact_reply(reply: str, recent_user_messages: list[str] | None = None) -> str:
    """
    Cleans stray markdown artifacts, leaked context headers, leaked reaction footers,
    echoed user inputs, novel-style outer quotation marks, hallucinated multi-turn scripts,
    and self-cloned second turns, converts bracket emotes to asterisks, prunes
    excessive stage directions, and compacts spacing.
    """
    if not reply:
        return ""

    if INTENTIONAL_NO_RESPONSE_REGEX.search(reply) and len(reply.strip()) < 95:
        return ""

    reply = ERROR_REPLY_STRIP_REGEX.sub('', reply)
    reply = REACTION_FOOTER_STRIP_REGEX.sub('', reply)
    reply = LEARN_USER_STRIP_REGEX.sub('', reply)
    reply = LEARN_GENERAL_STRIP_REGEX.sub('', reply)
    reply = LEARN_STRIP_REGEX.sub('', reply)
    _, reply = extract_and_strip_thought_tag(reply)
    reply = IMAGE_DESC_STRIP_REGEX.sub('', reply)
    reply = LEAKED_CONTEXT_TAG_REGEX.sub('', reply)
    reply = TRAILING_META_NOTE_REGEX.sub('', reply)

    # 1. Strip any leading `[BotName]:` header at the start of the response
    reply = LEADING_BOT_HEADER_REGEX.sub('', reply).strip()

    # 2. Strip echoed short user input prefixes like `erm : *glances up*` or `um : *sighs*`
    reply = ECHOED_INPUT_PREFIX_REGEX.sub('', reply).strip()

    # 2b. Crown Jewel #6 — Persona-Signature Drift Guard (strips robotic assistant openings if substantive reply follows)
    drift_stripped = GENERIC_ASSISTANT_DRIFT_REGEX.sub('', reply).strip()
    if len(drift_stripped) >= 12:
        reply = drift_stripped

    # 3. Truncate any hallucinated multi-turn continuation
    cutoff_match = HALLUCINATED_MULTI_TURN_CUTOFF_REGEX.search(reply)
    if cutoff_match and cutoff_match.start() > 8:
        reply = reply[:cutoff_match.start()].strip()

    reply = LEAKED_CHAT_PREFIX_REGEX.sub('', reply)
    reply = SYSTEM_NOTE_STRIP_REGEX.sub('', reply)
    reply = LEARNED_NOTE_STRIP_REGEX.sub('', reply)

    reply = convert_bracket_emotes_to_asterisks(reply)

    reply = EMPTY_CODEBLOCK_REGEX.sub('', reply).strip()
    reply = EMPTY_BLOCKQUOTE_REGEX.sub('', reply).strip()
    reply = STRAY_MARKDOWN_LINE_REGEX.sub('', reply).strip()
    reply = TRAILING_ASTERISKS_REGEX.sub('', reply).strip()

    raw_lines = [LEAKED_CHAT_PREFIX_REGEX.sub('', line.strip()).strip() for line in reply.split('\n') if line.strip()]
    cleaned_lines = []
    for ln in raw_lines:
        ln = ECHOED_INPUT_PREFIX_REGEX.sub('', ln).strip()
        ln = _strip_outer_novel_quotes(ln)
        ln = _heal_unclosed_trailing_emote(ln)
        if ln:
            cleaned_lines.append(ln)

    if not cleaned_lines:
        return ""

    has_structured_list = any(STRUCTURED_LIST_LINE_REGEX.match(line) for line in cleaned_lines)

    # 4. Prevent "two-turn self-contradiction" (e.g. Line 1 apologizes and pivots topic,
    # and Line 2 starts with a brand-new `*emote*` repeating the old joke as a second turn)
    if not has_structured_list and len(cleaned_lines) >= 2:
        deduped_lines = [cleaned_lines[0]]
        for ln in cleaned_lines[1:]:
            if len(deduped_lines[0]) >= 25 and LEADING_ACTION_EMOTE_LINE_REGEX.match(ln):
                break
            deduped_lines.append(ln)
        cleaned_lines = deduped_lines

    combined = "\n".join(cleaned_lines).strip() if has_structured_list else (
        " ".join(cleaned_lines).strip() if (len(cleaned_lines) <= 4 and sum(len(l) for l in cleaned_lines) < 420) else "\n".join(cleaned_lines).strip()
    )

    combined = _strip_outer_novel_quotes(combined)
    combined = _prune_excessive_inline_emotes(combined, max_emotes=2)
    if recent_user_messages:
        combined = strip_verbatim_user_echo(combined, recent_user_messages)
    return combined


def _is_placeholder_caption(text: str) -> bool:
    """Returns True if a meme caption is just a literal template placeholder like 'TOP CAPTION'."""
    if not text:
        return True
    cleaned = text.strip().strip('"\'*[]<>()').strip().lower()
    cleaned = MULTI_SPACE_REGEX.sub(' ', cleaned)
    if not cleaned:
        return True
    if cleaned in MEME_PLACEHOLDER_SET:
        return True
    if MEME_PLACEHOLDER_REGEX.match(cleaned):
        return True
    return False


def parse_meme_gen_tag(raw_meme_str: str) -> tuple[str, str, str]:
    """
    Parses a [MEME_GEN: visual scene | TOP TEXT | BOTTOM TEXT] tag into
    (visual_prompt, top_text, bottom_text), gracefully handling 2-part tags,
    stripping baked-in text instructions from the visual prompt, and blocking
    any template placeholder strings like 'TOP CAPTION' or 'SHORT BOTTOM PUNCHLINE'.
    """
    if not raw_meme_str:
        return "", "", ""

    cleaned_raw = re.sub(r'\s+', ' ', raw_meme_str.strip()).rstrip(']')
    cleaned_raw = re.sub(r'^\[?\s*(?:MEME_GEN|IMAGE_GEN)\s*:\s*', '', cleaned_raw, flags=re.IGNORECASE).strip()
    # Strip any trailing nested media or learning tag if the LLM concatenated two tags
    cleaned_raw = re.split(
        r'\[\s*(?:GIF|IMAGE_GEN|VOICE_MSG|LEARN_USER|LEARN_GENERAL|LEARN)\b',
        cleaned_raw,
        flags=re.IGNORECASE
    )[0].strip()

    parts = [p.strip().strip('"\'') for p in cleaned_raw.split("|") if p.strip()]

    visual_prompt = parts[0] if len(parts) >= 1 else ""
    top_text = parts[1] if len(parts) >= 2 else ""
    bottom_text = parts[2] if len(parts) >= 3 else ""

    visual_prompt = SCENE_PREFIX_REGEX.sub('', visual_prompt).strip()
    visual_prompt = PROMPT_BAKED_TEXT_STRIP_REGEX.sub('', visual_prompt).strip()

    top_text = TOP_CAPTION_PREFIX_REGEX.sub('', top_text).strip(' "\'*[]<>')
    bottom_text = BOTTOM_CAPTION_PREFIX_REGEX.sub('', bottom_text).strip(' "\'*[]<>')

    if _is_placeholder_caption(top_text):
        top_text = ""
    if _is_placeholder_caption(bottom_text):
        bottom_text = ""

    # If the LLM only provided 1 valid caption block and it has a natural setup/punchline split
    if top_text and not bottom_text and len(top_text) > 26:
        if " / " in top_text:
            sp = top_text.split(" / ", 1)
            top_text, bottom_text = sp[0].strip(), sp[1].strip()
        elif " - " in top_text:
            sp = top_text.split(" - ", 1)
            top_text, bottom_text = sp[0].strip(), sp[1].strip()
        elif " — " in top_text:
            sp = top_text.split(" — ", 1)
            top_text, bottom_text = sp[0].strip(), sp[1].strip()
        elif ": " in top_text and len(top_text) > 38:
            sp = top_text.split(": ", 1)
            if len(sp[0]) >= 6 and len(sp[1]) >= 6:
                top_text, bottom_text = sp[0].strip(), sp[1].strip()

    if _is_placeholder_caption(top_text):
        top_text = ""
    if _is_placeholder_caption(bottom_text):
        bottom_text = ""

    if top_text and bottom_text and top_text.lower() == bottom_text.lower():
        bottom_text = ""

    return visual_prompt, top_text, bottom_text


# ==============================================================================
# 0MS DETERMINISTIC LOCAL GROUNDING MICRO-TOOLS (MATH, UNITS & PYTHON AST CHECK)
# ==============================================================================
_SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
}
_SAFE_MATH_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_FENCED_PYTHON_BLOCK_REGEX = re.compile(
    r'```(?:python|py)\s*\n([\s\S]{4,4000}?)```',
    re.IGNORECASE
)
_PERCENT_OF_REGEX = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:,\d{3})*(?:\.\d+)?)\b',
    re.IGNORECASE
)
_MATH_ATOM_PAT = r'(?:(?:sqrt|log10|log2|log|sin|cos|tan|ceil|floor|abs|round)\s*\([^\)\n]{1,35}\)|\(?\s*\-?\d+(?:,\d{3})*(?:\.\d+)?\s*\)?)'
_ARITHMETIC_EXPR_REGEX = re.compile(
    r'(?<![a-zA-Z0-9_/\\\-])'
    rf'(?:{_MATH_ATOM_PAT}\s*(?:\*\*|\^|[\+\-\*/%])\s*){{1,6}}{_MATH_ATOM_PAT}'
    rf'|(?:sqrt|log10|log2|log|sin|cos|tan|ceil|floor)\s*\([^\)\n]{{1,35}}\)',
    re.IGNORECASE
)
_DATE_OR_VERSION_SKIP_REGEX = re.compile(
    r'^\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4}$|^\d+\.\d+\.\d+$'
)
_UNIT_CONV_REGEX = re.compile(
    r'\b(\-?\d+(?:\.\d+)?)\s*(?:degrees?\s*)?(°?[fc]|fahrenheit|celsius|miles?|mi|kilometers?|km|lbs?|pounds?|kg|kilograms?)\b'
    r'(?:\s+(?:in|to|into|as)\s+(°?[fc]|fahrenheit|celsius|miles?|mi|kilometers?|km|lbs?|pounds?|kg|kilograms?))?',
    re.IGNORECASE
)


def _eval_safe_ast_node(node, depth: int = 0) -> float | int:
    if depth > 14:
        raise ValueError("Expression too deep")
    if isinstance(node, ast.Expression):
        return _eval_safe_ast_node(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > 1e18:
            raise ValueError("Operand magnitude too large")
        return node.value
    if isinstance(node, ast.Name) and node.id.lower() in _SAFE_MATH_CONSTS:
        return _SAFE_MATH_CONSTS[node.id.lower()]
    if isinstance(node, ast.UnaryOp):
        val = _eval_safe_ast_node(node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise ValueError("Unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = _eval_safe_ast_node(node.left, depth + 1)
        right = _eval_safe_ast_node(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            res = left + right
        elif isinstance(node.op, ast.Sub):
            res = left - right
        elif isinstance(node.op, ast.Mult):
            res = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            res = left / float(right)
        elif isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            res = left // right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("modulo by zero")
            res = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 1000 or (abs(left) > 1e6 and abs(right) > 12):
                raise ValueError("Exponent too large")
            res = left ** right
        else:
            raise ValueError("Unsupported binary op")
        if abs(res) > 1e24:
            raise ValueError("Result overflow")
        return res
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn_name = node.func.id.lower()
        if fn_name in _SAFE_MATH_FUNCS and 1 <= len(node.args) <= 2 and not node.keywords:
            eval_args = [_eval_safe_ast_node(a, depth + 1) for a in node.args]
            return _SAFE_MATH_FUNCS[fn_name](*eval_args)
    raise ValueError("Unsupported AST node")


def evaluate_deterministic_grounding_tools(user_text: str) -> str:
    """
    0ms Local Grounding Micro-Tools (Zero network calls, 100% deterministic):
    1. Safe AST Arithmetic & Percentage Evaluator
    2. Instant Unit & Temperature Converter (°F <-> °C, mi <-> km, lbs <-> kg)
    3. Fenced Python Code-Block Syntax & Traceback Inspector (`ast.parse`)
    Returns a compact `[System Verified Grounding: ...]` block when a match is found, or `""`.
    """
    if not user_text or not isinstance(user_text, str):
        return ""

    notes = []

    # 1. Inspect fenced Python code blocks for syntax errors in 0ms
    for py_match in _FENCED_PYTHON_BLOCK_REGEX.finditer(user_text):
        code_snippet = py_match.group(1).strip()
        if not code_snippet:
            continue
        try:
            ast.parse(code_snippet)
            notes.append("Python code block syntax check: VALID (0 syntax errors in AST)")
        except SyntaxError as syn_err:
            err_ln = (syn_err.text or "").strip()[:70]
            notes.append(
                f"Python SyntaxError on line {syn_err.lineno}, col {syn_err.offset}: {syn_err.msg}"
                + (f" -> `{err_ln}`" if err_ln else "")
            )
        break

    text_no_code = _FENCED_PYTHON_BLOCK_REGEX.sub(' ', user_text)
    text_no_urls = URL_REGEX.sub(' ', text_no_code)

    # 2. Percentage expressions ("15% of 340")
    for pm in _PERCENT_OF_REGEX.finditer(text_no_urls):
        try:
            pct = float(pm.group(1))
            base_val = float(pm.group(2).replace(",", ""))
            ans = (pct / 100.0) * base_val
            ans_str = f"{int(ans)}" if ans == int(ans) else f"{ans:.4f}".rstrip("0").rstrip(".")
            notes.append(f"{pm.group(0).strip()} = {ans_str}")
        except Exception:
            pass
        if len(notes) >= 4:
            break

    # 3. Safe AST arithmetic & math expressions
    for am in _ARITHMETIC_EXPR_REGEX.finditer(text_no_urls):
        raw_expr = am.group(0).strip()
        if _DATE_OR_VERSION_SKIP_REGEX.match(raw_expr):
            continue
        # Skip simple hyphenated ranges like "30-160" or "1-2" unless math context or other operators exist
        if re.fullmatch(r'\d+\s*\-\s*\d+', raw_expr) and not re.search(r'\b(?:what\s+is|calculate|equals|minus|math)\b|=', text_no_urls, re.IGNORECASE):
            continue
        norm_expr = raw_expr.replace(",", "").replace("^", "**")
        try:
            tree = ast.parse(norm_expr, mode="eval")
            val = _eval_safe_ast_node(tree)
            if isinstance(val, float):
                val_str = f"{int(val)}" if val.is_integer() and abs(val) < 1e15 else f"{val:.6g}"
            else:
                val_str = str(val)
            notes.append(f"{raw_expr} = {val_str}")
        except Exception:
            continue
        if len(notes) >= 4:
            break

    # 4. Unit conversions (when explicitly asked or containing temperature/unit values)
    if len(notes) < 4 and re.search(r'\b(?:convert|in\s+(?:celsius|fahrenheit|km|miles|kg|lbs)|°[fc]|fahrenheit|celsius)\b', text_no_urls, re.IGNORECASE):
        for um in _UNIT_CONV_REGEX.finditer(text_no_urls):
            try:
                val = float(um.group(1))
                u_from = um.group(2).lower().lstrip("°")
                if u_from in ("f", "fahrenheit"):
                    c_val = (val - 32.0) * 5.0 / 9.0
                    notes.append(f"{val:g}°F = {c_val:.1f}°C")
                elif u_from in ("c", "celsius"):
                    f_val = (val * 9.0 / 5.0) + 32.0
                    notes.append(f"{val:g}°C = {f_val:.1f}°F")
                elif u_from in ("mi", "mile", "miles") and um.group(3):
                    notes.append(f"{val:g} mi = {val * 1.60934:.2f} km")
                elif u_from in ("km", "kilometer", "kilometers") and um.group(3):
                    notes.append(f"{val:g} km = {val * 0.621371:.2f} mi")
                elif u_from in ("lb", "lbs", "pound", "pounds") and um.group(3):
                    notes.append(f"{val:g} lbs = {val * 0.453592:.2f} kg")
                elif u_from in ("kg", "kilogram", "kilograms") and um.group(3):
                    notes.append(f"{val:g} kg = {val * 2.20462:.2f} lbs")
            except Exception:
                continue
            if len(notes) >= 4:
                break

    if not notes:
        return ""
    return "[System Verified Grounding (0ms AST Tool — use naturally in-character if relevant): " + " | ".join(notes[:4]) + "]"
