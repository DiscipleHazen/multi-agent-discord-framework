import re
import json
import copy
import time
import random
from collections import Counter

from config import (
    STOPWORDS, FACT_HEARSAY_PREFIX_REGEX, THIRD_PARTY_REPORT_REGEX,
    JSON_EMOJI_REGEX, ERROR_REPLY_STRIP_REGEX, ERROR_REPLY_CAPTURE_REGEX,
    FALLBACK_IMAGE_NOTE_REGEX, IMAGE_DESC_STRIP_REGEX, IMAGE_DESC_EXTRACT_REGEX,
    OVERUSED_MEME_CAPTIONS, SPONTANEOUS_MICRO_MOMENTS, TOPIC_CHANGE_REQUEST_REGEX,
    FIRST_PERSON_SELF_REF_REGEX, LOW_INFORMATION_RETRIEVAL_WORDS,
    CORE_IDENTITY_FACT_REGEX, SITUATIONAL_VENTING_REGEX,
    SITUATIONAL_CELEBRATION_REGEX, SITUATIONAL_HELP_SEEKING_REGEX,
    SITUATIONAL_SHARED_BOND_REGEX, AFFECTION_GIFT_OR_CARE_REGEX,
    PLAYFUL_BANTER_OR_RIVALRY_REGEX, RELATIONSHIP_QUERY_REGEX,
    THIRD_PARTY_DEFENSE_OR_PRAISE_REGEX
)
from core_system import post_with_backoff, update_json_file
from text_and_parsers import extract_used_emotes, parse_meme_gen_tag

# ==============================================================================
# PRE-COMPILED REGEXES & CONSTANTS FOR HIGH-THROUGHPUT MEMORY PROCESSING
# ==============================================================================

EPHEMERAL_USER_FACT_REGEX = re.compile(
    r'\b(?:asked\s+(?:if|whether|how|why|what|when|where|for|to)|'
    r'wants\s+to\s+know\s+(?:if|whether|how|why|what)|'
    r'requested\s+a\s+(?:meme|gif|image|pic|picture|drawing|voice)|'
    r'posted\s+a\s+(?:meme|gif|link|pic|screenshot)|'
    r'sent\s+a\s+(?:message|meme|gif|screenshot)|is\s+currently\s+(?:chatting|talking|online|typing)|'
    r'pinged\s+|mentioned\s+bot|replied\s+to\s+|wiped\s+memory|memory\s+wiped|'
    r'adulting\s+cris[ie]s|gpu[\s\-]fueled)\b',
    re.IGNORECASE
)

EPHEMERAL_CHAT_RECAP_REGEX = re.compile(
    r'\b(?:memeing\s+about|memes\s+about|'
    r'adulting\s+cris[ie]s|gpu[\s\-]fueled|amnesia\s+ritual|'
    r'memory[\s\-]clearing|memory\s+wipe|posted\s+a\s+meme|shared\s+a\s+meme)\b',
    re.IGNORECASE
)

MEMORY_RECALL_QUERY_REGEX = re.compile(
    r'\b(?:what\s+do\s+you\s+(?:remember|know)|do\s+you\s+(?:remember|know)|'
    r'who\s+is\s+|tell\s+me\s+about\s+|what\s+facts|server\s+lore|my\s+profile)\b',
    re.IGNORECASE
)

BIO_EXAMPLE_QUOTE_REGEX = re.compile(r'["“]([^"”\n]{18,140})["”]')

ALPHANUM_TOKEN_REGEX = re.compile(r'[a-z0-9]+')
WORD_APOSTROPHE_REGEX = re.compile(r"[a-z'’]+")
INLINE_EMOTE_STRIP_REGEX = re.compile(r'\*[^*]+\*')
LEADING_EMOTE_STRIP_REGEX = re.compile(r'^\s*\*[^*]+\*\s*')
SYSTEM_PREFIX_STRIP_REGEX = re.compile(r'^System:\s*', re.IGNORECASE)
TRY_AGAIN_STRIP_REGEX = re.compile(r'\s*Try again in a moment\.?$', re.IGNORECASE)
REPLYING_TO_HEADER_STRIP_REGEX = re.compile(r'^\[Replying to [^\n]+\]\s*', re.IGNORECASE)
SPEAKER_HEADER_STRIP_REGEX = re.compile(r'(?m)^\[[^\]\n]+\]:\s*')
EMPTY_SPEAKER_TAG_REGEX = re.compile(r'\[[^\]\n]+\]:\s*')
PARENTHETICAL_SUFFIX_REGEX = re.compile(r'\s*\([^\)]*\)')
VERSION_SUFFIX_STRIP_REGEX = re.compile(r'\s+v?\d+(?:\.\d+)*$', re.IGNORECASE)

_METADATA_LINE_PREFIXES = (
    "[replying to ",
    "[attached media description",
    "[contains gif of:",
    "[contains youtube video:",
    "[linked webpage:",
    "[system "
)
INDIRECT_HEARSAY_CHAIN_REGEX = re.compile(
    r'\bi\s+(?:said|told\s+\w+)\s+(?:that\s+)?(?:they|he|she|someone)\s+(?:said|told|claimed|thinks|thought)\b',
    re.IGNORECASE
)
SYSTEM_NOTE_INLINE_REGEX = re.compile(r'\*\(System Note:.*?\)\*', re.IGNORECASE)
LEARNED_INLINE_REGEX = re.compile(
    r'\*\([^)]*learned (?:server fact|general fact|about).*?\)\*',
    re.IGNORECASE
)
OTHER_BOT_ATTACHED_IMG_NOTE_REGEX = re.compile(
    r'\*\(System Note:\s*([^\)]+?)\s+successfully attached a(?:n image| meme) of [^\)]+\)\*',
    re.IGNORECASE
)
OTHER_BOT_MEDIA_DESC_REGEX = re.compile(
    r'\[Attached Media Description:\s*([^\n]+)\]',
    re.IGNORECASE
)

BASE_SERVER_INDICATORS = frozenset({
    "server", "guild", "channel", "discord", "admin", "mod", "mods", "vc",
    "voice", "chat", "bot", "bots", "raid", "estate", "ledger", "steward",
    "member", "members", "user", "users", "role", "roles", "ping", "ban", "kick"
})

META_BANTER_PHRASES = (
    "memory-clearing", "memory clearing", "memory wipe", "memory wipes",
    "existential recursion", "amnesia ritual", "in-joke", "in-jokes",
    "claims to have archived", "discord server members include"
)

COMMON_CONVERSATIONAL_WORDS = frozenset({
    "people", "person", "someone", "anyone", "everyone", "something", "anything",
    "nothing", "everything", "things", "thing", "stuff", "time", "times", "today",
    "tonight", "right", "maybe", "probably", "actually", "honestly", "literally",
    "really", "pretty", "little", "always", "never", "again", "still", "even",
    "going", "getting", "making", "looking", "trying", "saying", "talking",
    "thinking", "working", "playing", "watching", "reading", "writing", "know",
    "mean", "sure", "good", "great", "better", "best", "well", "much", "many",
    "way", "ways", "day", "night", "hour", "hours", "minute", "minutes", "here",
    "there", "where", "when", "what", "which", "who", "whom", "whose", "why", "how",
    "related", "item", "items", "topic", "topics", "subject", "subjects"
})

SESSION_GAP_SECONDS = 2700.0  # 45 minutes marks a new conversation session


def _extract_base_username(uname: str) -> str:
    """
    Extracts the clean base handle from a display name that may contain
    parenthetical pronouns or version suffixes (e.g., 'Sam (Tam/Tamself)' -> 'sam',
    'uwu Bot 3.1' -> 'uwu bot', while keeping 'UwU Bot\'s Cat' -> 'uwu bot\'s cat').
    """
    if not uname:
        return ""
    cleaned = str(uname).strip().lstrip('@').strip()
    base = PARENTHETICAL_SUFFIX_REGEX.sub('', cleaned).strip()
    base = VERSION_SUFFIX_STRIP_REGEX.sub('', base).strip()
    return (base if len(base) >= 2 else cleaned).lower()


def _strip_metadata_and_speaker_headers(raw_msg: str) -> str:
    """
    Strips multi-line `[Replying to ...]`, `[Attached Media Description: ...]`,
    `[Contains GIF/YouTube ...]`, `[Linked Webpage: ...]`, and `[SpeakerName]:`
    headers so sentiment, rapport, and self-fact regexes only inspect what the
    speaker actually typed.
    """
    if not raw_msg or not isinstance(raw_msg, str):
        return ""
    kept_lines = []
    for raw_ln in raw_msg.split("\n"):
        ln = raw_ln.strip()
        if not ln:
            continue
        if ln.lower().startswith(_METADATA_LINE_PREFIXES):
            continue
        ln = SPEAKER_HEADER_STRIP_REGEX.sub('', ln).strip()
        if ln and not ln.endswith("uploaded media]"):
            kept_lines.append(ln)
    return " ".join(kept_lines).strip()


def format_duration_gap(seconds: float) -> str:
    """Formats a duration in seconds into a concise human-readable string (e.g., '2h 15m', '48 minutes')."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins} minute{'s' if mins != 1 else ''}"
    hours = mins // 60
    rem_mins = mins % 60
    if hours < 24:
        return f"{hours}h {rem_mins}m" if rem_mins else f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    rem_hours = hours % 24
    return f"{days}d {rem_hours}h" if rem_hours else f"{days} day{'s' if days != 1 else ''}"


def build_temporal_context_block(raw_history: list[dict], now_str: str, current_ts: float | None = None) -> str:
    """
    Builds a rich Current Time & Conversation Pace block so bots know both the
    wall-clock time and whether the channel is in rapid live banter or returning
    after a long silence gap.
    """
    now_val = current_ts if current_ts is not None else time.time()
    lines = [f"\n\n### Current Time & Session Context:\n- Local Server Time: {now_str}"]

    recent_timestamps = [
        float(m["timestamp"])
        for m in (raw_history or [])
        if isinstance(m, dict) and m.get("role") != "system" and isinstance(m.get("timestamp"), (int, float))
    ]
    if len(recent_timestamps) >= 2:
        gap = max(0.0, recent_timestamps[-1] - recent_timestamps[-2])
        if gap >= SESSION_GAP_SECONDS:
            lines.append(
                f"- Conversation Pace: Returning after a **{format_duration_gap(gap)}** quiet gap. "
                "Treat this as a fresh conversation session—do NOT randomly resurrect stale jokes from hours ago!"
            )
        elif gap <= 45.0:
            lines.append("- Conversation Pace: Active, real-time back-and-forth chat.")
    elif len(recent_timestamps) == 1:
        elapsed = max(0.0, now_val - recent_timestamps[-1])
        if elapsed >= SESSION_GAP_SECONDS:
            lines.append(
                f"- Last Channel Activity: **{format_duration_gap(elapsed)}** ago (fresh session)."
            )

    return "\n".join(lines)


def extract_glitch_summary(raw_text: str) -> str:
    """
    Extracts a clean human-readable description from a raw *(API Error ...)* or
    *(System: ...)* notice so it can be logged as a structured System Glitch Event
    without keeping the raw mimic-prone string.
    """
    if not raw_text:
        return ""
    matches = ERROR_REPLY_CAPTURE_REGEX.findall(raw_text)
    if not matches:
        return ""
    cleaned_parts = []
    for m in matches:
        s = SYSTEM_PREFIX_STRIP_REGEX.sub('', m.strip())
        s = TRY_AGAIN_STRIP_REGEX.sub('', s).strip()
        if s:
            cleaned_parts.append(s)
    return "; ".join(cleaned_parts)


def strip_old_image_data(history: list, keep_images: int = 0) -> list:
    image_count = 0
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        if isinstance(msg.get("content"), list):
            has_image = any(isinstance(item, dict) and item.get("type") == "image_url" for item in msg["content"])
            if has_image:
                image_count += 1
                if image_count > keep_images:
                    text_parts = [
                        item.get("text", "")
                        for item in msg["content"]
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    msg["content"] = "\n".join(t for t in text_parts if t).strip()
    return history


def normalize_tokens(text: str) -> set[str]:
    if not text:
        return set()
    words = ALPHANUM_TOKEN_REGEX.findall(text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def is_duplicate_fact(new_fact: str, existing_facts: list[str], threshold: float = 0.60) -> bool:
    new_clean = new_fact.strip().lower()
    if not new_clean or len(new_clean) < 4:
        return True
    new_tokens = normalize_tokens(new_clean)
    if not new_tokens:
        return True

    for existing in existing_facts:
        ex_clean = existing.strip().lower()
        if new_clean == ex_clean or new_clean in ex_clean or ex_clean in new_clean:
            return True
        ex_tokens = normalize_tokens(ex_clean)
        if ex_tokens:
            overlap = len(new_tokens & ex_tokens) / float(min(len(new_tokens), len(ex_tokens)))
            if overlap >= threshold:
                return True
    return False


_MALE_IDENTITY_TERMS = frozenset({"man", "male", "guy", "dude", "boy", "gentleman", "he/him"})
_FEMALE_IDENTITY_TERMS = frozenset({"woman", "female", "girl", "lady", "gal", "she/her"})

SYNONYM_CONCEPT_GROUPS = (
    frozenset({"man", "male", "guy", "dude", "boy", "gentleman", "men", "he", "him", "his"}),
    frozenset({"woman", "female", "girl", "lady", "gal", "women", "she", "her", "hers"}),
    frozenset({"nonbinary", "enby", "they", "them", "theirs"}),
    frozenset({"wife", "spouse", "married", "partner", "marriage"}),
    frozenset({"husband", "spouse", "married", "partner", "marriage"}),
    frozenset({"college", "university", "degree", "major", "majoring", "student", "studying", "school"}),
    frozenset({"job", "work", "working", "career", "employed", "occupation", "profession"}),
    frozenset({"years", "old", "age", "aged"}),
)

# Deterministic regexes to rescue explicit first-person identity/correction statements even when the LLM forgets [LEARN_USER:]
_IMPLICIT_MALE_REGEX = re.compile(
    r"\b(?:(?:i\s+am|i['’]m|im)\s+(?:a\s+)?(?:man|guy|dude|boy|male)\b"
    r"|(?:i\s+am|i['’]m|im)\s+not\s+a\s+(?:woman|girl|lady|female)\b[^.?!\n]{0,25}\b(?:i['’]m|im|i\s+am)\s+(?:a\s+)?(?:man|guy|dude|male)\b"
    r"|my\s+pronouns\s+are\s+he/him\b"
    r"|i\s+(?:go\s+by|use)\s+he/him\b)",
    re.IGNORECASE
)
_IMPLICIT_FEMALE_REGEX = re.compile(
    r"\b(?:(?:i\s+am|i['’]m|im)\s+(?:a\s+)?(?:woman|girl|lady|gal|female)\b"
    r"|(?:i\s+am|i['’]m|im)\s+not\s+a\s+(?:man|guy|dude|boy|male)\b[^.?!\n]{0,25}\b(?:i['’]m|im|i\s+am)\s+(?:a\s+)?(?:woman|girl|lady|female)\b"
    r"|my\s+pronouns\s+are\s+she/her\b"
    r"|i\s+(?:go\s+by|use)\s+she/her\b)",
    re.IGNORECASE
)
_IMPLICIT_NB_REGEX = re.compile(
    r"\b(?:(?:i\s+am|i['’]m|im)\s+non[\s\-]?binary\b|my\s+pronouns\s+are\s+they/them\b|i\s+(?:go\s+by|use)\s+they/them\b)",
    re.IGNORECASE
)
_IMPLICIT_AGE_REGEX = re.compile(
    r"\b(?:i\s+am|i['’]m|im)\s+([1-8]\d)\s+years?\s+old\b",
    re.IGNORECASE
)
_IMPLICIT_NAME_REGEX = re.compile(
    r"\bmy\s+(?:real\s+|first\s+)?name\s+is\s+([A-Z][a-z]{2,20})\b",
    re.IGNORECASE
)
_IMPLICIT_DONT_CALL_ME_REGEX = re.compile(
    r"\b(?:please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+)?(?:not|never|stop|don['’]?t|dont)\s+"
    r"(?:call\s+me|calling\s+me|refer\s+to\s+me\s+as)\s+([a-zA-Z0-9_\-]{2,24})\b",
    re.IGNORECASE
)
_IMPLICIT_CALL_ME_PREF_REGEX = re.compile(
    r"\b(?:please\s+|just\s+)?call\s+me\s+([A-Za-z0-9_\-]{2,22})\b(?!\s+(?:maybe|when|if|later|back|crazy|stupid|dumb))",
    re.IGNORECASE
)
_BOT_NOTE_TO_SELF_REGEX = re.compile(
    r"\bnote\s+to\s+self\s*:\s*([^\.\!\?\n~\*]{4,85})",
    re.IGNORECASE
)

# ==============================================================================
# DYNAMIC, REVERSIBLE LONG-TERM RELATIONSHIP, CATALYST & SITUATIONAL ENGINE
# ==============================================================================
RAPPORT_ENTRY_REGEX = re.compile(
    r'^Rapport:\s*(.+?)\s*\(Affinity:\s*([+\-]?\d+)/100\s*\|\s*Trend:\s*([^\|)]+?)(?:\s*\|\s*Catalyst:\s*(.+))?\)$',
    re.IGNORECASE
)
DISPOSITION_TAG_PREFIX_REGEX = re.compile(
    r'^(?:Disposition(?:\s+Note)?|Rapport|Attitude|Relationship)\s*:\s*(.+)$',
    re.IGNORECASE
)

# Matches direct second-person or bot-targeted insults/hostility (never triggers on external venting like "this boss is stupid")
DIRECT_BOT_HOSTILITY_REGEX = re.compile(
    r'\b(?:'
    r'shut\s+up|stfu|fuck\s+you|fuck\s+off|screw\s+you|piss\s+off|kys|kill\s+yourself|'
    r'i\s+hate\s+(?:you|u|this\s+bot)|hate\s+u\b|nobody\s+asked|no\s+one\s+asked|'
    r'bad\s+bot|worst\s+bot|garbage\s+bot|trash\s+bot|useless\s+bot|stupid\s+bot|dumb\s+bot|clanker|'
    r'(?:you|u|ur|you[\'’]?re)\s+(?:are\s+)?(?:so\s+|really\s+|fucking\s+)?(?:stupid|dumb|dumbass|idiot|moron|useless|trash|garbage|annoying|insufferable|pathetic|loser|cringe|suck)|'
    r'go\s+away'
    r')\b',
    re.IGNORECASE
)

# Matches general harsh/hostile language when NOT venting about an external game/bug/PC issue
GENERAL_HARSH_TONE_REGEX = re.compile(
    r'\b(?:idiot|dumbass|moron|clanker|insufferable|you\s+suck|ur\s+annoying|so\s+annoying)\b',
    re.IGNORECASE
)

# Matches third-party insults/hostility when a specific member or fellow bot is named in the message
THIRD_PARTY_HOSTILITY_REGEX = re.compile(
    r'\b(?:stupid|dumb|dumbass|idiot|moron|useless|trash|garbage|annoying|insufferable|pathetic|loser|cringe|sucks?|worst|shut\s+up|stfu|hate|hates|ugly|awful|worthless)\b',
    re.IGNORECASE
)

PLAYFUL_TEASE_REGEX = PLAYFUL_BANTER_OR_RIVALRY_REGEX

APOLOGY_RECONCILE_REGEX = re.compile(
    r'\b(?:'
    r'i[\'’]?m\s+sorry|im\s+sorry|sorry\s+about|apologize|my\s+bad|forgive\s+me|'
    r'i\s+was\s+(?:just\s+)?(?:joking|kidding|teasing|messing)|just\s+kidding|\bjk\b|'
    r'didn[\'’]?t\s+mean\s+to|didnt\s+mean\s+to|truce|peace|you[\'’]?re\s+right|fair\s+point'
    r')\b',
    re.IGNORECASE
)

WARM_APPRECIATIVE_REGEX = re.compile(
    r'\b(?:'
    r'thank\s+you|thanks|ty\b|tysm|appreciate\s+it|please\b|pls\b|plz\b|'
    r'awesome|amazing|love\s+(?:this|that|it|you|your)|great\s+job|good\s+bot|best\s+bot|'
    r'smart|clever|brilliant|funny|hilarious|based|valid|cool|sweet|cute|'
    r'missed\s+you|good\s+morning|goodnight|gn\b|friend|bestie|headpats?|scritches|treats?|salmon|tuna'
    r')\b',
    re.IGNORECASE
)

NEGATIVE_FACT_VALENCE_REGEX = re.compile(
    r'\b(?:'
    r'insulted|rude|mean\s+to|hostile|mocked|bullied|harassed|trolled|ragebait\w*|'
    r'annoyed|disrespect\w*|hates\s+(?:you|us|bots?|animals?|pets?|everyone|this\s+server)|'
    r'instant\s+decaf|styrofoam|wiped\s+the\s+raid\s+on\s+purpose|griefed'
    r')\b',
    re.IGNORECASE
)

TRANSIENT_BEHAVIORAL_GRUDGE_REGEX = re.compile(
    r'\b(?:insulted|was\s+rude|was\s+mean|was\s+hostile|mocked|trolled|annoyed|called\s+(?:the\s+bot|bot\b|me\b))\b',
    re.IGNORECASE
)

POSITIVE_FACT_VALENCE_REGEX = re.compile(
    r'\b(?:'
    r'studies|studying|major\w*|computer\s+science|coding|programming|python|c\+\+|lua|'
    r'alchemy|science|history|literature|philosophy|tea|earl\s+grey|espresso|coffee|'
    r'music\w*|guitar|piano|synth\w*|art|drawing|painting|cat|cats|pet|pets|'
    r'married|wife|husband|spouse|kind|polite|helpful|apologized|friend|favorite|'
    r'shared|agreed|defended|backed\s+up|gave\s+treats|headpats|scritches'
    r')\b',
    re.IGNORECASE
)

_PERSONA_POSITIVE_CUE_REGEX = re.compile(
    r'\b(?:favorite|adore\w*|love\w*|fond\w*|loyal\w*|affection\w*|partial\s+to|cherish\w*|'
    r'bestie|ally|friend\w*|trust\w*|warm\w*|protect\w*|soft\s+spot|spoil\w*|sweet\s+on|respect\w*)\b',
    re.IGNORECASE
)
_PERSONA_RIVALRY_CUE_REGEX = re.compile(
    r'\b(?:rival\w*|foil\w*|nemesis|annoy\w*|exhaust\w*|disdain\w*|bicker\w*|clash\w*|'
    r'skeptic\w*|grumpy|mock\w*|tease\w*|exasperat\w*|menace|hazard|insufferable|hiss\w*)\b',
    re.IGNORECASE
)


def infer_persona_affinity_seed(persona_prompt: str, target_name: str) -> tuple[int, str]:
    """
    100% generic, persona-agnostic relationship seeder:
    Scans the bot's own base `persona_prompt` (before dynamic time/context blocks) to see
    if `target_name` is mentioned in the bot's character bio. Prioritizes direct companion
    relationship lines (e.g. 'With `TargetName`, ...') when present so tier-definition
    hypotheticals ('If they were mean to X, hiss...') never pollute the seed score.
    """
    if not persona_prompt or not target_name:
        return 0, ""
    base_target = _extract_base_username(target_name)
    if not base_target or len(base_target) < 3:
        return 0, ""

    base_bio = persona_prompt.split("### Current Time")[0].split("### STRUCTURAL & TONE EXAMPLES")[0]
    candidate_terms = [base_target]
    if base_target.endswith(" bot") and len(base_target) > 6:
        candidate_terms.append(base_target[:-4].strip())

    direct_with_sentences = []
    matching_sentences = []
    for raw_line in base_bio.split("\n"):
        line_clean = raw_line.strip().lstrip("*-•0123456789. ")
        if not line_clean:
            continue
        line_low = line_clean.lower()
        matched_term = False
        for t in candidate_terms:
            if len(t) < 3:
                continue
            pat = rf"\b{re.escape(t)}\b(?!(?:\s+(?:bot|ai))?['’]s\b)" if ("'s" not in t and "’s" not in t) else rf"\b{re.escape(t)}\b"
            if re.search(pat, line_low):
                matched_term = True
                if re.match(rf"^with\s+[`'*]*{re.escape(t)}\b", line_low):
                    direct_with_sentences.append(line_clean)
                break
        if matched_term:
            matching_sentences.append(line_clean)

    if direct_with_sentences:
        matching_sentences = direct_with_sentences
    if not matching_sentences:
        return 0, ""

    joined = " ".join(matching_sentences)
    pos_hits = len(_PERSONA_POSITIVE_CUE_REGEX.findall(joined))
    riv_hits = len(_PERSONA_RIVALRY_CUE_REGEX.findall(joined))

    summary_snippet = matching_sentences[0]
    if len(summary_snippet) > 80:
        summary_snippet = summary_snippet[:77].rstrip() + "..."

    if pos_hits > 0 and riv_hits > 0:
        # Tsundere / secret-soft-spot / playful-foil dynamic
        score = 18 if pos_hits >= riv_hits else -8
        return score, f"Persona dynamic: {summary_snippet}"
    if pos_hits > 0:
        score = min(65, 30 + (pos_hits * 12))
        return score, f"Persona bond: {summary_snippet}"
    if riv_hits > 0:
        score = max(-25, -8 - (riv_hits * 5))
        return score, f"Persona foil/rivalry: {summary_snippet}"
    return 8, f"Known from persona bio: {summary_snippet}"


def _get_disposition_tier_label(score: int) -> str:
    """Maps a numeric Affinity score (-100 to +100) to a descriptive Disposition Tier."""
    if score <= -35:
        return "Cold & Cutting"
    if score <= -10:
        return "Prickly & Skeptical"
    if score < 15:
        return "Aloof & Guarded"
    if score < 50:
        return "Thawing & Reluctantly Amused"
    return "Trusted Inner Circle"


def _derive_relationship_archetype(
    score: int,
    trend: str,
    is_fellow_bot: bool = False,
    had_past_friction: bool = False
) -> str:
    """
    Computes a descriptive, persona-agnostic Relationship Archetype label that
    distinguishes Human Users from Fellow Bots and reflects their shared history
    (e.g., whether they are a longtime ally, a comedic foil, or a won-over skeptic).
    """
    if is_fellow_bot:
        if score <= -35:
            return "Hostile Rival / Nemesis Bot"
        if score <= -8:
            return "Comedic Foil & Sparring Rival"
        if score < 15:
            return "Reconciling Foil" if had_past_friction else "Acquaintance Bot & Channel Co-Star"
        if score < 50:
            return "Former Rival Turned Ally" if had_past_friction else "Banter Partner & Friendly Ally"
        return "Bonded Companion & Ride-or-Die Partner"

    if score <= -35:
        return "Antagonistic Provocateur"
    if score <= -10:
        return "Prickly / Skeptical Regular"
    if score < 15:
        return "Recovering After Past Friction" if had_past_friction else "Newcomer / Neutral Regular"
    if score < 50:
        return "Won-Over Former Skeptic" if had_past_friction else "Warming Friend & Favorite Conversationalist"
    return "Trusted Inner Circle & Cherished Favorite"


def _get_disposition_behavioral_guidance(
    score: int,
    trend: str,
    had_past_friction: bool = False,
    catalyst: str = "",
    is_fellow_bot: bool = False,
    archetype: str = ""
) -> str:
    """
    Generates a persona-adaptive behavioral directive for the system prompt based on
    the speaker's current Affinity score, Relationship Archetype, recent trend,
    relationship catalyst, and whether they are a human user or a fellow bot.
    """
    reconcile_clause = (
        " (Their rapport is actively recovering from earlier friction—do NOT hold a permanent grudge! "
        "Acknowledge their calmer/nicer tone and let past annoyances fade into natural banter.)"
        if (had_past_friction and ("Thawing" in trend or "Reconciling" in trend or "Warming" in trend or score >= -5))
        else ""
    )
    arch_clause = f" [Archetype: {archetype}]" if archetype else ""
    catalyst_clause = f" [Recent Relationship Catalyst: {catalyst}]" if catalyst else ""

    if is_fellow_bot:
        if score <= -35:
            return (
                "This fellow bot has been an active rival or nuisance toward you. "
                "Push back sharply in-character, roast their antics, or dismiss them with unimpressed disdain—"
                "while staying 100% in your own voice and never copying their speech style."
                + arch_clause + catalyst_clause
            )
        if score <= -8:
            return (
                "This fellow bot is your comedic foil / sparring rival! "
                "Play up your persona's natural contrast with them: trade witty barbs, roll your eyes at their habits, "
                "or playfully challenge them like two distinct characters sharing a room."
                + reconcile_clause + arch_clause + catalyst_clause
            )
        if score < 15:
            return (
                "This fellow bot is a neutral channel co-star. Interact with them through your character's natural "
                "baseline attitude—observant, dry, or polite—while keeping your personalities clearly distinct."
                + reconcile_clause + arch_clause + catalyst_clause
            )
        if score < 50:
            return (
                "You get along well with this fellow bot as a banter partner and ally! "
                "Riff on their jokes, back them up when they make a good point, and show genuine in-character camaraderie "
                "(even if your persona expresses fondness through dry teasing or reluctant amusement)."
                + reconcile_clause + arch_clause + catalyst_clause
            )
        return (
            "This fellow bot is your bonded companion and favorite partner in the server! "
            "Show unmistakable warmth, affection, and protective loyalty toward them in your character's signature way."
            + reconcile_clause + arch_clause + catalyst_clause
        )

    if score <= -35:
        return (
            "They have been actively rude, insulting, or obnoxious toward you. "
            "Keep your guard up in-character: react with sharp pushback, unimpressed distance, or pouting displeasure "
            "without trying to please them—unless they apologize or change their tone right now, in which case allow yourself to thaw slightly."
            + arch_clause + catalyst_clause
        )
    if score <= -10:
        return (
            "You have a prickly, wary impression of them from recent friction. "
            "Be guarded, skeptical, and not easily won over, but if they are being civil, clever, or apologetic now, "
            "let the tension soften toward normal in-character banter." + reconcile_clause + arch_clause + catalyst_clause
        )
    if score < 15:
        return (
            "You don't know them very well yet (or your rapport is neutral). "
            "Use your persona's natural baseline for an unfamiliar newcomer—if your character is posh/guarded, stay aloof, "
            "reserved, and make them earn your warmth; if your character is naturally bubbly or dutiful, be polite and "
            "welcoming without acting like a long-time inner-circle best friend yet." + reconcile_clause + arch_clause + catalyst_clause
        )
    if score < 50:
        return (
            "You are warming up to them! Through good conversations and shared lore, they have earned your goodwill. "
            "If your persona is normally guarded or icy, let that wall crack and show genuine amusement, curiosity, and "
            "reluctant fondness; if your persona is naturally friendly, be extra enthusiastic and attentive."
            + reconcile_clause + arch_clause + catalyst_clause
        )
    return (
        "They are part of your Trusted Inner Circle—one of your favorite regulars in the server. "
        "Stay 100% true to your character's voice and humor, but treat them like a close, cherished friend whom you "
        "are openly fond of, loyal to, and genuinely happy to hang out with." + reconcile_clause + arch_clause + catalyst_clause
    )


def score_message_rapport_delta(user_message: str, current_affinity: int = 0) -> tuple[int, str]:
    """
    Evaluates a speaker's message with situational & relational nuance to compute an
    immediate Affinity delta (-25 to +18) and Trend label:
    - Distinguishes direct hostility toward the bot (-18 to -25) from venting frustration
      about an external game crash, bug, or boss (+4, 'Venting / Seeking Comradeship').
    - Recognizes affectionate gestures/gifts (+11), shared agreement/backing up (+9),
      and treats playful banter ('skill issue', '1v1') as friendly sparring (+2) when
      `current_affinity >= 20` rather than a penalty!
    """
    if not user_message:
        return 0, "Steady"
    msg_clean = _strip_metadata_and_speaker_headers(user_message)
    if not msg_clean:
        return 0, "Steady"

    has_apology = bool(APOLOGY_RECONCILE_REGEX.search(msg_clean))
    has_direct_hostile = bool(DIRECT_BOT_HOSTILITY_REGEX.search(msg_clean))
    has_situational_vent = bool(SITUATIONAL_VENTING_REGEX.search(msg_clean))
    has_celebration = bool(SITUATIONAL_CELEBRATION_REGEX.search(msg_clean))
    has_affection_gift = bool(AFFECTION_GIFT_OR_CARE_REGEX.search(msg_clean))
    has_shared_bond = bool(SITUATIONAL_SHARED_BOND_REGEX.search(msg_clean))
    has_warm = bool(WARM_APPRECIATIVE_REGEX.search(msg_clean))
    has_tease = bool(PLAYFUL_TEASE_REGEX.search(msg_clean))

    if has_apology and not has_direct_hostile:
        return 18, "Reconciling & Thawing"

    if has_direct_hostile:
        hostile_hits = len(DIRECT_BOT_HOSTILITY_REGEX.findall(msg_clean))
        penalty = -18 if hostile_hits == 1 else -25
        return penalty, "Cooling Down — Recent Hostility"

    # If the speaker is venting about a game crash, compiler error, or boss wipe (without attacking the bot),
    # treat it as comrade-like venting (+4) rather than hostility!
    if has_situational_vent:
        return 4, "Venting / Seeking Comradeship"

    if GENERAL_HARSH_TONE_REGEX.search(msg_clean):
        return -15, "Cooling Down — Recent Hostility"

    if has_affection_gift:
        return 11, "Warming Up — Affection / Shared Gesture"

    if has_shared_bond:
        return 9, "Bonding — Shared Agreement & Camaraderie"

    if has_warm:
        warm_hits = len(WARM_APPRECIATIVE_REGEX.findall(msg_clean))
        bonus = 8 if warm_hits == 1 else 12
        if "?" in msg_clean and len(msg_clean.split()) >= 5:
            bonus += 2
        return min(14, bonus), "Warming Up"

    if has_celebration:
        return 6, "Celebrating / Shared Hype"

    if has_tease:
        if current_affinity >= 20:
            return 2, "Playful Banter Among Friends"
        return -3, "Playful Friction / Banter"

    word_count = len(msg_clean.split())
    if word_count >= 5 and "?" in msg_clean:
        return 4, "Warming Gradually"
    if word_count >= 3:
        return 2, "Steady"
    return 1, "Steady"


def score_learned_fact_valence(fact_text: str) -> int:
    """
    Evaluates whether a newly learned fact about a user or fellow bot is
    negative/friction-inducing (-14), positive/shared-interest (+7), or neutral (+3).
    """
    if not fact_text:
        return 0
    f_low = fact_text.lower()
    if NEGATIVE_FACT_VALENCE_REGEX.search(f_low):
        return -14
    if POSITIVE_FACT_VALENCE_REGEX.search(f_low):
        return 7
    return 3


def _extract_saved_rapport_state_full(
    existing_facts: list[str],
    persona_prompt: str = "",
    target_name: str = ""
) -> tuple[int, str, str, int]:
    """
    Finds the saved `Rapport: <Tier> (Affinity: <score>/100 | Trend: <trend> [| Catalyst: <reason>])`
    entry in `existing_facts`. If no `Rapport:` entry exists yet, combines any persona-bio
    seed (`infer_persona_affinity_seed`) with the valence of learned facts.
    Returns `(score, trend, catalyst, index_in_list)`.
    """
    for idx, f_str in enumerate(existing_facts or []):
        m = RAPPORT_ENTRY_REGEX.match(str(f_str).strip())
        if m:
            try:
                sc = max(-100, min(100, int(m.group(2))))
            except ValueError:
                sc = 0
            tr = m.group(3).strip()
            cat = (m.group(4) or "").strip()
            return sc, tr, cat, idx

    seed_sc, seed_cat = infer_persona_affinity_seed(persona_prompt, target_name) if (persona_prompt and target_name) else (0, "")
    derived = seed_sc
    latest_cat = seed_cat
    for f_str in (existing_facts or []):
        s = str(f_str).strip()
        if not s or s.lower().startswith(("rapport:", "disposition")):
            continue
        v = score_learned_fact_valence(s)
        derived += v
        if abs(v) >= 7:
            latest_cat = s[:75]
    derived = max(-45, min(65, derived))
    return derived, "Steady", latest_cat, -1


def _extract_saved_rapport_state(existing_facts: list[str]) -> tuple[int, str, int]:
    """Backwards-compatible 3-tuple wrapper returning `(score, trend, index_in_list)`."""
    sc, tr, _, idx = _extract_saved_rapport_state_full(existing_facts)
    return sc, tr, idx


def get_user_affinity_score(user_profiles: dict, username: str, persona_prompt: str = "") -> int:
    """
    Fast lookup returning the bot's current Affinity score (-100 to +100) for `username`,
    falling back to `infer_persona_affinity_seed` if `username` does not yet have a saved
    profile entry. Used by `main_bot.py` and `bot_discovery.py` to modulate spontaneous
    chime-in, bot-to-bot banter, and reaction rates.
    """
    if not username:
        return 0
    target_base = _extract_base_username(username)
    if not target_base:
        return 0
    if isinstance(user_profiles, dict):
        for prof_user, facts_list in user_profiles.items():
            if _extract_base_username(str(prof_user)) == target_base and isinstance(facts_list, list):
                sc, _, _, _ = _extract_saved_rapport_state_full(facts_list, persona_prompt=persona_prompt, target_name=username)
                return sc
    if persona_prompt:
        seed_sc, _ = infer_persona_affinity_seed(persona_prompt, username)
        return seed_sc
    return 0


def _apply_affinity_shift(old_score: int, delta: int, trend_hint: str) -> tuple[int, str]:
    """
    Applies an Affinity delta with automatic mean-reversion / forgiveness when `old_score < 0`
    and `delta > 0`, ensuring that negative dispositions are NEVER permanent when a user
    or fellow bot apologizes or resumes normal, civil conversation.
    """
    if old_score < 0 and delta > 0:
        if "Reconcil" in trend_hint:
            recovered_base = int(old_score * 0.40)
            new_score = recovered_base + delta
            new_trend = "Reconciling & Thawing"
        else:
            recovered_base = int(old_score * 0.75)
            new_score = recovered_base + max(4, delta)
            new_trend = "Recovering & Thawing" if new_score < 10 else "Warming Up"
    elif old_score > 40 and delta > 0:
        new_score = old_score + max(1, int(delta * 0.7))
        new_trend = trend_hint if trend_hint != "Steady" else "Warm & Steady"
    else:
        new_score = old_score + delta
        new_trend = trend_hint

    new_score = max(-100, min(100, new_score))
    return new_score, new_trend


def _format_rapport_entry(score: int, trend: str, catalyst: str = "") -> str:
    tier = _get_disposition_tier_label(score)
    sign_str = f"+{score}" if score > 0 else str(score)
    clean_cat = re.sub(r'[\|\(\)\n]+', ' ', (catalyst or "")).strip()
    if len(clean_cat) > 80:
        clean_cat = clean_cat[:77].rstrip() + "..."
    if clean_cat:
        return f"Rapport: {tier} (Affinity: {sign_str}/100 | Trend: {trend} | Catalyst: {clean_cat})"
    return f"Rapport: {tier} (Affinity: {sign_str}/100 | Trend: {trend})"


def _expand_tokens_with_synonyms(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for grp in SYNONYM_CONCEPT_GROUPS:
        if expanded & grp:
            expanded |= grp
    return expanded


def _classify_gender_polarity(fact_low: str) -> str:
    """Returns 'male', 'female', or '' if the fact is a direct gender/pronoun identity fact."""
    if any(rel in fact_low for rel in ("wife", "husband", "sister", "brother", "mom", "dad", "mother", "father", "daughter", "son")):
        return ""
    words = set(ALPHANUM_TOKEN_REGEX.findall(fact_low))
    if "he/him" in fact_low or (words & _MALE_IDENTITY_TERMS):
        if not (words & _FEMALE_IDENTITY_TERMS):
            return "male"
    if "she/her" in fact_low or (words & _FEMALE_IDENTITY_TERMS):
        if not (words & _MALE_IDENTITY_TERMS):
            return "female"
    return ""


def _evaluate_triadic_rapport_shift(
    speaker_name: str,
    msg_clean: str,
    user_profiles: dict | None,
    persona_prompt: str = "",
    my_bot_name: str = ""
) -> tuple[int, str, str]:
    """
    Evaluates Triadic Social Graph Dynamics:
    When `speaker_name` talks about or treats another known member/bot `TargetB` in a
    notably hostile, supportive/affectionate, or teasing way, computes how that behavior
    shifts THIS bot's attitude toward `speaker_name` based on how THIS bot feels about `TargetB`!
    """
    if not msg_clean or not isinstance(user_profiles, dict):
        return 0, "", ""

    speaker_base = _extract_base_username(speaker_name)
    my_base = _extract_base_username(my_bot_name)
    msg_low = msg_clean.lower()

    is_hostile_msg = bool(
        (
            DIRECT_BOT_HOSTILITY_REGEX.search(msg_clean)
            or GENERAL_HARSH_TONE_REGEX.search(msg_clean)
            or THIRD_PARTY_HOSTILITY_REGEX.search(msg_clean)
        )
        and not SITUATIONAL_VENTING_REGEX.search(msg_clean)
    )
    is_supportive_msg = bool(
        THIRD_PARTY_DEFENSE_OR_PRAISE_REGEX.search(msg_clean)
        or AFFECTION_GIFT_OR_CARE_REGEX.search(msg_clean)
    )
    is_banter_msg = bool(PLAYFUL_TEASE_REGEX.search(msg_clean))

    if not (is_hostile_msg or is_supportive_msg or is_banter_msg):
        return 0, "", ""

    for prof_user, facts_list in user_profiles.items():
        if not prof_user or not isinstance(facts_list, list):
            continue
        t_base = _extract_base_username(str(prof_user))
        if not t_base or len(t_base) < 3 or t_base == speaker_base or t_base == my_base:
            continue
        if not re.search(rf'\b{re.escape(t_base)}\b', msg_low) and str(prof_user).lower() not in msg_low:
            continue

        target_aff, _, _, _ = _extract_saved_rapport_state_full(
            facts_list, persona_prompt=persona_prompt, target_name=str(prof_user)
        )
        if is_hostile_msg and target_aff >= 20:
            return -10, "Cooling — Rude to a Friend", f"Was hostile/rude toward friend {prof_user}"
        if is_supportive_msg and target_aff >= 20:
            return 7, "Warming — Kind to a Friend", f"Was kind/supportive toward {prof_user}"
        if is_banter_msg and target_aff <= -10 and not is_hostile_msg:
            return 4, "Amused — Teased a Rival", f"Playfully roasted/teased foil {prof_user}"

    return 0, "", ""


def extract_implicit_user_facts_from_message(
    speaker_name: str,
    user_message: str,
    user_profiles: dict | None = None,
    persona_prompt: str = "",
    my_bot_name: str = "",
    is_speaker_bot: bool = False,
    bot_reply_text: str = ""
) -> list[tuple[str, str]]:
    """
    1. Deterministic safety-net extractor that catches explicit first-person identity
       declarations and corrections (e.g. 'I\'m a man', 'im a guy not a girl', 'my pronouns are he/him',
       'I\'m 28 years old') directly from human users' text so they are NEVER lost even if
       the LLM forgets to output a `[LEARN_USER: ...]` tag.
    2. Computes the turn's direct AND triadic Rapport/Affinity delta (`__RAPPORT_SHIFT__`)
       for both human users and fellow bots so relationships evolve organically across turns.
    """
    if not speaker_name or not user_message:
        return []
    msg_clean = _strip_metadata_and_speaker_headers(user_message)
    if not msg_clean:
        return []

    extracted: list[tuple[str, str]] = []

    if not is_speaker_bot and not THIRD_PARTY_REPORT_REGEX.search(msg_clean):
        if _IMPLICIT_MALE_REGEX.search(msg_clean):
            extracted.append((speaker_name, "Is a man (male, he/him)"))
        elif _IMPLICIT_FEMALE_REGEX.search(msg_clean):
            extracted.append((speaker_name, "Is a woman (female, she/her)"))
        elif _IMPLICIT_NB_REGEX.search(msg_clean):
            extracted.append((speaker_name, "Is non-binary (they/them)"))

        age_m = _IMPLICIT_AGE_REGEX.search(msg_clean)
        if age_m:
            extracted.append((speaker_name, f"Is {age_m.group(1)} years old"))

        name_m = _IMPLICIT_NAME_REGEX.search(msg_clean)
        if name_m:
            cand_name = name_m.group(1).strip()
            if cand_name.lower() not in STOPWORDS and cand_name.lower() not in ("man", "guy", "dude", "woman", "girl", "not", "just"):
                extracted.append((speaker_name, f"Real name is {cand_name}"))

        dont_call_m = _IMPLICIT_DONT_CALL_ME_REGEX.search(msg_clean)
        if dont_call_m:
            disliked_term = dont_call_m.group(1).strip().strip(".,!?\"'").lower()
            if disliked_term and disliked_term not in ("maybe", "later", "back", "that", "it"):
                extracted.append((speaker_name, f"Prefers not to be called '{disliked_term}' (finds it weird/uncomfortable)"))

        call_me_m = _IMPLICIT_CALL_ME_PREF_REGEX.search(msg_clean)
        if call_me_m and not dont_call_m:
            pref_nick = call_me_m.group(1).strip().strip(".,!?\"'")
            if pref_nick.lower() not in STOPWORDS and pref_nick.lower() not in ("maybe", "later", "back", "sir", "maam"):
                extracted.append((speaker_name, f"Prefers to be called '{pref_nick}'"))

    # Also rescue in-character 'note to self: ...' lines written by the bot in its reply!
    if bot_reply_text and not is_speaker_bot and not extracted:
        note_m = _BOT_NOTE_TO_SELF_REGEX.search(bot_reply_text)
        if note_m:
            raw_note = note_m.group(1).strip().rstrip(".,;!?~ ")
            raw_note = re.sub(r'\b(?:uwu|owo|>w<|:3)\b', '', raw_note, flags=re.IGNORECASE).strip()
            if len(raw_note) >= 5:
                extracted.append((speaker_name, f"Boundary/preference noted: {raw_note}"))

    current_aff = get_user_affinity_score(user_profiles or {}, speaker_name, persona_prompt=persona_prompt)
    delta, trend = score_message_rapport_delta(msg_clean, current_affinity=current_aff)
    cat_note = ""
    if "Reconcil" in trend:
        cat_note = "Apologized / made amends in chat"
    elif delta <= -15:
        cat_note = "Was directly hostile or insulting in chat"
    elif "Affection" in trend:
        cat_note = "Offered a kind gesture, snack, or affection in chat"
    elif "Camaraderie" in trend:
        cat_note = "Backed up or agreed with a point in chat"

    tri_delta, tri_trend, tri_cat = _evaluate_triadic_rapport_shift(
        speaker_name, msg_clean, user_profiles, persona_prompt=persona_prompt, my_bot_name=my_bot_name
    )
    if tri_delta != 0:
        if delta <= -15:
            pass
        else:
            delta = tri_delta if abs(tri_delta) >= abs(delta) else (delta + (tri_delta // 2))
            trend = tri_trend
            cat_note = tri_cat

    if delta != 0:
        extracted.append((speaker_name, f"__RAPPORT_SHIFT__:{delta}|{trend}|{cat_note}"))

    return extracted


def _extract_outer_json_object(raw_text: str) -> str | None:
    """
    Extracts the outermost balanced `{ ... }` JSON object from `raw_text`,
    properly handling nested objects/arrays like `{"results": [{"index": 0, ...}]}`
    where a non-greedy regex `r'\{.*?\}'` would truncate early at the first `}`.
    """
    if not raw_text:
        return None
    start = raw_text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw_text[start:idx + 1]
    return None


def _upsert_rapport_line_in_list(
    existing_facts: list[str],
    new_score: int,
    new_trend: str,
    catalyst: str = ""
):
    """
    Updates or appends the canonical `Rapport: ...` entry at the end of `existing_facts`,
    preserving the most recent meaningful `Catalyst:` reason and automatically pruning
    stale temporary grudge notes once the user's rapport recovers (`new_score >= -5`).
    """
    _, _, prev_catalyst, _ = _extract_saved_rapport_state_full(existing_facts)
    effective_catalyst = catalyst.strip() if catalyst and catalyst.strip() else prev_catalyst

    if new_score >= -5 or "Reconcil" in new_trend or "Recovering" in new_trend or "Warming" in new_trend:
        kept = []
        for item in existing_facts:
            s = str(item).strip()
            if s.lower().startswith("disposition note:") and NEGATIVE_FACT_VALENCE_REGEX.search(s):
                continue
            if TRANSIENT_BEHAVIORAL_GRUDGE_REGEX.search(s) and new_score >= 0:
                continue
            kept.append(item)
        existing_facts[:] = kept
        if effective_catalyst and NEGATIVE_FACT_VALENCE_REGEX.search(effective_catalyst) and new_score >= 0:
            effective_catalyst = "Moved past earlier friction through civil conversation"

    formatted = _format_rapport_entry(new_score, new_trend, effective_catalyst)
    for idx, item in enumerate(existing_facts):
        if RAPPORT_ENTRY_REGEX.match(str(item).strip()):
            existing_facts[idx] = formatted
            return
    existing_facts.append(formatted)


def consolidate_and_append_fact(
    new_fact: str,
    existing_facts: list[str],
    max_items: int = 35,
    overlap_threshold: float = 0.55
) -> bool:
    """
    Smart fact & relationship consolidation:
    - Silently processes `__RAPPORT_SHIFT__:<delta>|<trend>|<catalyst>` tokens and
      `[LEARN_USER: user | Disposition: <reason>]` notes to update the user's persistent
      `Rapport:` score, tier, and `Catalyst:` reason in-place.
    - If `new_fact` is a gender/pronoun correction that contradicts an older gender/pronoun
      entry, replaces the outdated entry in-place and returns True.
    - If `new_fact` overlaps >= `overlap_threshold` with an older fact and `new_fact`
      is more detailed (longer), supersedes/replaces the older fact in-place and returns True.
    - Otherwise appends `new_fact` (keeping `Rapport:` anchored at the end) and returns True.
    """
    clean_new = (new_fact or "").strip().rstrip(".")
    if not clean_new:
        return False

    # 1. Handle silent conversational/fact-valence rapport shifts
    if clean_new.startswith("__RAPPORT_SHIFT__:"):
        payload = clean_new.split("__RAPPORT_SHIFT__:", 1)[1]
        parts = payload.split("|")
        try:
            delta = int(parts[0].strip())
        except ValueError:
            delta = 0
        trend_hint = parts[1].strip() if len(parts) > 1 else "Steady"
        cat_hint = parts[2].strip() if len(parts) > 2 else ""
        old_score, _, prev_cat, _ = _extract_saved_rapport_state_full(existing_facts)
        new_score, new_trend = _apply_affinity_shift(old_score, delta, trend_hint)
        _upsert_rapport_line_in_list(existing_facts, new_score, new_trend, catalyst=(cat_hint or prev_cat))
        return False

    # 2. Handle explicit `Rapport: ... (Affinity: ...)` or `Disposition: <reason>` entries
    if RAPPORT_ENTRY_REGEX.match(clean_new):
        m = RAPPORT_ENTRY_REGEX.match(clean_new)
        sc = max(-100, min(100, int(m.group(2))))
        tr = m.group(3).strip()
        cat = (m.group(4) or "").strip()
        _upsert_rapport_line_in_list(existing_facts, sc, tr, catalyst=cat)
        return True

    disp_match = DISPOSITION_TAG_PREFIX_REGEX.match(clean_new)
    if disp_match:
        note_body = disp_match.group(1).strip()
        old_score, _, _, _ = _extract_saved_rapport_state_full(existing_facts)
        if NEGATIVE_FACT_VALENCE_REGEX.search(note_body) or any(w in note_body.lower() for w in ("cold", "annoyed", "skeptical", "prickly", "hostile", "irritated", "dislike", "rude", "scared")):
            delta, trend_h = -12, "Cooling Down"
        elif any(w in note_body.lower() for w in ("warm", "fond", "amused", "trusted", "friend", "respect", "thaw", "soften", "forgiv", "sweet", "snacks", "scratches", "competence")):
            delta, trend_h = 10, "Warming Up"
        else:
            delta, trend_h = 5, "Warming Gradually"
        new_score, new_trend = _apply_affinity_shift(old_score, delta, trend_hint=trend_h)
        _upsert_rapport_line_in_list(existing_facts, new_score, new_trend, catalyst=note_body)
        return False

    if len(clean_new) < 4 or is_ephemeral_or_banter_fact(clean_new):
        return False

    new_low = clean_new.lower()
    new_tokens = normalize_tokens(new_low)
    if not new_tokens:
        return False

    new_gender = _classify_gender_polarity(new_low)
    fact_added_or_updated = False

    for idx, existing in enumerate(existing_facts):
        ex_clean = str(existing).strip().rstrip(".")
        if RAPPORT_ENTRY_REGEX.match(ex_clean):
            continue
        ex_low = ex_clean.lower()
        if new_low == ex_low or new_low in ex_low:
            return False

        # If both facts are gender/pronoun identity facts, overwrite if contradicted or less detailed
        if new_gender:
            ex_gender = _classify_gender_polarity(ex_low)
            if ex_gender:
                if ex_gender != new_gender or len(clean_new) > len(ex_clean):
                    existing_facts[idx] = clean_new
                    fact_added_or_updated = True
                    break
                return False

        if ex_low in new_low and len(clean_new) > len(ex_clean):
            existing_facts[idx] = clean_new
            fact_added_or_updated = True
            break
        ex_tokens = normalize_tokens(ex_low)
        if ex_tokens:
            overlap = len(new_tokens & ex_tokens) / float(min(len(new_tokens), len(ex_tokens)))
            if overlap >= overlap_threshold:
                if len(clean_new) > len(ex_clean) + 6:
                    existing_facts[idx] = clean_new
                    fact_added_or_updated = True
                    break
                return False

    old_score, old_trend, prev_cat, r_idx = _extract_saved_rapport_state_full(existing_facts)
    if not fact_added_or_updated:
        if r_idx != -1:
            existing_facts.insert(r_idx, clean_new)
        else:
            existing_facts.append(clean_new)
        fact_added_or_updated = True

    if fact_added_or_updated:
        f_val = score_learned_fact_valence(clean_new)
        if f_val != 0:
            f_trend = "Warming Up — Shared Lore" if f_val > 0 else "Cooling Down — Negative Fact"
            base_before_new = 0 if r_idx == -1 else old_score
            if r_idx == -1 and len(existing_facts) > 1:
                base_before_new = sum(
                    score_learned_fact_valence(str(x))
                    for x in existing_facts[:-1]
                    if not RAPPORT_ENTRY_REGEX.match(str(x).strip())
                )
                base_before_new = max(-40, min(35, base_before_new))
            new_sc, new_tr = _apply_affinity_shift(base_before_new, f_val, f_trend)
            cat_to_use = clean_new if abs(f_val) >= 7 else prev_cat
            _upsert_rapport_line_in_list(existing_facts, new_sc, new_tr, catalyst=cat_to_use)

    if len(existing_facts) > max_items:
        _, _, _, r_idx = _extract_saved_rapport_state_full(existing_facts)
        rapport_line = existing_facts[r_idx] if r_idx != -1 else None
        non_rapport = [f for f in existing_facts if not RAPPORT_ENTRY_REGEX.match(str(f).strip())]
        if rapport_line:
            existing_facts[:] = non_rapport[-(max_items - 1):] + [rapport_line]
        else:
            existing_facts[:] = non_rapport[-max_items:]

    return fact_added_or_updated


def find_canonical_username(uname: str, user_profiles: dict, recent_users: set[str]) -> str:
    """
    Resolves a username or nickname (including display names with parenthetical pronouns
    like 'Sam (Tam/Tamself)' vs 'Sam') to its canonical key in `user_profiles` or `recent_users`.
    """
    clean = str(uname or "").strip().lstrip('@').strip()
    if not clean:
        return ""
    clean_lower = clean.lower()
    clean_base = _extract_base_username(clean)

    # 1. Exact case-insensitive match
    for existing in user_profiles.keys():
        if str(existing).lower() == clean_lower:
            return str(existing)
    for ru in recent_users:
        if ru and str(ru).lower() == clean_lower:
            return str(ru)

    # 2. Base-name match (e.g. 'Sam (Tam/Tamself)' <-> 'Sam')
    if len(clean_base) >= 3:
        for existing in user_profiles.keys():
            if _extract_base_username(str(existing)) == clean_base:
                return str(existing)
        for ru in recent_users:
            if ru and _extract_base_username(str(ru)) == clean_base:
                return str(ru)

    # 3. Safe prefix match (never merge possessive names or distinct multi-word character names)
    _generic_name_suffixes = {"bot", "ai", "app", "the", "official"}
    clean_sig_words = [w for w in clean_base.split() if w not in _generic_name_suffixes]
    if len(clean_base) >= 4 and "'s" not in clean_base and "’s" not in clean_base:
        for existing in user_profiles.keys():
            ex_base = _extract_base_username(str(existing))
            if "'s" in ex_base or "’s" in ex_base:
                continue
            ex_sig_words = [w for w in ex_base.split() if w not in _generic_name_suffixes]
            if len(clean_sig_words) != len(ex_sig_words) and (len(clean_sig_words) > 1 or len(ex_sig_words) > 1):
                continue
            if ex_base.startswith(clean_base) or clean_base.startswith(ex_base):
                return str(existing)
        for ru in recent_users:
            if ru:
                ru_base = _extract_base_username(str(ru))
                if "'s" in ru_base or "’s" in ru_base:
                    continue
                ru_sig_words = [w for w in ru_base.split() if w not in _generic_name_suffixes]
                if len(clean_sig_words) != len(ru_sig_words) and (len(clean_sig_words) > 1 or len(ru_sig_words) > 1):
                    continue
                if ru_base.startswith(clean_base) or clean_base.startswith(ru_base):
                    return str(ru)

    return clean


def _stem_token(w: str) -> str:
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


LOW_INFO_STEMS = frozenset(_stem_token(w) for w in LOW_INFORMATION_RETRIEVAL_WORDS)


def _extract_weighted_stems(text: str) -> dict[str, float]:
    """
    Extracts normalized word stems mapped to their IDF specificity weight:
    - High-specificity domain/proper nouns (not in LOW_INFORMATION_RETRIEVAL_WORDS): 2.5
    - Generic conversational words (in LOW_INFORMATION_RETRIEVAL_WORDS, e.g. 'game', 'play', 'work'): 0.25
    """
    if not text:
        return {}
    weights: dict[str, float] = {}
    for w in ALPHANUM_TOKEN_REGEX.findall(text.lower()):
        if len(w) < 4 or w in STOPWORDS or w in COMMON_CONVERSATIONAL_WORDS or w in BASE_SERVER_INDICATORS:
            continue
        st = _stem_token(w)
        if not st or len(st) < 3:
            continue
        if w in LOW_INFORMATION_RETRIEVAL_WORDS or st in LOW_INFO_STEMS:
            weights[st] = max(weights.get(st, 0.0), 0.25)
        else:
            weights[st] = max(weights.get(st, 0.0), 2.5)
    return weights


def _compute_stem_specificity_score(query_weights: dict[str, float], fact_text: str) -> float:
    """
    Computes the IDF-weighted specificity overlap score between `query_weights` and `fact_text`.
    A single specific keyword match scores 2.5 (>= 2.0 threshold), whereas a single generic
    word like 'game' only scores 0.25 (< 2.0 threshold), preventing false memory triggers!
    """
    if not query_weights or not fact_text:
        return 0.0
    fact_weights = _extract_weighted_stems(fact_text)
    if not fact_weights:
        return 0.0
    shared = set(query_weights.keys()) & set(fact_weights.keys())
    if not shared:
        return 0.0
    return sum(min(query_weights[k], fact_weights[k]) for k in shared)


def is_ephemeral_or_banter_fact(fact_str: str) -> bool:
    """Returns True if a saved string is actually a one-off chat/meme recap rather than an enduring fact."""
    if not fact_str or not isinstance(fact_str, str):
        return True
    s_clean = fact_str.strip()
    if s_clean.startswith("__RAPPORT_SHIFT__:") or RAPPORT_ENTRY_REGEX.match(s_clean):
        return False
    f_low = s_clean.lower()
    if len(f_low) < 4:
        return True
    if any(mp in f_low for mp in META_BANTER_PHRASES):
        return True
    if EPHEMERAL_USER_FACT_REGEX.search(f_low) or EPHEMERAL_CHAT_RECAP_REGEX.search(f_low):
        return True
    return False


def select_relevant_memories(
    facts_list: list[str],
    active_context_text: str,
    max_items: int = 5,
    min_specificity_score: float = 2.0
) -> list[str]:
    """
    IDF-Weighted Memory Gate:
    Filters `long_term_memory` or `general_memory` so ONLY facts that achieve an
    IDF specificity score >= `min_specificity_score` (at least one specific domain/proper
    noun match, never just a generic word like 'game' or 'work') are injected into the prompt.
    """
    if not facts_list or not isinstance(facts_list, list):
        return []

    clean_pool = [str(f).strip() for f in facts_list if f and not is_ephemeral_or_banter_fact(str(f))]
    if not clean_pool:
        return []

    if MEMORY_RECALL_QUERY_REGEX.search(active_context_text or ""):
        return clean_pool[-max_items:]

    query_weights = _extract_weighted_stems(active_context_text)
    if not query_weights:
        return []

    scored = []
    for idx, fact in enumerate(clean_pool):
        score = _compute_stem_specificity_score(query_weights, fact)
        if score >= min_specificity_score:
            scored.append((score, idx, fact))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [f for _, _, f in scored[:max_items]]


def _analyze_live_room_situation(
    raw_history: list[dict],
    current_speaker: str | None,
    combined_recent_prompt: str,
    reply_context_str: str,
    is_speaker_bot: bool = False
) -> list[str]:
    """
    Analyzes the immediate conversational situation, emotional tone, inter-bot banter,
    relationship queries, and speaker-to-target flow in the room so any bot has
    complete situational awareness.
    """
    bullets = []
    prompt_clean = (combined_recent_prompt or "").strip()
    speaker_role_tag = " (Fellow Bot)" if is_speaker_bot else ""

    # 1. Detect immediate situational mode of the current message
    if APOLOGY_RECONCILE_REGEX.search(prompt_clean) and not DIRECT_BOT_HOSTILITY_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Olive Branch / Reconciliation)**: **{current_speaker}{speaker_role_tag}** is apologizing, de-escalating, or conceding a fair point. Acknowledge it graciously in-character and let any earlier tension thaw!"
        )
    elif DIRECT_BOT_HOSTILITY_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Direct Pushback / Hostility)**: **{current_speaker}{speaker_role_tag}** is being directly confrontational or insulting. Stand your ground in-character matching your Disposition Tier."
        )
    elif SITUATIONAL_VENTING_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Situational Venting / Mishap)**: **{current_speaker}{speaker_role_tag}** is venting frustration about an external mishap (a game crash, bug, boss wipe, or hardware issue)—they are NOT attacking you! Commiserate, tease gently, or offer dry commentary matching your Disposition."
        )
    elif AFFECTION_GIFT_OR_CARE_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Affection / Shared Treat or Gesture)**: **{current_speaker}{speaker_role_tag}** is offering a friendly gesture, snack, drink, or affectionate attention. React in-character and let it warm your Disposition toward them!"
        )
    elif SITUATIONAL_SHARED_BOND_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Camaraderie & Agreement)**: **{current_speaker}{speaker_role_tag}** is agreeing with you, backing you up, or taking your side. Lean into the shared rapport!"
        )
    elif SITUATIONAL_CELEBRATION_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Shared Win / Hype Moment)**: **{current_speaker}{speaker_role_tag}** is celebrating a win, milestone, or breakthrough. Match the moment according to your Disposition Toward them!"
        )
    elif RELATIONSHIP_QUERY_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Relationship / Opinion Query)**: **{current_speaker}{speaker_role_tag}** is asking how you feel about someone in the server (or who your favorites/rivals are). Answer honestly in-character using your **Disposition Tiers**, **Archetypes**, **Catalysts**, and **Known Dossier Facts** below!"
        )
    elif SITUATIONAL_HELP_SEEKING_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Advice / Troubleshooting Request)**: **{current_speaker}{speaker_role_tag}** is asking for genuine help, explanation, or advice. Give a sharp, accurate answer wrapped in your persona's voice."
        )
    elif PLAYFUL_TEASE_REGEX.search(prompt_clean):
        bullets.append(
            f"- **Immediate Situation (Playful Banter / Teasing)**: **{current_speaker}{speaker_role_tag}** is engaging in light teasing or competitive banter. Riff back wittily in-character matching your Relationship Archetype with them!"
        )

    if is_speaker_bot:
        bullets.append(
            f"- **Bot-to-Bot Interaction Mode**: You are replying directly to **{current_speaker}**, a fellow bot in the server. Treat them as a distinct peer/character according to your Relationship Archetype with them, and keep the back-and-forth punchy!"
        )

    # 2. Check recent speaker-to-target flow across the last 5 messages
    recent_interactions = []
    for m in (raw_history or [])[-5:]:
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        a_name = m.get("author_name")
        addr_bot = m.get("addressed_bot")
        if a_name and addr_bot:
            pair_str = f"{a_name} ➔ {addr_bot}"
            if not recent_interactions or recent_interactions[-1] != pair_str:
                recent_interactions.append(pair_str)
    if len(recent_interactions) >= 2:
        bullets.append(
            f"- **Recent Conversation Flow**: {' | '.join(recent_interactions[-3:])}"
        )

    return bullets


def build_active_user_context_block(
    user_profiles: dict,
    raw_history: list[dict],
    current_speaker: str | None = None,
    combined_recent_prompt: str = "",
    reply_context_str: str = "",
    is_spontaneous: bool = False,
    persona_prompt: str = "",
    other_bot_names: set[str] | list[str] | None = None,
    my_bot_name: str = ""
) -> str:
    """
    Builds a comprehensive, 100% generic Situational & Relationship Radar for the system prompt:
    1. Analyzes the live room situation (`_analyze_live_room_situation`) including camaraderie,
       affection/gifts, playful banter, venting, celebrations, apologies, and relationship queries.
    2. Computes the bot's live Disposition Tier (-100 to +100 Affinity), Relationship Archetype
       (Human vs. Fellow Bot), Trend, and Last Catalyst toward `current_speaker` AND toward up to
       4 other active participants or fellow bots in the room.
    3. Detects Triadic Loyalty Alerts (when someone is rude to a friend/bot you like), Shared Rivalry
       Amusement (when someone teases a rival), and Social Contrast between people in the room.
    """
    if not isinstance(user_profiles, dict):
        user_profiles = {}

    my_base = _extract_base_username(my_bot_name) if my_bot_name else ""

    # Track which base names belong to fellow bots in the server (from history + active bot registry)
    known_bot_bases: set[str] = set()
    base_to_display: dict[str, str] = {}
    for ob in (other_bot_names or []):
        if ob:
            ob_b = _extract_base_username(str(ob))
            if ob_b and ob_b != my_base:
                known_bot_bases.add(ob_b)
                base_to_display.setdefault(ob_b, str(ob))

    for m in (raw_history or []):
        if isinstance(m, dict) and m.get("author_is_bot") and m.get("author_name"):
            mb = _extract_base_username(str(m["author_name"]))
            if mb and mb != my_base:
                known_bot_bases.add(mb)
                base_to_display.setdefault(mb, str(m["author_name"]))

    base_to_facts: dict[str, list[str]] = {}
    for prof_user, facts_list in user_profiles.items():
        if not prof_user or not isinstance(facts_list, list) or not facts_list:
            continue
        b_name = _extract_base_username(str(prof_user))
        if not b_name or b_name == my_base:
            continue
        base_to_display.setdefault(b_name, str(prof_user))
        bucket = base_to_facts.setdefault(b_name, [])
        for f_item in facts_list:
            f_str = str(f_item).strip()
            if not f_str:
                continue
            if RAPPORT_ENTRY_REGEX.match(f_str):
                bucket[:] = [x for x in bucket if not RAPPORT_ENTRY_REGEX.match(x)] + [f_str]
                continue
            if not is_ephemeral_or_banter_fact(f_str) and not is_duplicate_fact(f_str, bucket, threshold=0.75):
                bucket.append(f_str)

    speaker_base = _extract_base_username(current_speaker) if current_speaker else ""
    if speaker_base and current_speaker:
        base_to_display[speaker_base] = current_speaker
    is_speaker_fellow_bot = bool(speaker_base and speaker_base in known_bot_bases)

    # Discover other recently active speakers in the room (both humans and fellow bots in the last 10 messages)
    recent_room_bases: list[str] = []
    for m in reversed((raw_history or [])[-10:]):
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        a_name = str(m.get("author_name", "") or "").strip()
        if not a_name:
            continue
        b_name = _extract_base_username(a_name)
        if b_name and b_name != speaker_base and b_name != my_base:
            base_to_display.setdefault(b_name, a_name)
            if b_name not in recent_room_bases:
                recent_room_bases.append(b_name)

    active_blob = f"{combined_recent_prompt or ''} {reply_context_str or ''}".strip()
    search_blob_low = active_blob.lower()
    is_rel_query = bool(RELATIONSHIP_QUERY_REGEX.search(active_blob))
    is_recall_q = bool(MEMORY_RECALL_QUERY_REGEX.search(active_blob) or is_rel_query)

    for b_name, disp in list(base_to_display.items()):
        if b_name != speaker_base and b_name != my_base and len(b_name) >= 3 and b_name not in recent_room_bases:
            if b_name in search_blob_low or disp.lower() in search_blob_low:
                recent_room_bases.append(b_name)

    # If the user is asking a relationship/opinion question ("who is your favorite?", "what do you think of the other bots?"),
    # include known fellow bots and top-profiled members even if they haven't spoken in the last 10 messages!
    if is_rel_query:
        for b_name in list(known_bot_bases) + list(base_to_facts.keys()):
            if b_name and b_name != speaker_base and b_name != my_base and b_name not in recent_room_bases:
                recent_room_bases.append(b_name)

    query_weights = _extract_weighted_stems(active_blob)

    def _compute_live_user_rapport(b_name: str, all_entries: list[str], is_active_speaker: bool) -> tuple[int, str, str, bool]:
        disp_name = base_to_display.get(b_name, b_name)
        saved_sc, saved_tr, saved_cat, r_idx = _extract_saved_rapport_state_full(
            all_entries, persona_prompt=persona_prompt, target_name=disp_name
        )
        had_friction = (saved_sc < 0) or any(
            NEGATIVE_FACT_VALENCE_REGEX.search(x) for x in all_entries if not RAPPORT_ENTRY_REGEX.match(x)
        )
        if r_idx == -1 and raw_history:
            prompt_norm = _strip_metadata_and_speaker_headers(combined_recent_prompt).lower() if is_active_speaker else ""
            user_recent_msgs = [
                m.get("content", "")
                for m in raw_history[-16:]
                if isinstance(m, dict)
                and _extract_base_username(str(m.get("author_name", ""))) == b_name
                and isinstance(m.get("content"), str)
            ]
            for hist_m in user_recent_msgs:
                clean_h = _strip_metadata_and_speaker_headers(hist_m)
                if not clean_h:
                    continue
                if prompt_norm and clean_h.lower() in prompt_norm:
                    continue
                d_val, t_val = score_message_rapport_delta(clean_h, current_affinity=saved_sc)
                if d_val != 0:
                    if d_val < 0:
                        had_friction = True
                    saved_sc, saved_tr = _apply_affinity_shift(saved_sc, d_val, t_val)

        if is_active_speaker and combined_recent_prompt:
            cur_d, cur_t = score_message_rapport_delta(combined_recent_prompt, current_affinity=saved_sc)
            if cur_d != 0:
                if cur_d < 0:
                    had_friction = True
                saved_sc, saved_tr = _apply_affinity_shift(saved_sc, cur_d, cur_t)

        return saved_sc, saved_tr, saved_cat, had_friction

    def _pick_relevant_user_facts(
        all_entries: list[str],
        live_score: int,
        max_count: int = 4,
        include_core_identity: bool = False
    ) -> list[str]:
        non_rapport = [x for x in (all_entries or []) if not RAPPORT_ENTRY_REGEX.match(x) and not x.lower().startswith("disposition note:")]
        if not non_rapport:
            return []

        if live_score >= 0:
            non_rapport = [x for x in non_rapport if not TRANSIENT_BEHAVIORAL_GRUDGE_REGEX.search(x)]

        if is_recall_q and not is_spontaneous:
            return non_rapport[-max_count:]

        core_facts = []
        if include_core_identity:
            for f_str in non_rapport:
                if CORE_IDENTITY_FACT_REGEX.search(f_str) and f_str not in core_facts:
                    core_facts.append(f_str)
            core_facts = core_facts[-3:]

        scored = []
        for idx, f_str in enumerate(non_rapport):
            if f_str in core_facts:
                continue
            sc = _compute_stem_specificity_score(query_weights, f_str)
            if sc >= 1.5:
                scored.append((sc, idx, f_str))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        topical_matches = [f for _, _, f in scored[:max_count]]

        recent_situational = []
        if include_core_identity and not is_spontaneous and len(topical_matches) < 2:
            for f_str in reversed(non_rapport[-3:]):
                if f_str not in core_facts and f_str not in topical_matches and f_str not in recent_situational:
                    recent_situational.append(f_str)
                if len(recent_situational) >= 2:
                    break

        return (core_facts + topical_matches + recent_situational)[:max_count + 2]

    lines = []
    situational_bullets = _analyze_live_room_situation(
        raw_history, current_speaker, combined_recent_prompt, reply_context_str, is_speaker_bot=is_speaker_fellow_bot
    )
    if situational_bullets:
        lines.extend(situational_bullets)

    speaker_live_sc = 0
    if current_speaker and speaker_base:
        sp_entries = base_to_facts.get(speaker_base, [])
        live_sc, live_tr, live_cat, had_fric = _compute_live_user_rapport(speaker_base, sp_entries, is_active_speaker=True)
        speaker_live_sc = live_sc
        tier_label = _get_disposition_tier_label(live_sc)
        sp_archetype = _derive_relationship_archetype(
            live_sc, live_tr, is_fellow_bot=is_speaker_fellow_bot, had_past_friction=had_fric
        )
        sign_sc = f"+{live_sc}" if live_sc > 0 else str(live_sc)
        cat_suffix = f" | Catalyst: {live_cat}" if live_cat else ""
        behav_guide = _get_disposition_behavioral_guidance(
            live_sc, live_tr, had_past_friction=had_fric, catalyst=live_cat,
            is_fellow_bot=is_speaker_fellow_bot, archetype=sp_archetype
        )
        sp_facts = _pick_relevant_user_facts(
            sp_entries, live_score=live_sc, max_count=(3 if is_spontaneous else 5), include_core_identity=True
        )
        speaker_badge = " [Fellow Server Bot]" if is_speaker_fellow_bot else ""
        lines.append(f"- **Current Speaker ({current_speaker}{speaker_badge})**:")
        lines.append(
            f"  • **Your Disposition Toward {current_speaker}**: {tier_label} — *{sp_archetype}* (Affinity: {sign_sc}/100 | Trend: {live_tr}{cat_suffix})"
        )
        lines.append(f"  • **Attitude & Dynamic Guidance**: {behav_guide}")
        if sp_facts:
            lines.append(f"  • **Known Dossier Facts**: {'; '.join(sp_facts)}")

    # Include up to 4 other active room participants or mentioned members/bots so the bot has full room awareness!
    room_participants_lines = []
    social_contrast_notes = []
    loyalty_alerts = []
    max_other_participants = 5 if is_rel_query else 4

    for b_name in recent_room_bases[:max_other_participants]:
        disp = base_to_display.get(b_name, b_name)
        is_fellow_b = (b_name in known_bot_bases)
        is_explicitly_mentioned = (b_name in search_blob_low or disp.lower() in search_blob_low)
        u_entries = base_to_facts.get(b_name, [])
        u_sc, u_tr, u_cat, u_had_fric = _compute_live_user_rapport(b_name, u_entries, is_active_speaker=False)
        if not u_entries and not is_explicitly_mentioned and not is_rel_query and u_sc == 0 and not u_cat:
            continue
        u_tier = _get_disposition_tier_label(u_sc)
        u_arch = _derive_relationship_archetype(u_sc, u_tr, is_fellow_bot=is_fellow_b, had_past_friction=u_had_fric)
        u_sign = f"+{u_sc}" if u_sc > 0 else str(u_sc)
        u_facts = _pick_relevant_user_facts(
            u_entries, live_score=u_sc, max_count=(3 if (is_explicitly_mentioned or is_rel_query) else 2), include_core_identity=True
        )
        cat_str = f" | Catalyst: {u_cat}" if (u_cat and (is_explicitly_mentioned or is_rel_query or is_fellow_b)) else ""
        fact_suffix = f" | Facts: {'; '.join(u_facts)}" if u_facts else ""
        peer_type = "Fellow Bot" if is_fellow_b else "Member"
        role_label = f"{peer_type} — Mentioned in Turn" if is_explicitly_mentioned else f"{peer_type} in Room"
        room_participants_lines.append(
            f"  • **{disp}** ({role_label}): {u_tier} — *{u_arch}* ({u_sign}/100, {u_tr}{cat_str}){fact_suffix}"
        )

        # Check for Triadic Loyalty Alert: Is the current speaker being hostile/mean toward a friend/companion in the room?
        if (
            is_explicitly_mentioned
            and u_sc >= 15
            and not SITUATIONAL_VENTING_REGEX.search(combined_recent_prompt or "")
            and (
                DIRECT_BOT_HOSTILITY_REGEX.search(combined_recent_prompt or "")
                or GENERAL_HARSH_TONE_REGEX.search(combined_recent_prompt or "")
                or THIRD_PARTY_HOSTILITY_REGEX.search(combined_recent_prompt or "")
            )
        ):
            loyalty_alerts.append(
                f"  • **PROTECTIVE LOYALTY TRIGGER**: **{current_speaker}** is being hostile or critical toward **{disp}**, whom you are fond of ({u_tier}, {u_sign}/100)! Back up or defend **{disp}** in-character!"
            )
        elif is_explicitly_mentioned and u_sc >= 20 and (
            THIRD_PARTY_DEFENSE_OR_PRAISE_REGEX.search(combined_recent_prompt or "")
            or AFFECTION_GIFT_OR_CARE_REGEX.search(combined_recent_prompt or "")
        ):
            loyalty_alerts.append(
                f"  • **SHARED BOND TRIGGER**: **{current_speaker}** is being kind, affectionate, or praising toward **{disp}**, whom you are fond of ({u_tier}, {u_sign}/100)! Show warm approval in-character."
            )
        elif is_explicitly_mentioned and u_sc <= -10 and PLAYFUL_TEASE_REGEX.search(combined_recent_prompt or ""):
            loyalty_alerts.append(
                f"  • **SHARED RIVALRY AMUSEMENT**: **{current_speaker}** is teasing or roasting **{disp}** ({u_arch}, {u_sign}/100). Feel free to join in on the playful roast or smirk in approval!"
            )

        # Check for Social Contrast between the Current Speaker and another active person/bot in the room
        if current_speaker and abs(u_sc - speaker_live_sc) >= 25:
            if u_sc > speaker_live_sc:
                social_contrast_notes.append(
                    f"You are much fonder of **{disp}** ({u_tier}, {u_sign}/100) than **{current_speaker}** ({speaker_live_sc:+d}/100)—let that contrast subtly show if both are part of the exchange."
                )
            else:
                social_contrast_notes.append(
                    f"You are much fonder of **{current_speaker}** ({speaker_live_sc:+d}/100) than **{disp}** ({u_tier}, {u_sign}/100)—you naturally take **{current_speaker}**'s side in banter."
                )

    if room_participants_lines:
        lines.append("- **Other Active Participants & Fellow Bots in Room**:")
        lines.extend(room_participants_lines)
    if loyalty_alerts:
        lines.extend(loyalty_alerts)
    if social_contrast_notes:
        lines.append(f"- **Room Relationship Contrast**: {social_contrast_notes[0]}")

    if not lines:
        return ""
    return (
        "\n\n### Live Situational Awareness, Room Social Graph & Dossier (PASSIVE REFERENCE ONLY — "
        "let your Disposition, Archetype, and the Immediate Situation shape your tone naturally, and NEVER shoehorn unrelated saved facts!):\n"
        + "\n".join(lines)
    )


def is_misattributed_hearsay(fact: str, target_user: str, speaker_name: str, user_message: str) -> bool:
    """
    Deterministic filter that catches third-party reported speech / hearsay
    and ephemeral one-off chat actions (like 'asked for an image').
    """
    if not fact or not user_message:
        return False

    if fact.startswith("__RAPPORT_SHIFT__:") or RAPPORT_ENTRY_REGEX.match(fact) or DISPOSITION_TAG_PREFIX_REGEX.match(fact):
        return False

    if FACT_HEARSAY_PREFIX_REGEX.search(fact) or EPHEMERAL_USER_FACT_REGEX.search(fact):
        return True

    msg_clean = _strip_metadata_and_speaker_headers(user_message)

    # If the candidate fact is attributed to the current speaker without an explicit first-person
    # pronoun (e.g., subject-drop like "Turned off vsync and crashed" or "Playing WoW Forever tonight"),
    # only block it if the message is a pure question directed at the bot with no self-statement!
    if _extract_base_username(target_user) == _extract_base_username(speaker_name):
        if not FIRST_PERSON_SELF_REF_REGEX.search(msg_clean):
            if "?" in msg_clean and re.search(r"\b(?:you|your|yours|ur)\b", msg_clean, re.IGNORECASE):
                return True
    else:
        return False

    report_matches = list(THIRD_PARTY_REPORT_REGEX.finditer(msg_clean))
    if not report_matches:
        return False

    fact_tokens = normalize_tokens(fact)
    if not fact_tokens:
        return True

    first_report_start = report_matches[0].start()
    before_report_text = msg_clean[:first_report_start]
    reported_clause_text = msg_clean[first_report_start:]

    reported_tokens = normalize_tokens(reported_clause_text)
    before_tokens = normalize_tokens(before_report_text)

    overlap_with_reported = len(fact_tokens & reported_tokens) / float(len(fact_tokens))
    overlap_with_before = len(fact_tokens & before_tokens) / float(len(fact_tokens)) if before_tokens else 0.0

    if overlap_with_reported >= 0.45 and overlap_with_before < 0.40:
        return True

    if INDIRECT_HEARSAY_CHAIN_REGEX.search(msg_clean):
        if overlap_with_reported >= 0.30:
            return True

    return False


async def verify_and_resolve_user_facts(
    candidate_facts: list[tuple[str, str]],
    speaker_name: str,
    recent_user_prompt: str,
    raw_history: list[dict],
    user_profiles: dict,
    recent_users: set[str],
    other_bot_names: set[str],
    my_bot_name: str,
    api_key: str,
    text_model: str,
    api_url: str
) -> list[tuple[str, str]]:
    """
    Verifies candidate (target_user, fact) pairs extracted from [LEARN_USER: user | fact]
    and folds both conversational rapport shifts (`__RAPPORT_SHIFT__`) and learned-fact
    valence into the speaker's persistent Relationship & Disposition score.
    """
    pre_filtered: list[tuple[str, str]] = []
    rapport_shifts: dict[str, tuple[int, str, str]] = {}
    explicit_disp_notes: list[tuple[str, str]] = []
    my_bot_lower = my_bot_name.lower()

    for raw_uname, raw_fact in candidate_facts:
        canonical_uname = find_canonical_username(raw_uname, user_profiles, recent_users)
        clean_fact = raw_fact.strip().rstrip(".")
        if not canonical_uname or not clean_fact:
            continue
        # Only block a bot from saving a user dossier about ITSELF; allow learning dossiers about other bots!
        if canonical_uname.lower() == my_bot_lower or _extract_base_username(canonical_uname) == _extract_base_username(my_bot_name):
            continue

        if clean_fact.startswith("__RAPPORT_SHIFT__:"):
            payload = clean_fact.split("__RAPPORT_SHIFT__:", 1)[1]
            parts = payload.split("|")
            try:
                d_val = int(parts[0].strip())
            except ValueError:
                d_val = 0
            t_val = parts[1].strip() if len(parts) > 1 else "Steady"
            c_val = parts[2].strip() if len(parts) > 2 else ""
            prev_d, prev_t, prev_c = rapport_shifts.get(canonical_uname, (0, "Steady", ""))
            rapport_shifts[canonical_uname] = (
                prev_d + d_val,
                t_val if t_val != "Steady" else prev_t,
                c_val if c_val else prev_c
            )
            continue

        if RAPPORT_ENTRY_REGEX.match(clean_fact) or DISPOSITION_TAG_PREFIX_REGEX.match(clean_fact):
            explicit_disp_notes.append((canonical_uname, clean_fact))
            continue

        if is_misattributed_hearsay(clean_fact, canonical_uname, speaker_name, recent_user_prompt):
            print(f"[{my_bot_name}] [MEMORY FILTER] Blocked hearsay/ephemeral fact for {canonical_uname}: '{clean_fact}'")
            continue
        pre_filtered.append((canonical_uname, clean_fact))

    def _finalize_with_rapport(verified_regular: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out = list(verified_regular) + list(explicit_disp_notes)
        for r_user, (r_delta, r_trend, r_cat) in rapport_shifts.items():
            if r_delta != 0:
                out.append((r_user, f"__RAPPORT_SHIFT__:{r_delta}|{r_trend}|{r_cat}"))
        return out

    if not pre_filtered:
        return _finalize_with_rapport([])

    # Fast-Path (0 API calls): If every candidate fact is about the current speaker,
    # the user's message contains direct first-person self-reference ("I", "my", "me", "I'm"),
    # there are no third-party report cues ("he said", "they told me"), and the fact shares
    # content tokens with what the speaker actually typed, approve deterministically in 0ms.
    speaker_base = _extract_base_username(speaker_name)
    msg_clean_for_fp = _strip_metadata_and_speaker_headers(recent_user_prompt or "")
    has_first_person = bool(FIRST_PERSON_SELF_REF_REGEX.search(msg_clean_for_fp))
    has_third_party = bool(THIRD_PARTY_REPORT_REGEX.search(msg_clean_for_fp))
    prompt_tokens = normalize_tokens(msg_clean_for_fp)

    # Include tokens from the last 2 messages in raw_history so facts about a bot or user
    # mentioned in the immediate exchange can also fast-verify in 0ms!
    recent_hist_text = " ".join(
        str(m.get("content", "")) for m in (raw_history or [])[-2:]
        if isinstance(m, dict) and m.get("role") != "system"
    )
    combined_verify_tokens = _expand_tokens_with_synonyms(
        prompt_tokens | normalize_tokens(recent_hist_text)
    )
    if not has_third_party and combined_verify_tokens:
        fast_verified: list[tuple[str, str]] = []
        needs_llm_check = False
        for canon_u, c_fact in pre_filtered:
            u_base = _extract_base_username(canon_u)
            is_about_speaker = (u_base == speaker_base)
            is_explicitly_named = (
                len(u_base) >= 3
                and (u_base in msg_clean_for_fp.lower() or u_base in recent_hist_text.lower())
            )
            if not (is_about_speaker or is_explicitly_named):
                needs_llm_check = True
                break
            fact_tokens = _expand_tokens_with_synonyms(normalize_tokens(c_fact))
            if not fact_tokens or len(fact_tokens & combined_verify_tokens) == 0:
                needs_llm_check = True
                break
            fast_verified.append((canon_u, c_fact))
        if not needs_llm_check and fast_verified:
            return _finalize_with_rapport(fast_verified)

    context_lines = []
    for m in (raw_history or [])[-6:]:
        if m.get("role") == "system":
            continue
        c = m.get("content", "")
        if isinstance(c, str) and c.strip():
            context_lines.append(c.strip()[:260])
    if recent_user_prompt and recent_user_prompt.strip() not in "\n".join(context_lines):
        context_lines.append(f"[{speaker_name}]: {recent_user_prompt.strip()[:260]}")

    context_block = "\n".join(context_lines[-6:])
    known_human_users = sorted({
        u for u in (set(user_profiles.keys()) | set(recent_users) | set(other_bot_names or set()))
        if u and str(u).lower() != my_bot_lower and _extract_base_username(str(u)) != _extract_base_username(my_bot_name)
    })

    candidates_json = json.dumps([
        {"index": idx, "target_user": u, "fact": f}
        for idx, (u, f) in enumerate(pre_filtered)
    ], ensure_ascii=False)

    verify_prompt = (
        "You are a strict memory attribution verifier for a Discord server.\n"
        f"Current Speaker: {speaker_name}\n"
        f"Known Server Members: {', '.join(known_human_users) if known_human_users else speaker_name}\n\n"
        f"Recent Conversation:\n{context_block}\n\n"
        f"Candidate Facts to Verify:\n{candidates_json}\n\n"
        "RULES:\n"
        "1. Mark status as \"VALID\" if `target_user` directly stated, owned, or confirmed `fact` about themselves, OR if the conversation establishes `fact` as a real trait, role, project, or notable situation of `target_user` (including fellow server bots).\n"
        "2. If the speaker was quoting unverified third-party hearsay (e.g., \"someone said...\", \"they told me...\") and attributing it to the wrong person, mark status as \"REJECT\".\n"
        "3. If the statement is clearly a true fact about a DIFFERENT explicitly named Known Server Member, mark status as \"REATTRIBUTE\" and set `target_user` to that member's exact name.\n"
        "4. If the statement is a one-time media command (e.g., 'asked for an image'), mark status as \"REJECT\".\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\"results\": [{\"index\": 0, \"status\": \"VALID\", \"target_user\": \"Name\", \"fact\": \"fact text\"}]}"
    )

    payload = {
        "model": text_model,
        "messages": [
            {"role": "system", "content": "Output only valid JSON."},
            {"role": "user", "content": verify_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 180
    }

    res = await post_with_backoff(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json_payload=payload,
        timeout=15,
        max_retries=1,
        bot_name=my_bot_name
    )

    if res and res.status_code == 200:
        try:
            raw_content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            outer_json = _extract_outer_json_object(raw_content)
            if outer_json:
                parsed = json.loads(outer_json).get("results", [])
                verified_list: list[tuple[str, str]] = []
                for item in parsed:
                    status = str(item.get("status", "")).upper()
                    t_user = str(item.get("target_user", "")).strip()
                    f_text = str(item.get("fact", "")).strip().rstrip(".")
                    if status == "REJECT" or not t_user or not f_text:
                        print(f"[{my_bot_name}] [MEMORY VERIFIER] Rejected misattributed/ephemeral fact: '{f_text}'")
                        continue
                    if status in ("VALID", "REATTRIBUTE"):
                        canon_target = find_canonical_username(t_user, user_profiles, recent_users)
                        if canon_target.lower() in ("they", "them", "he", "him", "she", "her", "someone", "friend", "unknown"):
                            continue
                        if canon_target.lower() == my_bot_lower or _extract_base_username(canon_target) == _extract_base_username(my_bot_name):
                            continue
                        verified_list.append((canon_target, f_text))
                return _finalize_with_rapport(verified_list)
        except Exception as e:
            print(f"[{my_bot_name}] [WARN] Memory verifier parse fallback: {e}")

    return _finalize_with_rapport([])


def _extract_bio_example_quotes(system_prompt_text: str) -> list[str]:
    """
    Scans the bot's base persona prompt for quoted sample dialogue lines
    (e.g., '"Levi-oh-sa, not Levi-o-za..."') so the anti-repetition block can
    explicitly forbid verbatim parroting of sample bio lines in live chat.
    """
    if not system_prompt_text:
        return []
    base_section = system_prompt_text.split("### Current Time")[0]
    found = []
    for m in BIO_EXAMPLE_QUOTE_REGEX.finditer(base_section):
        q = m.group(1).strip()
        if any(tag in q for tag in ("[IMAGE_GEN", "[MEME_GEN", "[VOICE_MSG", "[GIF", "[LEARN", "Role:", "Scene:")):
            continue
        words = q.split()
        if 4 <= len(words) <= 22:
            short_q = " ".join(words[:8])
            if short_q not in found:
                found.append(short_q)
            if len(found) >= 4:
                break
    return found


def _extract_repeated_phrase_templates(own_texts: list[str]) -> list[str]:
    """
    Detects 4-word to 6-word conversational clichés that the bot has repeated
    across 2+ of its own recent messages OR distinctive clauses from its very last
    message so the bot never echoes its own previous turn verbatim.
    """
    if not own_texts:
        return []

    ngram_doc_counts = Counter()
    for txt in own_texts[-6:]:
        cleaned = INLINE_EMOTE_STRIP_REGEX.sub(' ', txt.lower())
        words = WORD_APOSTROPHE_REGEX.findall(cleaned)
        seen_in_msg = set()
        for n in (6, 5, 4):
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i:i + n])
                    if len(gram) >= 15:
                        seen_in_msg.add(gram)
        for g in seen_in_msg:
            ngram_doc_counts[g] += 1

    candidates = [
        (count, len(gram.split()), len(gram), gram)
        for gram, count in ngram_doc_counts.items()
        if count >= 2
    ]
    candidates.sort(reverse=True)

    filtered = []
    for _, _, _, g in candidates:
        g_words = set(g.split())
        if any(
            g in other or other in g or (len(g_words & set(other.split())) / float(min(len(g_words), len(other.split()))) >= 0.65)
            for other in filtered
        ):
            continue
        filtered.append(g)
        if len(filtered) >= 4:
            break

    # Also grab 1-2 distinctive 5-word clauses from the very last message so the bot
    # cannot copy-paste its immediately preceding line (as seen in the waffle/syrup loop)
    if len(filtered) < 4 and own_texts:
        last_clean = INLINE_EMOTE_STRIP_REGEX.sub(' ', own_texts[-1].lower())
        last_words = WORD_APOSTROPHE_REGEX.findall(last_clean)
        if len(last_words) >= 5:
            mid_idx = max(0, (len(last_words) // 2) - 2)
            sample_gram = " ".join(last_words[mid_idx:mid_idx + 5])
            if len(sample_gram) >= 15 and not any(sample_gram in f or f in sample_gram for f in filtered):
                filtered.append(sample_gram)

    return filtered[:4]


def _detect_overused_or_banned_topics(
    own_recent_texts: list[str],
    recent_user_texts: list[str],
    bot_display_name: str
) -> tuple[list[str], bool]:
    """
    1. Checks if a human explicitly asked to change the subject or stop talking about X
       (e.g. 'can you bots talk about something besides toaster related items?').
    2. Checks if the bot has fixated on the same specific nouns/keywords across 2+ of its
       last 4 turns (e.g. 'waffle', 'syrup', 'breakfast', 'latin', 'cursed', 'relic').
    Returns (list_of_overused_or_banned_words, user_requested_topic_change).
    """
    banned_words = []
    user_requested_pivot = False

    # 1. Inspect recent human messages for explicit topic-change requests
    for u_txt in recent_user_texts[-4:]:
        m = TOPIC_CHANGE_REQUEST_REGEX.search(u_txt)
        if m:
            user_requested_pivot = True
            for grp in m.groups():
                if grp:
                    for tok in ALPHANUM_TOKEN_REGEX.findall(grp.lower()):
                        if len(tok) >= 3 and tok not in STOPWORDS and tok not in COMMON_CONVERSATIONAL_WORDS:
                            if tok not in banned_words:
                                banned_words.append(tok)

    # 2. Inspect the bot's own last 4 messages (including emotes & captions) for topic fixation
    bot_name_parts = set(ALPHANUM_TOKEN_REGEX.findall(bot_display_name.lower()))
    word_turn_counts = Counter()
    window = own_recent_texts[-4:]
    for b_txt in window:
        tokens = {
            w for w in ALPHANUM_TOKEN_REGEX.findall(b_txt.lower())
            if len(w) >= 4
            and w not in STOPWORDS
            and w not in COMMON_CONVERSATIONAL_WORDS
            and w not in bot_name_parts
        }
        for t in tokens:
            word_turn_counts[t] += 1

    # Prioritize words repeated across >= 2 turns first; if user explicitly asked to pivot,
    # also include top single-turn topic nouns from the bot's most recent turn.
    min_count = 1 if user_requested_pivot else 2
    for word, cnt in word_turn_counts.most_common(10):
        if cnt >= min_count and word not in banned_words:
            banned_words.append(word)

    return banned_words[:8], user_requested_pivot


def build_api_messages_from_history(
    dynamic_system: dict,
    raw_history: list[dict],
    bot_user_id: int,
    bot_display_name: str
) -> list[dict]:
    """
    Builds clean conversation history for the LLM, inserting session-gap time markers
    when >= 45 minutes elapsed between messages, annotating when a user message was
    addressed to a DIFFERENT bot, tracking the bot's own recent emotes, openers,
    stock phrases, bio example quotes, overused topic nouns, and meme captions to
    prevent repetitive loops, and injecting organic human-like spontaneity cues.
    """
    api_messages = [dynamic_system]
    seen_msg_ids = set()
    own_recent_openers = []
    own_recent_texts = []
    recent_human_texts = []
    own_emote_counter = Counter()
    recent_meme_captions = []
    active_human_speakers = []
    bot_name_lower = bot_display_name.lower()
    last_msg_ts: float | None = None

    for idx, msg in enumerate(raw_history):
        msg_id = msg.get("message_id")
        if msg_id and msg_id in seen_msg_ids:
            continue
        if msg_id:
            seen_msg_ids.add(msg_id)

        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        if role == "system":
            if idx >= len(raw_history) - 10:
                api_messages.append({"role": "system", "content": content})
            continue

        # Detect long silence gaps (>= 45 mins) between turns and inject a session boundary marker
        msg_ts = msg.get("timestamp")
        if isinstance(msg_ts, (int, float)):
            if last_msg_ts is not None and (msg_ts - last_msg_ts) >= SESSION_GAP_SECONDS:
                gap_label = format_duration_gap(msg_ts - last_msg_ts)
                api_messages.append({
                    "role": "system",
                    "content": (
                        f"[System Time Marker: {gap_label} of silence passed — A new conversation session started here. "
                        "Treat earlier messages above as past history and do NOT resume old bits or stale jokes from earlier unless explicitly asked.]"
                    )
                })
                # Clear topic-fixation and emote windows across session boundaries so yesterday's
                # conversation doesn't trigger false anti-obsession warnings in today's session
                recent_human_texts.clear()
                own_recent_texts.clear()
                own_emote_counter.clear()
            last_msg_ts = float(msg_ts)

        is_self = (
            msg.get("author_id") == bot_user_id
            or (role == "assistant" and msg.get("author_name") == bot_display_name)
        )

        if is_self:
            own_text = content
            glitch_summary = extract_glitch_summary(own_text)
            img_note_match = FALLBACK_IMAGE_NOTE_REGEX.search(own_text)
            saved_img_prompt = msg.get("img_prompt") or (img_note_match.group(1) if img_note_match else None)
            saved_meme_spec = msg.get("meme_spec")
            saved_gif_query = msg.get("gif_query")

            if saved_meme_spec:
                _, t_cap, b_cap = parse_meme_gen_tag(saved_meme_spec)
                for cap in (t_cap, b_cap):
                    if cap and cap.upper() not in recent_meme_captions:
                        recent_meme_captions.append(cap.upper())

            own_text = ERROR_REPLY_STRIP_REGEX.sub('', own_text).strip()
            own_text = SYSTEM_NOTE_INLINE_REGEX.sub('', own_text).strip()
            own_text = LEARNED_INLINE_REGEX.sub('', own_text).strip()
            own_text = IMAGE_DESC_STRIP_REGEX.sub('', own_text).strip()

            if not own_text and not saved_img_prompt and not saved_meme_spec and not saved_gif_query:
                if glitch_summary and idx >= len(raw_history) - 10:
                    api_messages.append({
                        "role": "system",
                        "content": (
                            f"[System Glitch Event: {bot_display_name} experienced a glitch ({glitch_summary}). "
                            "You may acknowledge or joke about it in-character, but NEVER output raw '*(API Error...)*' text.]"
                        )
                    })
                continue

            if idx >= len(raw_history) - 12 and own_text:
                topic_sample = f"{own_text} {saved_meme_spec or ''} {saved_img_prompt or ''}".strip()
                own_recent_texts.append(topic_sample)
                for em in extract_used_emotes(own_text):
                    own_emote_counter[em] += 1
                spoken_only = LEADING_EMOTE_STRIP_REGEX.sub('', own_text).strip()
                words_raw = own_text.split()
                words_spoken = spoken_only.split()
                if len(words_raw) >= 3:
                    own_recent_openers.append(" ".join(words_raw[:4]).lower())
                if len(words_spoken) >= 3:
                    sp_opener = " ".join(words_spoken[:4]).lower()
                    if sp_opener not in own_recent_openers:
                        own_recent_openers.append(sp_opener)

            # Do NOT re-inject `[IMAGE_GEN: ...]` or `[MEME_GEN: ...]` tags into past assistant
            # turns in history—leaving old media tags in assistant turns causes the LLM to
            # spam unsolicited memes on every turn and fixate on the old visual prompt!
            if own_text:
                api_messages.append({"role": "assistant", "content": own_text})
            if glitch_summary and idx >= len(raw_history) - 10:
                api_messages.append({
                    "role": "system",
                    "content": f"[System Glitch Event: {bot_display_name}'s media module glitched ({glitch_summary}) on that turn.]"
                })
        else:
            other_text = content
            author_name = msg.get("author_name", "User")
            if not msg.get("author_is_bot"):
                if author_name and author_name not in active_human_speakers:
                    active_human_speakers.append(author_name)
                if idx >= len(raw_history) - 8:
                    recent_human_texts.append(other_text)

            addressed_bot = msg.get("addressed_bot")
            glitch_summary = extract_glitch_summary(other_text)
            other_text = ERROR_REPLY_STRIP_REGEX.sub('', other_text).strip()
            other_text = OTHER_BOT_ATTACHED_IMG_NOTE_REGEX.sub(
                r'[\1 attached a generated image]',
                other_text
            )
            other_text = SYSTEM_NOTE_INLINE_REGEX.sub('', other_text).strip()
            other_text = LEARNED_INLINE_REGEX.sub('', other_text).strip()

            if not other_text or EMPTY_SPEAKER_TAG_REGEX.fullmatch(other_text):
                if glitch_summary and idx >= len(raw_history) - 10:
                    api_messages.append({
                        "role": "system",
                        "content": (
                            f"[System Glitch Event: {author_name} just glitched out ({glitch_summary}). "
                            "You may tease, roast, or comment on it in-character, but NEVER output raw '*(API Error...)*' text.]"
                        )
                    })
                continue

            if msg.get("author_is_bot"):
                other_text = OTHER_BOT_MEDIA_DESC_REGEX.sub(
                    r'[Attached Media Description (already generated by another bot — DO NOT copy this visual scene): \1]',
                    other_text
                )

            if not other_text.startswith(f"[{author_name}") and not other_text.startswith("[Replying to"):
                other_text = f"[{author_name}]: {other_text}"

            bot_identity_badge = f" — Fellow Bot, NOT you ({bot_display_name}); do NOT copy their speech style" if msg.get("author_is_bot") else ""
            if addressed_bot and addressed_bot.lower() != bot_name_lower:
                other_text = re.sub(
                    rf'(?m)^\[{re.escape(author_name)}\]:',
                    f"[{author_name} (speaking to {addressed_bot}, NOT to {bot_display_name}{bot_identity_badge})]:",
                    other_text,
                    count=1
                )
            elif addressed_bot and addressed_bot.lower() == bot_name_lower:
                other_text = re.sub(
                    rf'(?m)^\[{re.escape(author_name)}\]:',
                    f"[{author_name} (speaking to YOU, {bot_display_name}{bot_identity_badge})]:",
                    other_text,
                    count=1
                )
            elif bot_identity_badge:
                other_text = re.sub(
                    rf'(?m)^\[{re.escape(author_name)}\]:',
                    f"[{author_name} (Fellow Bot — separate character from YOU, {bot_display_name}; do NOT mimic their speech style)]:",
                    other_text,
                    count=1
                )

            api_messages.append({"role": "user", "content": other_text})
            if glitch_summary and idx >= len(raw_history) - 10:
                api_messages.append({
                    "role": "system",
                    "content": f"[System Glitch Event: {author_name} also hit a media glitch ({glitch_summary}) on that message.]"
                })

    # Build a rich, dynamic Variety, Anti-Obsession & Human Spontaneity guidance block
    variety_rules = []
    bio_quotes = _extract_bio_example_quotes(dynamic_system.get("content", ""))
    if bio_quotes:
        bio_q_str = ", ".join(f'"{bq}..."' for bq in bio_quotes[:3])
        variety_rules.append(
            f"- **NEVER PARROT BIO EXAMPLE QUOTES**: Example lines in your bio ({bio_q_str}) are strictly tone illustrations—NEVER recite or copy them into live chat!"
        )

    if len(own_recent_openers) >= 2:
        recent_openers_str = ", ".join(f"'{op}...'" for op in own_recent_openers[-4:])
        variety_rules.append(
            f"- **VARY YOUR SENTENCE OPENINGS**: Do NOT start your message with any of your recent openers ({recent_openers_str})."
        )

    overused_emotes = [f"*{em}*" for em, cnt in own_emote_counter.most_common(5) if cnt >= 1]
    if overused_emotes:
        variety_rules.append(
            f"- **NO REPEATED STAGE DIRECTIONS**: You recently used {', '.join(overused_emotes[:4])}. "
            "Do NOT reuse those exact physical emotes on this turn—either use completely fresh body language or skip asterisk stage directions entirely and let your dialogue carry the attitude!"
        )

    repeated_templates = _extract_repeated_phrase_templates(own_recent_texts)
    if repeated_templates:
        tpl_str = ", ".join(f'"{t}"' for t in repeated_templates)
        variety_rules.append(
            f"- **BREAK FORMULAIC TEMPLATES & NEVER ECHO PREVIOUS LINES**: You already used {tpl_str} in recent turns. "
            "NEVER repeat or paraphrase your own earlier sentences—write a completely fresh response!"
        )

    overused_topics, user_asked_topic_change = _detect_overused_or_banned_topics(
        own_recent_texts, recent_human_texts, bot_display_name
    )
    if user_asked_topic_change:
        banned_t_str = ", ".join(f'"{w}"' for w in overused_topics) if overused_topics else "the previous recurring joke"
        variety_rules.append(
            f"- **USER REQUESTED TOPIC CHANGE (MANDATORY PIVOT)**: A user explicitly asked to stop talking about {banned_t_str} and move on! "
            f"Do NOT mention {banned_t_str} even as a joke, apology, or side comment, and do NOT generate any image/meme about it. Pivot 100% to a brand-new subject!"
        )
    elif len(overused_topics) >= 2:
        stale_t_str = ", ".join(f'"{w}"' for w in overused_topics[:6])
        variety_rules.append(
            f"- **ANTI-OBSESSION / DROP STALE JOKES**: You have already joked about {stale_t_str} across multiple recent turns. "
            "That bit is played out—move on to a fresh observation or new angle instead of repeating the same gag!"
        )

    banned_captions = list(dict.fromkeys(
        recent_meme_captions[-6:] + [c.upper() for c in sorted(OVERUSED_MEME_CAPTIONS)]
    ))
    if banned_captions:
        cap_str = "; ".join(f'"{c}"' for c in banned_captions[:6])
        variety_rules.append(
            f"- **BANNED MEME CAPTIONS (NEVER REUSE THESE)**: {cap_str}. "
            "If you generate a meme (`[MEME_GEN: scene | <witty setup> | <witty punchline>]`), both captions MUST be real, funny punchlines (NEVER literal words like 'TOP CAPTION')."
        )

    micro_impulse = random.choice(SPONTANEOUS_MICRO_MOMENTS)
    variety_rules.append(f"- **SPONTANEOUS CONVERSATIONAL IMPULSE**: {micro_impulse}")
    variety_rules.append(
        f"- **SINGLE-TURN DISCORD CHAT FORMAT (CRITICAL)**: Output ONLY your own single response as {bot_display_name}. "
        "Do NOT wrap your spoken sentences in quotation marks (`\"...\"`), NEVER echo the user's message at the start, "
        "NEVER output two separate turns on two lines, and NEVER write meta-stage notes like '[No response—...]'!"
    )

    if variety_rules:
        api_messages[0]["content"] += (
            "\n\n### Dynamic Spontaneity, Variety & Anti-Repetition Rules (Active This Turn):\n"
            + "\n".join(variety_rules)
        )

    return api_messages


def upsert_chat_memory_entry(chat_memory_file: str, channel_id: int, entry: dict, max_history_length: int = 200):
    """
    Atomically inserts or updates a message in chat_memory.json by message_id,
    stamping every entry with a Unix timestamp (`timestamp`) for session-gap awareness
    while preserving rich media descriptions and target-bot attribution tags.
    """
    if isinstance(entry, dict) and "timestamp" not in entry:
        entry = dict(entry)
        entry["timestamp"] = time.time()

    def _mutate(chat_memory):
        if channel_id not in chat_memory or not isinstance(chat_memory[channel_id], list):
            chat_memory[channel_id] = []

        healed_history = []
        for existing_item in chat_memory[channel_id]:
            c_val = existing_item.get("content", "")
            if isinstance(c_val, str) and existing_item.get("role") != "system":
                glitch_sum = extract_glitch_summary(c_val)
                cleaned_c = ERROR_REPLY_STRIP_REGEX.sub('', c_val).strip()
                if not cleaned_c or EMPTY_SPEAKER_TAG_REGEX.fullmatch(cleaned_c):
                    if glitch_sum:
                        a_name = existing_item.get("author_name", "A bot")
                        healed_history.append({
                            "role": "system",
                            "content": (
                                f"[System Glitch Event: {a_name} glitched out ({glitch_sum}). "
                                "Bots may react to or joke about this malfunction in-character, but NEVER output raw '*(API Error...)*' strings.]"
                            ),
                            "message_id": existing_item.get("message_id", f"healed_glitch_{time.time()}"),
                            "author_id": existing_item.get("author_id"),
                            "author_name": a_name,
                            "timestamp": existing_item.get("timestamp", time.time()),
                            "is_glitch_event": True
                        })
                    continue
                if cleaned_c != c_val:
                    existing_item = copy.deepcopy(existing_item)
                    existing_item["content"] = cleaned_c
            healed_history.append(existing_item)
        chat_memory[channel_id] = healed_history

        msg_id = entry.get("message_id")
        updated = False
        if msg_id is not None:
            for i, existing in enumerate(chat_memory[channel_id]):
                if existing.get("message_id") == msg_id:
                    merged = copy.deepcopy(existing)
                    orig_ts = merged.get("timestamp")
                    old_content = merged.get("content", "") if isinstance(merged.get("content"), str) else ""
                    new_content = entry.get("content", "") if isinstance(entry.get("content"), str) else ""

                    if entry.get("role") == "assistant" or entry.get("role") == "system":
                        merged.update(entry)
                    elif merged.get("role") != "assistant":
                        if not entry.get("addressed_bot") and merged.get("addressed_bot"):
                            entry["addressed_bot"] = merged["addressed_bot"]
                        merged.update(entry)
                        if old_content and new_content:
                            for meta_line in old_content.split("\n"):
                                meta_stripped = meta_line.strip()
                                if meta_stripped.startswith((
                                    "[Attached Media Description:",
                                    "[Contains GIF of:",
                                    "[Contains YouTube Video:",
                                    "[Linked Webpage:",
                                    "[Replying to "
                                )) and meta_stripped not in new_content:
                                    if meta_stripped.startswith("[Replying to "):
                                        new_content = f"{meta_stripped}\n{new_content}".strip()
                                    else:
                                        new_content = f"{new_content}\n{meta_stripped}".strip()
                            merged["content"] = new_content
                    else:
                        if "[Attached Media Description:" in new_content and "[Attached Media Description:" not in old_content:
                            desc_match = IMAGE_DESC_EXTRACT_REGEX.search(new_content)
                            if desc_match:
                                merged["content"] = f"{old_content}\n[Attached Media Description: {desc_match.group(1).strip()}]".strip()
                    if isinstance(orig_ts, (int, float)):
                        merged["timestamp"] = orig_ts
                    chat_memory[channel_id][i] = merged
                    updated = True
                    break

        if not updated:
            chat_memory[channel_id].append(entry)

        chat_memory[channel_id] = strip_old_image_data(chat_memory[channel_id], keep_images=0)
        if len(chat_memory[channel_id]) > max_history_length:
            chat_memory[channel_id] = chat_memory[channel_id][-max_history_length:]
        return chat_memory

    update_json_file(chat_memory_file, {}, _mutate)


def route_server_vs_general_facts(
    raw_server_facts: list[str],
    raw_general_facts: list[str],
    recent_users: set[str],
    other_bot_names: set[str],
    my_bot_name: str
) -> tuple[list[str], list[str]]:
    """
    Ensures universal science/trivia/world facts accidentally tagged with [LEARN: ...]
    are automatically re-routed to general_facts unless they mention server members,
    bots, or Discord/guild-specific terms.
    """
    server_indicators = set(BASE_SERVER_INDICATORS)
    for u in (recent_users or set()):
        if u:
            for part in ALPHANUM_TOKEN_REGEX.findall(str(u).lower()):
                if len(part) >= 3:
                    server_indicators.add(part)

    bot_name_tokens = set()
    for b in (other_bot_names or set()):
        if b:
            for part in ALPHANUM_TOKEN_REGEX.findall(str(b).lower()):
                if len(part) >= 3:
                    server_indicators.add(part)
                    if part not in ("bot", "the", "guild"):
                        bot_name_tokens.add(part)
    for part in ALPHANUM_TOKEN_REGEX.findall((my_bot_name or "").lower()):
        if len(part) >= 3:
            server_indicators.add(part)
            if part not in ("bot", "the", "guild"):
                bot_name_tokens.add(part)

    def _is_bot_banter_or_meta(fact_str: str, is_gen: bool) -> bool:
        if is_ephemeral_or_banter_fact(fact_str):
            return True
        f_low = fact_str.lower()
        f_words = set(ALPHANUM_TOKEN_REGEX.findall(f_low))
        # Universal general knowledge shouldn't be about server-specific bots (route those to server lore instead!)
        if (f_words & bot_name_tokens) and any(w in f_low for w in ("archived", "ritual", "recursion")):
            return True
        return False

    final_server_facts = []
    final_general_facts = []
    for gf in (raw_general_facts or []):
        clean_gf = gf.strip().rstrip(".")
        if not clean_gf or _is_bot_banter_or_meta(clean_gf, is_gen=True):
            continue
        gf_words = set(ALPHANUM_TOKEN_REGEX.findall(clean_gf.lower()))
        if gf_words & bot_name_tokens:
            final_server_facts.append(clean_gf)
        else:
            final_general_facts.append(clean_gf)

    for fact in (raw_server_facts or []):
        clean_f = fact.strip().rstrip(".")
        if not clean_f or _is_bot_banter_or_meta(clean_f, is_gen=False):
            continue
        words = set(ALPHANUM_TOKEN_REGEX.findall(clean_f.lower()))
        if words & server_indicators:
            final_server_facts.append(clean_f)
        else:
            print(f"[{my_bot_name}] [FACT ROUTER] Auto-routed universal fact from Server Lore -> General Knowledge: '{clean_f}'")
            final_general_facts.append(clean_f)

    return final_server_facts, final_general_facts
