import re
import math
import zlib
import json
import time
import copy
from datetime import datetime, timezone
from collections import Counter

import config
from config import (
    STOPWORDS,
    CORE_IDENTITY_FACT_REGEX,
    THIRD_PARTY_REPORT_REGEX,
    LOW_INFORMATION_RETRIEVAL_WORDS,
)
from core_system import post_with_backoff, update_json_file, load_json_file

# ==============================================================================
# PHASE 1 CONSTANTS & SAFE CONFIG FALLBACKS
# ==============================================================================
REUNION_ABSENCE_SECONDS = getattr(config, "REUNION_ABSENCE_SECONDS", 172800.0)  # 48 hours
REUNION_COOLDOWN_SECONDS = 21600.0  # 6 hours cooldown after a reunion trigger
EPISODIC_MIN_SESSION_MSGS = getattr(config, "EPISODIC_MIN_SESSION_MSGS", 18)
MAX_EPISODIC_CHAPTERS = getattr(config, "MAX_EPISODIC_CHAPTERS", 12)
DEFAULT_SUBCONSCIOUS_MODEL = "Qwen/Qwen3-30B-A3B"

ALPHANUM_TOKEN_REGEX = re.compile(r"[a-z0-9]+")

# ==============================================================================
# 1. DETERMINISTIC 0ms ZERO-PII & SENSITIVE DATA FIREWALL
# ==============================================================================
# Explicit surname / last-name patterns (strictly blocked)
_SURNAME_FACT_REGEX = re.compile(
    r"\b(?:last\s+name|surname|family\s+name|maiden\s+name|middle\s+name|full\s+legal\s+name)\b",
    re.IGNORECASE,
)

# Full-name capture ("Real name is First Last", "Full name is First Last", "Name is First Last")
# Automatically strips the surname so ONLY "First name is <First>" is retained!
_FULL_NAME_DECLARATION_REGEX = re.compile(
    r"^(?:his|her|their|user['’]?s|speaker['’]?s|real|full|first|legal)?\s*"
    r"(?:real\s+|full\s+|first\s+|legal\s+)?name\s+is\s+"
    r"([A-Z][a-z]{1,20})(?:\s+([A-Z][a-zA-Z'\-]{1,25}))+\b",
    re.IGNORECASE,
)
_INLINE_FULL_NAME_REGEX = re.compile(
    r"\b(?:real\s+name|full\s+name|first\s+name|named|goes\s+by)\s+(?:is\s+)?"
    r"([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,24})\b"
)

# Emails, Phone Numbers, SSNs, IPv4/IPv6, Street Addresses, ZIP codes, Financial & Secret Keys
_EMAIL_PII_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,15}\b"
)
_SSN_PII_REGEX = re.compile(
    r"\b(?:\d{3}[\-\s]\d{2}[\-\s]\d{4}|ssn\s*(?:is|:)?\s*\d{4,9}|social\s+security)\b",
    re.IGNORECASE,
)
_PHONE_PII_REGEX = re.compile(
    r"(?:\b(?:phone|cell|mobile|number|whatsapp|call\s+me\s+at|text\s+me\s+at)\b[^0-9\n]{0,15})?"
    r"\b(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}\b",
    re.IGNORECASE,
)
_IPV4_PII_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_IPV6_PII_REGEX = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b"
)
_CARD_CANDIDATE_REGEX = re.compile(
    r"\b(?:\d[ \-]?){13,19}\b"
)
_FINANCIAL_KEYWORDS_REGEX = re.compile(
    r"\b(?:credit\s+card|debit\s+card|card\s+number|cvv|cvc|security\s+code|"
    r"routing\s+number|bank\s+account|iban|swift\s+code|seed\s+phrase|private\s+key|"
    r"wallet\s+key|recovery\s+phrase|pin\s+number|pin\s+is)\b",
    re.IGNORECASE,
)
_STREET_ADDRESS_PII_REGEX = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\-'\s]{2,32}\s+"
    r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|"
    r"drive|dr\.?|court|ct\.?|circle|cir\.?|parkway|pkwy\.?|place|pl\.?|way|terrace|ter\.?|trail|trl\.?)\b"
    r"|\b(?:apt\.?|apartment|suite|ste\.?|unit|p\.?\s*o\.?\s*box)\s*#?\s*[0-9A-Za-z\-]{1,8}\b"
    r"|\b(?:zip\s+code|postal\s+code|home\s+address|street\s+address|mailing\s+address|physical\s+address)\b",
    re.IGNORECASE,
)
_GPS_COORDS_PII_REGEX = re.compile(
    r"[\+\-]?\d{1,2}\.\d{4,}\s*,\s*[\+\-]?\d{1,3}\.\d{4,}"
)
_CREDENTIALS_PII_REGEX = re.compile(
    r"\b(?:password\s+is|passcode\s+is|my\s+password|api\s*key|secret\s*key|bearer\s+[A-Za-z0-9_\-\.]+|"
    r"sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{25,}|xox[baprs]-[0-9A-Za-z\-]{10,})\b",
    re.IGNORECASE,
)

_NON_SURNAME_CAPITALIZED_WORDS = frozenset({
    "University", "College", "School", "Academy", "Institute", "State", "City",
    "County", "Texas", "California", "Florida", "York", "London", "Canada",
    "America", "Discord", "Server", "Guild", "Python", "Java", "Linux",
    "Windows", "Gaming", "World", "Warcraft", "Minecraft", "Elden", "Ring",
    "Modern", "Warfare", "Factorio", "Palworld", "Steam", "Twitch", "Major",
    "Degree", "Bachelor", "Master", "Science", "Arts", "Engineering", "Music",
})


def _passes_luhn_check(digits_str: str) -> bool:
    """Returns True if a 13-19 digit string satisfies the Luhn credit-card checksum."""
    digits = [int(ch) for ch in digits_str if ch.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    if len(set(digits)) == 1:
        return True
    total = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (total % 10) == 0


def sanitize_or_reject_sensitive_pii(raw_fact: str) -> str | None:
    """
    Deterministic 0ms Zero-PII & Sensitive Data Firewall:
    1. Allows first names ('First name is Alex', 'Goes by Alex') while stripping surnames
       from full-name strings ('Real name is Alex Mercer' -> 'First name is Alex') and
       rejecting explicit surname facts ('Last name is...').
    2. Rejects credit/debit card numbers, CVVs, bank/routing numbers, and crypto keys.
    3. Rejects physical street addresses, apartment/unit numbers, ZIP codes, and GPS coords
       (broad state/timezone like 'Lives in Texas' or 'CST timezone' is allowed).
    4. Rejects SSNs, phone numbers, emails, IP addresses, passwords, and API keys/tokens.
    Returns the sanitized fact string, or `None` if the fact must be blocked.
    """
    if not raw_fact or not isinstance(raw_fact, str):
        return None

    # Preserve internal rapport shift command tokens untouched
    stripped = raw_fact.strip()
    if stripped.startswith("__RAPPORT_SHIFT__:") or stripped.lower().startswith("rapport:"):
        return stripped

    # Unpack optional [S:XX|CAT|DATE|R:N] metadata prefix before scanning text
    unpacked = unpack_salience_fact(stripped)
    text_body = unpacked["text"].strip()
    if not text_body:
        return None

    # 1. Hard-block explicit surname / maiden name / middle name facts
    if _SURNAME_FACT_REGEX.search(text_body):
        return None

    # 2. Hard-block financial, credentials, SSN, email, phone, IP, street address, GPS
    if (
        _EMAIL_PII_REGEX.search(text_body)
        or _SSN_PII_REGEX.search(text_body)
        or _PHONE_PII_REGEX.search(text_body)
        or _IPV4_PII_REGEX.search(text_body)
        or _IPV6_PII_REGEX.search(text_body)
        or _FINANCIAL_KEYWORDS_REGEX.search(text_body)
        or _STREET_ADDRESS_PII_REGEX.search(text_body)
        or _GPS_COORDS_PII_REGEX.search(text_body)
        or _CREDENTIALS_PII_REGEX.search(text_body)
    ):
        return None

    # Check 13-19 digit sequences for credit/debit cards or raw account numbers
    for card_m in _CARD_CANDIDATE_REGEX.finditer(text_body):
        raw_digits = "".join(ch for ch in card_m.group(0) if ch.isdigit())
        if 13 <= len(raw_digits) <= 19:
            return None

    # 3. Sanitize Full Legal Names -> Keep ONLY the First Name!
    full_m = _FULL_NAME_DECLARATION_REGEX.match(text_body)
    if full_m:
        first_name = full_m.group(1).strip().capitalize()
        text_body = f"First name is {first_name}"
    else:
        def _strip_inline_surname(m: re.Match) -> str:
            first_n = m.group(1).strip()
            second_n = m.group(2).strip()
            if second_n in _NON_SURNAME_CAPITALIZED_WORDS:
                return m.group(0)
            return f"First name is {first_n}"

        text_body = _INLINE_FULL_NAME_REGEX.sub(_strip_inline_surname, text_body)
        # Normalize "Real name is <First>" -> "First name is <First>"
        text_body = re.sub(
            r"^(?:real|full|legal)\s+name\s+is\s+([A-Z][a-z]{1,20})$",
            r"First name is \1",
            text_body,
            flags=re.IGNORECASE,
        )

    if unpacked["has_prefix"]:
        return pack_salience_fact(
            text_body,
            salience=unpacked["salience"],
            category=unpacked["category"],
            date_str=unpacked["date"],
            recalls=unpacked["recalls"],
        )
    return text_body


# ==============================================================================
# 2. 1-100 SALIENCE SPECTRUM + OKAPI BM25 + EBBINGHAUS FORGETTING CURVE (#4)
# ==============================================================================
SALIENCE_PREFIX_REGEX = re.compile(
    r"^\[S:(\d{1,3})\|([A-Z_]{2,12})\|(\d{4}-\d{2}-\d{2})(?:\|R:(\d+))?\]\s*(.+)$"
)

# Stability constants tau_cat (in days) for Ebbinghaus retention R(t) = S * exp(-dt / (tau * (1 + 0.4*R)))
CATEGORY_STABILITY_TAU_DAYS: dict[str, float] = {
    "CORE": float("inf"),      # Immune to decay and eviction
    "BOUNDARY": float("inf"),  # Immune to decay and eviction
    "PROJECT": 120.0,
    "PREF": 120.0,
    "REL": 120.0,
    "HABIT": 35.0,
    "OPINION": 35.0,
    "LORE": 35.0,
    "GEN": 60.0,
    "SITUATIONAL": 7.0,
}

CATEGORY_DISPLAY_LABELS: dict[str, str] = {
    "CORE": "Core",
    "BOUNDARY": "Boundary",
    "PROJECT": "Project",
    "PREF": "Preference",
    "REL": "Relationship",
    "HABIT": "Habit",
    "OPINION": "Opinion",
    "LORE": "Lore",
    "GEN": "Fact",
    "SITUATIONAL": "Recent",
}

_BOUNDARY_CUE_REGEX = re.compile(
    r"\b(?:prefers?\s+not\s+to\s+be\s+called|do(?:es)?\s+not\s+call|don['’]?t\s+call\s+me|"
    r"boundary|uncomfortable\s+with|never\s+call|prefers?\s+to\s+be\s+called)\b",
    re.IGNORECASE,
)
_PROJECT_PREF_CUE_REGEX = re.compile(
    r"\b(?:ryzen|rtx|gpu|cpu|ram|motherboard|pc\b|mini\s+pc|coding|programming|python|c\+\+|lua|"
    r"bot\b|framework|github|project|building|developing|plays?\s+|favorite|loves?\s+|enjoys?\s+|"
    r"guitar|piano|synth|music|cat\b|dog\b|pet\b|main\s+in|warcraft|minecraft|factorio|elden\s+ring)\b",
    re.IGNORECASE,
)
_HABIT_OPINION_CUE_REGEX = re.compile(
    r"\b(?:usually|every\s+morning|every\s+night|routine|habit|schedule|late\s+at\s+night|"
    r"thinks\s+that|believes|opinion|hot\s+take|running\s+gag|in[\s\-]joke|server\s+lore)\b",
    re.IGNORECASE,
)
_SITUATIONAL_CUE_REGEX = re.compile(
    r"\b(?:tonight|right\s+now|earlier\s+today|this\s+evening|this\s+morning|ate\s+|"
    r"having\s+for\s+dinner|feeling\s+tired|currently\s+debugging|just\s+finished)\b",
    re.IGNORECASE,
)


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def infer_fact_salience_and_category(
    fact_text: str,
    default_salience: int = 65,
    default_cat: str = "GEN"
) -> tuple[int, str]:
    """
    Deterministically assigns a 1-100 Salience score and Category when a fact
    is learned without explicit metadata.
    """
    if not fact_text:
        return default_salience, default_cat
    clean = fact_text.strip()
    if _BOUNDARY_CUE_REGEX.search(clean):
        return 96, "BOUNDARY"
    if CORE_IDENTITY_FACT_REGEX.search(clean) or re.search(
        r"\b(?:is\s+a\s+(?:man|woman|non-binary)|pronouns\s+are|years\s+old|first\s+name\s+is|"
        r"married\s+to|wife|husband|spouse|bachelor|master['’]?s|major(?:ing)?\s+in|computer\s+science)\b",
        clean,
        re.IGNORECASE,
    ):
        return 95, "CORE"
    if _SITUATIONAL_CUE_REGEX.search(clean):
        return 32, "SITUATIONAL"
    if _PROJECT_PREF_CUE_REGEX.search(clean):
        return 80, "PROJECT" if re.search(r"\b(?:project|building|developing|github|framework|pc\b|gpu|cpu)\b", clean, re.I) else "PREF"
    if _HABIT_OPINION_CUE_REGEX.search(clean):
        return 58, "HABIT"
    return max(1, min(100, int(default_salience))), default_cat.upper()


def unpack_salience_fact(raw_fact: str) -> dict:
    """
    Unpacks an optional `[S:XX|CAT|YYYY-MM-DD|R:N] <text>` string into a structured dict.
    Plain legacy strings seamlessly default to `salience=65, category='GEN', recalls=0`.
    """
    if not raw_fact or not isinstance(raw_fact, str):
        return {
            "text": "",
            "salience": 65,
            "category": "GEN",
            "date": _today_utc_str(),
            "recalls": 0,
            "raw": "",
            "has_prefix": False,
        }
    s_clean = raw_fact.strip()
    m = SALIENCE_PREFIX_REGEX.match(s_clean)
    if m:
        sal = max(1, min(100, int(m.group(1))))
        cat = m.group(2).strip().upper()
        dt_str = m.group(3).strip()
        recalls = int(m.group(4)) if m.group(4) is not None else 0
        txt = m.group(5).strip()
        return {
            "text": txt,
            "salience": sal,
            "category": cat,
            "date": dt_str,
            "recalls": max(0, recalls),
            "raw": s_clean,
            "has_prefix": True,
        }
    inferred_sal, inferred_cat = infer_fact_salience_and_category(s_clean)
    return {
        "text": s_clean,
        "salience": inferred_sal,
        "category": inferred_cat,
        "date": _today_utc_str(),
        "recalls": 0,
        "raw": s_clean,
        "has_prefix": False,
    }


def pack_salience_fact(
    fact_text: str,
    salience: int | None = None,
    category: str | None = None,
    date_str: str | None = None,
    recalls: int | None = None,
) -> str:
    """
    Packs a fact string with `[S:XX|CAT|YYYY-MM-DD|R:N] <text>`.
    If `fact_text` already has a prefix, preserves/updates its metadata cleanly.
    """
    existing = unpack_salience_fact(fact_text)
    clean_text = existing["text"].strip().rstrip(".")
    if not clean_text:
        return ""
    if clean_text.startswith("__RAPPORT_SHIFT__:") or clean_text.lower().startswith("rapport:"):
        return clean_text

    if salience is None or category is None:
        inf_s, inf_c = infer_fact_salience_and_category(
            clean_text,
            default_salience=existing["salience"],
            default_cat=existing["category"],
        )
        sal = inf_s if salience is None else int(salience)
        cat = inf_c if category is None else str(category).strip().upper()
    else:
        sal = int(salience)
        cat = str(category).strip().upper()

    sal = max(1, min(100, sal))
    if cat not in CATEGORY_STABILITY_TAU_DAYS:
        cat = "GEN"
    if sal >= 90 and cat not in ("CORE", "BOUNDARY"):
        cat = "BOUNDARY" if _BOUNDARY_CUE_REGEX.search(clean_text) else "CORE"

    dt = date_str if (date_str and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date_str))) else existing["date"]
    rec = existing["recalls"] if recalls is None else max(0, int(recalls))
    return f"[S:{sal:02d}|{cat}|{dt}|R:{rec}] {clean_text}"


def format_fact_for_display(raw_fact: str, include_badge: bool = False) -> str:
    """
    Formats a stored fact for clean presentation:
    - `include_badge=False` (used inside LLM prompts): returns ONLY the clean fact text
      so the LLM never echoes `[S:95|CORE|...]` metadata tags into Discord chat.
    - `include_badge=True` (used in `!whoami` embeds and Web-UI): prefixes a clean badge
      like `[Core • S:95] Is a man (male, he/him)`.
    """
    info = unpack_salience_fact(raw_fact)
    txt = info["text"]
    if not include_badge or not txt:
        return txt
    label = CATEGORY_DISPLAY_LABELS.get(info["category"], "Fact")
    return f"[{label} • S:{info['salience']}] {txt}"


def compute_ebbinghaus_retention(raw_fact: str, now_ts: float | None = None) -> float:
    """
    Computes the Ebbinghaus Exponential Memory Retention score:
      R(t) = S * exp( - delta_days / (tau_cat * (1 + 0.4 * C_recall)) )
    `CORE` and `BOUNDARY` facts (or any fact with `S >= 90`) have infinite stability
    and return `float(S)` with zero decay.
    """
    info = unpack_salience_fact(raw_fact)
    sal = float(info["salience"])
    cat = info["category"]
    if sal >= 90.0 or cat in ("CORE", "BOUNDARY"):
        return max(90.0, sal)

    tau = CATEGORY_STABILITY_TAU_DAYS.get(cat, 60.0)
    if math.isinf(tau):
        return sal

    now_val = now_ts if now_ts is not None else time.time()
    try:
        dt_obj = datetime.strptime(info["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        delta_days = max(0.0, (now_val - dt_obj.timestamp()) / 86400.0)
    except Exception:
        delta_days = 0.0

    effective_tau = tau * (1.0 + 0.4 * float(info["recalls"]))
    retention = sal * math.exp(-delta_days / max(1.0, effective_tau))
    return round(retention, 3)


def evict_lowest_retention_facts(
    existing_facts: list[str],
    max_items: int = 35,
    rapport_regex: re.Pattern | None = None,
) -> list[str]:
    """
    Evicts the lowest-retention ephemeral/situational facts when `existing_facts`
    exceeds `max_items`, strictly protecting `Rapport:` lines and `CORE`/`BOUNDARY`
    (`S >= 90`) facts from ever being forgotten.
    """
    if len(existing_facts) <= max_items:
        return existing_facts

    rapport_lines = []
    regular_items = []
    for idx, f_str in enumerate(existing_facts):
        s_clean = str(f_str).strip()
        if rapport_regex and rapport_regex.match(s_clean):
            rapport_lines.append(s_clean)
        elif s_clean.lower().startswith("rapport:"):
            rapport_lines.append(s_clean)
        else:
            regular_items.append((idx, s_clean))

    budget = max(1, max_items - (1 if rapport_lines else 0))
    if len(regular_items) <= budget:
        out = [f for _, f in regular_items]
        if rapport_lines:
            out.append(rapport_lines[-1])
        return out

    now_val = time.time()
    scored = []
    for idx, f_str in regular_items:
        info = unpack_salience_fact(f_str)
        is_immune = 1 if (info["salience"] >= 90 or info["category"] in ("CORE", "BOUNDARY")) else 0
        ret = compute_ebbinghaus_retention(f_str, now_ts=now_val)
        scored.append((is_immune, ret, idx, f_str))

    # Sort lowest retention first to drop `(len - budget)` weakest items, then restore chronological order
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    kept = scored[-budget:]
    kept.sort(key=lambda x: x[2])

    result = [f for _, _, _, f in kept]
    if rapport_lines:
        result.append(rapport_lines[-1])
    return result


def _stem_word_bm25(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxzho":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _tokenize_bm25(text: str) -> list[str]:
    if not text:
        return []
    tokens = []
    for w in ALPHANUM_TOKEN_REGEX.findall(text.lower()):
        if len(w) < 3 or w in STOPWORDS:
            continue
        st = _stem_word_bm25(w)
        if len(st) >= 3:
            tokens.append(st)
    return tokens


def select_relevant_memories_bm25(
    facts_list: list[str],
    active_context_text: str,
    semantic_tags: list[str] | None = None,
    max_items: int = 5,
    min_score: float = 1.65,
    is_recall_query: bool = False,
) -> list[str]:
    """
    Okapi BM25 (k1=1.5, b=0.75) + Ebbinghaus Retention + Semantic Tag Hybrid RAG:
    Scores candidate memories in <0.5ms using corpus-level IDF rarity, document length
    normalization, `_room_state['semantic_tags']` expansion, and Ebbinghaus retention R(t).
    Returns clean (unprefixed) fact strings ready for prompt injection.
    """
    if not facts_list or not isinstance(facts_list, list):
        return []

    parsed_docs = []
    for idx, raw_f in enumerate(facts_list):
        if not raw_f:
            continue
        s_clean = str(raw_f).strip()
        if not s_clean or s_clean.startswith("__RAPPORT_SHIFT__:") or s_clean.lower().startswith("rapport:"):
            continue
        info = unpack_salience_fact(s_clean)
        if len(info["text"]) < 4:
            continue
        doc_tokens = _tokenize_bm25(info["text"])
        if doc_tokens:
            parsed_docs.append((idx, info, doc_tokens, Counter(doc_tokens)))

    if not parsed_docs:
        return []

    now_val = time.time()
    if is_recall_query:
        # Sort by Ebbinghaus retention + recency and return clean strings
        ranked = sorted(
            parsed_docs,
            key=lambda item: (compute_ebbinghaus_retention(item[1]["raw"], now_ts=now_val), item[0]),
            reverse=True,
        )
        return [item[1]["text"] for item in ranked[:max_items]]

    query_tokens = _tokenize_bm25(active_context_text or "")
    tag_tokens = set(_tokenize_bm25(" ".join(semantic_tags or [])))
    combined_query_terms = list(dict.fromkeys(query_tokens + list(tag_tokens)))
    if not combined_query_terms:
        return []

    n_docs = len(parsed_docs)
    avgdl = max(1.0, sum(len(d[2]) for d in parsed_docs) / float(n_docs))
    k1 = 1.5
    b = 0.75

    # Compute document frequency df(q) across the memory pool
    df_counts: Counter = Counter()
    for _, _, _, tf_map in parsed_docs:
        for term in tf_map.keys():
            df_counts[term] += 1

    low_info_stems = {_stem_word_bm25(w) for w in LOW_INFORMATION_RETRIEVAL_WORDS}
    scored_results = []

    for idx, info, doc_tokens, tf_map in parsed_docs:
        doc_len = len(doc_tokens)
        bm25_sum = 0.0
        specific_hits = 0

        for q in combined_query_terms:
            tf = tf_map.get(q, 0)
            if tf <= 0:
                continue
            df = df_counts.get(q, 0)
            # Robertson-Spärck Jones IDF with floor boost for small corpora
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5))) + 0.85
            if q in low_info_stems:
                idf *= 0.20
            else:
                specific_hits += 1

            denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            bm25_sum += idf * ((tf * (k1 + 1.0)) / max(0.001, denom))

        if specific_hits == 0:
            continue

        # Semantic tag overlap bonus
        sem_overlap = len(tag_tokens & set(tf_map.keys()))
        sem_bonus = min(1.8, sem_overlap * 0.65)

        # Ebbinghaus retention multiplier (0.75x to 1.35x)
        ret = compute_ebbinghaus_retention(info["raw"], now_ts=now_val)
        ret_multiplier = 0.75 + 0.60 * (min(100.0, ret) / 100.0)

        total_score = (bm25_sum + sem_bonus) * ret_multiplier
        if total_score >= min_score:
            scored_results.append((total_score, idx, info["text"]))

    scored_results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [txt for _, _, txt in scored_results[:max_items]]


# ==============================================================================
# 3. HEIDER'S STRUCTURAL BALANCE & CONTROL-SYSTEMS AFFINITY DYNAMICS (#5 & #7)
# ==============================================================================
_SESSION_POSITIVE_RAPPORT_LOG: dict[tuple[int, str], list[float]] = {}
SESSION_DAMPING_WINDOW_SECONDS = 900.0  # 15 minutes


def compute_heider_triadic_balance_delta(w_bot_to_b: int, valence_a_to_b: float) -> int:
    """
    Upgrade #5 — Heider's Structural Balance Triangle Rule on Signed Directed Social Graph:
      Delta w_{i -> A} = round( 8.0 * tanh(w_{i -> B} / 45.0) * Valence(A -> B) )
    where `valence_a_to_b` is in [-1.5, +1.5] (positive when A praises/defends B,
    negative when A attacks/insults B, or mildly negative when A teases rival B).
    """
    if w_bot_to_b == 0 or abs(valence_a_to_b) < 0.05:
        return 0
    clamped_w = max(-100.0, min(100.0, float(w_bot_to_b)))
    clamped_v = max(-1.5, min(1.5, float(valence_a_to_b)))
    raw_shift = 8.0 * math.tanh(clamped_w / 45.0) * clamped_v
    return int(round(raw_shift))


def apply_session_diminishing_returns(
    channel_id: int,
    speaker_name: str,
    raw_delta: int,
    now_ts: float | None = None,
) -> int:
    """
    Upgrade #7 — Overdamped 1/sqrt(k) Session Diminishing Marginal Returns:
    Within a 15-minute sliding window, the k-th positive rapport shift from the
    same speaker is damped by 1 / sqrt(k) so compliment-spamming cannot artificially
    max out Affinity in a single session. Negative shifts (hostility) are never damped.
    """
    if raw_delta <= 0 or not speaker_name:
        return raw_delta
    now_val = now_ts if now_ts is not None else time.time()
    key = (int(channel_id or 0), speaker_name.strip().lower())
    history = _SESSION_POSITIVE_RAPPORT_LOG.get(key, [])
    history = [ts for ts in history if (now_val - ts) <= SESSION_DAMPING_WINDOW_SECONDS]
    history.append(now_val)
    _SESSION_POSITIVE_RAPPORT_LOG[key] = history[-25:]

    k = len(history)
    if k <= 1:
        return raw_delta
    damped = int(round(float(raw_delta) / math.sqrt(float(k))))
    return max(1, damped)


def get_disposition_tier_with_hysteresis(score: int, previous_tier: str = "") -> str:
    """
    Upgrade #7 — Schmitt-Trigger Hysteresis (+-3 points) around Disposition Tier
    boundaries (-35, -10, +15, +50) so a speaker hovering near a threshold never
    flip-flops between tiers on consecutive turns.
    """
    h = 3
    prev = (previous_tier or "").strip()

    if prev == "Cold & Cutting":
        if score <= -35 + h:
            return "Cold & Cutting"
    elif prev == "Prickly & Skeptical":
        if (-35 - h) < score <= (-10 + h):
            return "Prickly & Skeptical"
    elif prev == "Aloof & Guarded":
        if (-10 - h) < score < (15 + h):
            return "Aloof & Guarded"
    elif prev == "Thawing & Reluctantly Amused":
        if (15 - h) <= score < (50 + h):
            return "Thawing & Reluctantly Amused"
    elif prev == "Trusted Inner Circle":
        if score >= (50 - h):
            return "Trusted Inner Circle"

    if score <= -35:
        return "Cold & Cutting"
    if score <= -10:
        return "Prickly & Skeptical"
    if score < 15:
        return "Aloof & Guarded"
    if score < 50:
        return "Thawing & Reluctantly Amused"
    return "Trusted Inner Circle"


# ==============================================================================
# 4. KOLMOGOROV COMPLEXITY (zlib NCD) ANTI-LOOP & TOKEN-BUDGETED SLICER (#6, #11)
# ==============================================================================
def compute_zlib_ncd(text_x: str, text_y: str) -> float:
    """
    Upgrade #6 — Normalized Compression Distance via zlib (Kolmogorov Complexity proxy):
      NCD(x, y) = (C(x + y) - min(C(x), C(y))) / max(C(x), C(y))
    Runs in ~0.04ms and detects structural/cadence/syntax loops even when specific
    words are swapped.
    """
    x_clean = (text_x or "").strip()
    y_clean = (text_y or "").strip()
    if len(x_clean) < 24 or len(y_clean) < 24:
        return 1.0
    bx = x_clean.encode("utf-8", errors="ignore")
    by = y_clean.encode("utf-8", errors="ignore")
    cx = len(zlib.compress(bx, level=6))
    cy = len(zlib.compress(by, level=6))
    cxy = len(zlib.compress(bx + b" " + by, level=6))
    denom = float(max(cx, cy, 1))
    ncd = (cxy - min(cx, cy)) / denom
    return max(0.0, min(1.0, round(ncd, 4)))


def detect_ncd_structural_loop(
    own_recent_texts: list[str],
    threshold: float = 0.42,
) -> tuple[bool, float]:
    """
    Checks the bot's last 3 responses using `compute_zlib_ncd`.
    Returns `(True, min_ncd)` if any recent pair falls below `threshold` (0.42).
    """
    candidates = [t.strip() for t in (own_recent_texts or [])[-3:] if t and len(t.strip()) >= 28]
    if len(candidates) < 2:
        return False, 1.0
    min_ncd = 1.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            d = compute_zlib_ncd(candidates[i], candidates[j])
            if d < min_ncd:
                min_ncd = d
    return (min_ncd < threshold), min_ncd


def enforce_dynamic_token_budget(
    api_messages: list[dict],
    max_char_budget: int = 14000,
    keep_recent_intact: int = 12,
) -> list[dict]:
    """
    Upgrade #11 — Dynamic Token-Budgeted Context Slicer (~3,500 tokens / 14,000 chars):
    Keeps the system prompt (`api_messages[0]`) and the last `keep_recent_intact` (12)
    messages at 100% fidelity, while middle-compacting or trimming older history turns
    whenever total conversation history exceeds `max_char_budget`.
    """
    if not api_messages or len(api_messages) <= keep_recent_intact + 1:
        return api_messages

    sys_msg = api_messages[0]
    history_turns = api_messages[1:]
    total_chars = sum(len(str(m.get("content", ""))) for m in history_turns)
    if total_chars <= max_char_budget:
        return api_messages

    recent_slice = history_turns[-keep_recent_intact:]
    older_slice = history_turns[:-keep_recent_intact]
    recent_chars = sum(len(str(m.get("content", ""))) for m in recent_slice)
    remaining_budget = max(2400, max_char_budget - recent_chars)

    compacted_older = []
    used_chars = 0
    for m in reversed(older_slice):
        c_str = str(m.get("content", "")).strip()
        if not c_str:
            continue
        if len(c_str) > 260:
            c_str = c_str[:160].rstrip() + " ... " + c_str[-75:].lstrip()
        if used_chars + len(c_str) > remaining_budget:
            break
        m_copy = dict(m)
        m_copy["content"] = c_str
        compacted_older.append(m_copy)
        used_chars += len(c_str)

    compacted_older.reverse()
    return [sys_msg] + compacted_older + recent_slice


def format_message_reactions_footer(reactions_dict: dict | None) -> str:
    """
    Formats an in-place `entry["reactions"] = {"😂": ["Alex", "Cat Bot"]}` dict
    into a single compact footer line for `build_api_messages_from_history`.
    """
    if not reactions_dict or not isinstance(reactions_dict, dict):
        return ""
    parts = []
    for emoji_k, reactors in reactions_dict.items():
        if not emoji_k or not isinstance(reactors, list) or not reactors:
            continue
        clean_names = [str(r).strip() for r in reactors if r and str(r).strip()]
        if clean_names:
            parts.append(f"{emoji_k} ({', '.join(clean_names[:5])})")
    if not parts:
        return ""
    return f"\n[Reactions on this message: {' | '.join(parts[:6])}]"


# ==============================================================================
# 5. IN-PLACE REACTION LEDGER, REUNION RADAR, TTL ARCS & SUBCONSCIOUS TURN
# ==============================================================================
_POSITIVE_REACTION_EMOJIS = frozenset({
    "❤️", "❤", "🧡", "💛", "💚", "💙", "💜", "💖", "💗", "💓", "💞", "💕",
    "😂", "🤣", "😆", "😊", "🥰", "😍", "🔥", "👍", "👏", "🙌", "✨", "⭐",
    "🌟", "💯", "🎉", "🥳", "🤝", "🫶", "😺", "😸", "😻", "🐾", "🐟", "🍵",
})
_NEGATIVE_REACTION_EMOJIS = frozenset({
    "👎", "🖕", "🤮", "🤢", "💩", "🗑️", "😡", "🤬", "🍅",
})
_REACTION_RAPPORT_THROTTLE: dict[tuple[int, str], float] = {}


def update_message_reaction_ledger(
    chat_memory_file: str,
    user_profiles_file: str,
    channel_id: int,
    message_id: int,
    emoji_str: str,
    reactor_name: str,
    is_add: bool = True,
    reactor_is_bot: bool = False,
    my_bot_name: str = "",
) -> bool:
    """
    Atomically updates `entry["reactions"][emoji_str]` inside `chat_memory.json`
    (deduplicated across multi-bot processes). When a human reacts to `my_bot_name`'s
    own message, applies a throttled `+2` or `-3` Affinity micro-nudge!
    """
    if not chat_memory_file or not channel_id or not message_id or not emoji_str or not reactor_name:
        return False

    target_author_holder = {"author_name": "", "changed": False}

    def _mutate_chat(chat_memory: dict) -> dict:
        ch_list = chat_memory.get(channel_id)
        if not isinstance(ch_list, list):
            return chat_memory
        for entry in reversed(ch_list):
            if not isinstance(entry, dict):
                continue
            if entry.get("message_id") == message_id:
                target_author_holder["author_name"] = str(entry.get("author_name", "") or "")
                reactions = entry.get("reactions")
                if not isinstance(reactions, dict):
                    reactions = {}
                bucket = reactions.get(emoji_str)
                if not isinstance(bucket, list):
                    bucket = []
                if is_add:
                    if reactor_name not in bucket:
                        bucket.append(reactor_name)
                        reactions[emoji_str] = bucket[:12]
                        entry["reactions"] = reactions
                        target_author_holder["changed"] = True
                else:
                    if reactor_name in bucket:
                        bucket.remove(reactor_name)
                        if bucket:
                            reactions[emoji_str] = bucket
                        else:
                            reactions.pop(emoji_str, None)
                        entry["reactions"] = reactions
                        target_author_holder["changed"] = True
                break
        return chat_memory

    update_json_file(chat_memory_file, {}, _mutate_chat)

    # Apply throttled +2 / -3 Affinity micro-nudge when a human reacts to this bot's message
    if (
        is_add
        and target_author_holder["changed"]
        and not reactor_is_bot
        and my_bot_name
        and user_profiles_file
        and target_author_holder["author_name"].strip().lower() == my_bot_name.strip().lower()
    ):
        delta = 0
        if emoji_str in _POSITIVE_REACTION_EMOJIS:
            delta = 2
        elif emoji_str in _NEGATIVE_REACTION_EMOJIS:
            delta = -3
        if delta != 0:
            now_val = time.time()
            t_key = (int(channel_id), reactor_name.strip().lower())
            last_t = _REACTION_RAPPORT_THROTTLE.get(t_key, 0.0)
            if (now_val - last_t) >= 45.0:
                _REACTION_RAPPORT_THROTTLE[t_key] = now_val
                from memory_and_context import consolidate_and_append_fact, find_canonical_username

                def _mutate_profile(profiles: dict) -> dict:
                    canon = find_canonical_username(reactor_name, profiles, {reactor_name})
                    if canon:
                        facts = profiles.setdefault(canon, [])
                        trend_str = "Warming — Reacted Positively" if delta > 0 else "Cooling — Hostile Reaction"
                        cat_str = f"Reacted with {emoji_str} to {my_bot_name}'s post"
                        consolidate_and_append_fact(
                            f"__RAPPORT_SHIFT__:{delta}|{trend_str}|{cat_str}",
                            facts,
                        )
                    return profiles

                update_json_file(user_profiles_file, {}, _mutate_profile)

    return target_author_holder["changed"]


def record_and_check_user_reunion(
    shared_cache_file: str,
    channel_id: int,
    speaker_name: str,
    is_speaker_bot: bool = False,
    now_ts: float | None = None,
) -> tuple[bool, float]:
    """
    Tracks each human user's last seen timestamp in `shared_image_cache.json`.
    If the user returns after `>= REUNION_ABSENCE_SECONDS` (48 hours), returns
    `(True, absence_seconds)` for all bots responding within a 30-second window on
    that return message, then stamps a cooldown so it only triggers once per reunion.
    """
    if not shared_cache_file or not speaker_name or is_speaker_bot:
        return False, 0.0

    speaker_key = speaker_name.strip().lower()
    now_val = now_ts if now_ts is not None else time.time()
    result_holder = {"is_reunion": False, "gap": 0.0}

    def _mutate(cache: dict) -> dict:
        last_seen_map = cache.setdefault("_user_last_seen_ts", {})
        reunion_active_map = cache.setdefault("_user_reunion_window", {})

        # If another bot already detected this user's reunion within the last 25 seconds,
        # share the same reunion signal for this turn!
        active_info = reunion_active_map.get(speaker_key)
        if isinstance(active_info, dict):
            trig_ts = float(active_info.get("triggered_at", 0.0))
            if (now_val - trig_ts) <= 25.0:
                result_holder["is_reunion"] = True
                result_holder["gap"] = float(active_info.get("gap_seconds", REUNION_ABSENCE_SECONDS))
                last_seen_map[speaker_key] = now_val
                return cache

        prev_ts = last_seen_map.get(speaker_key)
        last_seen_map[speaker_key] = now_val

        if isinstance(prev_ts, (int, float)) and prev_ts > 0:
            gap = now_val - float(prev_ts)
            last_trig = float(active_info.get("triggered_at", 0.0)) if isinstance(active_info, dict) else 0.0
            if gap >= REUNION_ABSENCE_SECONDS and (now_val - last_trig) >= REUNION_COOLDOWN_SECONDS:
                result_holder["is_reunion"] = True
                result_holder["gap"] = gap
                reunion_active_map[speaker_key] = {
                    "triggered_at": now_val,
                    "gap_seconds": round(gap, 1),
                    "channel_id": int(channel_id or 0),
                }
        return cache

    update_json_file(shared_cache_file, {}, _mutate)
    return result_holder["is_reunion"], result_holder["gap"]


def get_and_tick_bot_internal_state(
    shared_cache_file: str,
    channel_id: int,
    bot_name: str,
    tick_ttl: bool = False,
) -> dict:
    """
    Reads (and optionally decrements TTL counters for) the bot's internal carryover state:
    - `mood`: str (`ttl` default 3 turns)
    - `inner_thought`: str (`ttl` default 2 turns)
    - `arc_notes`: list of `{"text": str, "ttl": int}` (default `ttl=8`, max 2 notes)
    """
    if not shared_cache_file or not bot_name:
        return {}
    ch_key = str(channel_id or 0)
    bot_key = bot_name.strip()

    if not tick_ttl:
        data = load_json_file(shared_cache_file, {})
        return copy.deepcopy(data.get("_bot_internal_state", {}).get(ch_key, {}).get(bot_key, {}))

    state_holder = {"state": {}}

    def _mutate(cache: dict) -> dict:
        all_states = cache.setdefault("_bot_internal_state", {})
        ch_states = all_states.setdefault(ch_key, {})
        b_state = ch_states.get(bot_key, {})
        if not isinstance(b_state, dict) or not b_state:
            return cache

        # Tick mood TTL
        mood_ttl = int(b_state.get("mood_ttl", 0)) - 1
        if mood_ttl <= 0:
            b_state.pop("mood", None)
            b_state.pop("mood_ttl", None)
        else:
            b_state["mood_ttl"] = mood_ttl

        # Tick inner_thought TTL
        thought_ttl = int(b_state.get("thought_ttl", 0)) - 1
        if thought_ttl <= 0:
            b_state.pop("inner_thought", None)
            b_state.pop("thought_ttl", None)
        else:
            b_state["thought_ttl"] = thought_ttl

        # Tick rotating character arc_notes TTL (prevents Flanderization!)
        raw_arcs = b_state.get("arc_notes", [])
        kept_arcs = []
        if isinstance(raw_arcs, list):
            for item in raw_arcs:
                if isinstance(item, dict) and item.get("text"):
                    rem = int(item.get("ttl", 1)) - 1
                    if rem > 0:
                        kept_arcs.append({"text": str(item["text"]).strip(), "ttl": rem})
        if kept_arcs:
            b_state["arc_notes"] = kept_arcs[-2:]
        else:
            b_state.pop("arc_notes", None)

        ch_states[bot_key] = b_state
        state_holder["state"] = copy.deepcopy(b_state)
        return cache

    update_json_file(shared_cache_file, {}, _mutate)
    return state_holder["state"]


def save_bot_thought_subtext(
    shared_cache_file: str,
    channel_id: int,
    bot_name: str,
    thought_text: str,
):
    """
    Saves the stripped `[THOUGHT: ...]` inner monologue from Phase A into
    `_bot_internal_state` (`ttl=2` turns) so the bot maintains subtext continuity
    and `orchestrator.py` can stream it to the Live Subconscious X-Ray Inspector.
    """
    if not shared_cache_file or not bot_name or not thought_text:
        return
    clean_t = thought_text.strip()
    if not clean_t:
        return
    ch_key = str(channel_id or 0)
    bot_key = bot_name.strip()

    def _mutate(cache: dict) -> dict:
        all_states = cache.setdefault("_bot_internal_state", {})
        ch_states = all_states.setdefault(ch_key, {})
        b_state = ch_states.setdefault(bot_key, {})
        b_state["inner_thought"] = clean_t[:220]
        b_state["thought_ttl"] = 2
        b_state["updated_at"] = time.time()
        return cache

    update_json_file(shared_cache_file, {}, _mutate)


def build_subconscious_context_injection(
    shared_cache_file: str,
    channel_id: int,
    bot_name: str,
    current_speaker: str | None = None,
    is_speaker_bot: bool = False,
) -> tuple[str, list[str]]:
    """
    Builds the Phase 1 Subconscious Context Block for `build_active_user_context_block`:
    - Reunion Radar Alert (`>= 48h` absence)
    - Pre-computed `_room_state` (`active_topic`, `room_vibe`, `pace_guidance`, `active_props`, `exhausted_jokes`)
    - Bot Internal State (`mood` with TTL=3, `inner_thought` with TTL=2, `arc_notes` with TTL=8 framed as 5% flavor)
    - Rolling Episodic Storyline Chapters (`_episodic_chapters`)
    - Optional `[THOUGHT: ...]` inner-monologue instruction
    Returns `(formatted_prompt_block, semantic_tags_list)`.
    """
    if not shared_cache_file:
        return "", []

    cache_data = load_json_file(shared_cache_file, {})
    ch_key = str(channel_id or 0)
    room_state = cache_data.get("_room_state", {}).get(ch_key, {})
    if not isinstance(room_state, dict):
        room_state = {}

    semantic_tags = [
        str(t).strip() for t in (room_state.get("semantic_tags") or []) if t and str(t).strip()
    ]
    bullets = []

    # 1. Reunion Radar check (>= 48h human return)
    if current_speaker and not is_speaker_bot:
        is_reunion, gap_secs = record_and_check_user_reunion(
            shared_cache_file, channel_id, current_speaker, is_speaker_bot=False
        )
        if is_reunion:
            days = max(2, int(round(gap_secs / 86400.0)))
            bullets.append(
                f"- **REUNION RADAR ALERT**: **{current_speaker}** just returned to chat after **{days} days** away! "
                "Briefly welcome them back or acknowledge their return in your signature persona style!"
            )

    # 2. Live Room Blackboard (`_room_state`)
    active_topic = str(room_state.get("active_topic", "") or "").strip()
    room_vibe = str(room_state.get("room_vibe", "") or "").strip()
    pace_guide = str(room_state.get("pace_guidance", "") or "").strip()
    if active_topic or room_vibe:
        topic_part = f"Active Topic: *{active_topic}*" if active_topic else ""
        vibe_part = f"Room Vibe: *{room_vibe}*" if room_vibe else ""
        join_str = " | ".join(p for p in (topic_part, vibe_part) if p)
        bullets.append(f"- **Live Room Blackboard**: {join_str}" + (f" — {pace_guide}" if pace_guide else ""))

    # Shared Physical Room Props Blackboard (Crown Jewel #2 — 5-turn TTL object permanence)
    raw_props = room_state.get("active_props", [])
    if isinstance(raw_props, list) and raw_props:
        prop_strs = [
            str(p.get("item", "")).strip()
            for p in raw_props
            if isinstance(p, dict) and p.get("item") and int(p.get("ttl", 0)) > 0
        ]
        if prop_strs:
            bullets.append(
                f"- **Active Room Props & Scene Objects (Optional Flavor)**: {', '.join(prop_strs[:3])}"
            )

    exhausted = [
        str(x).strip() for x in (room_state.get("exhausted_jokes") or []) if x and str(x).strip()
    ]
    if exhausted:
        bullets.append(
            f"- **Exhausted Bits (DO NOT REPEAT)**: {', '.join(exhausted[:4])} — these jokes are retired for this session."
        )

    # 3. Bot Carryover Mood, Subtext & TTL Character Arc (Strictly 5% Flavor Only)
    if bot_name:
        bot_state = cache_data.get("_bot_internal_state", {}).get(ch_key, {}).get(bot_name.strip(), {})
        if isinstance(bot_state, dict) and bot_state:
            mood_str = str(bot_state.get("mood", "") or "").strip()
            thought_str = str(bot_state.get("inner_thought", "") or "").strip()
            arc_list = [
                str(a.get("text", "")).strip()
                for a in (bot_state.get("arc_notes") or [])
                if isinstance(a, dict) and a.get("text")
            ]
            sub_parts = []
            if mood_str:
                sub_parts.append(f"Carryover Mood: *{mood_str}*")
            if thought_str:
                sub_parts.append(f"Previous Subtext: *\"{thought_str}\"*")
            if arc_list:
                sub_parts.append(f"Recent Mini-Arc: {'; '.join(arc_list[:2])}")
            if sub_parts:
                bullets.append(
                    "- **Your Internal Carryover State (5% Subtle Flavor Only — NEVER let this override your core persona bio!)**: "
                    + " | ".join(sub_parts)
                )

    # 4. Rolling Episodic Storyline Chapters
    chapters = cache_data.get("_episodic_chapters", {}).get(ch_key, [])
    if isinstance(chapters, list) and chapters:
        recent_ch = chapters[-2:]
        ch_summaries = [
            f"*{c.get('title', 'Session')}*: {c.get('summary', '')}"
            for c in recent_ch
            if isinstance(c, dict) and c.get("summary")
        ]
        if ch_summaries:
            bullets.append(f"- **Recent Server Storyline Chapters**: {' || '.join(ch_summaries)}")

    # 5. Hidden Inner-Monologue Subtext Tag Support
    bullets.append(
        "- **Optional Hidden Subtext (`[THOUGHT: ...]`)**: Before your spoken reply, you may optionally start with a "
        "single-sentence `[THOUGHT: <your brief private in-character motive or subtext>]` tag. It will be stripped automatically before sending to Discord."
    )

    return ("\n" + "\n".join(bullets)) if bullets else "", semantic_tags


def _extract_balanced_json(raw_text: str) -> str | None:
    if not raw_text:
        return None
    start = raw_text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
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
                    return raw_text[start : idx + 1]
    return None


async def run_subconscious_cognitive_turn(
    *,
    channel_id: int,
    trigger_message_id: int | str,
    speaker_name: str,
    user_message: str,
    bot_reply_text: str,
    bot_thought_text: str | None = None,
    my_bot_name: str,
    is_speaker_bot: bool = False,
    raw_history: list[dict],
    user_profiles_file: str,
    long_term_memory_file: str,
    general_knowledge_file: str,
    shared_cache_file: str,
    other_bot_names: set[str] | list[str] | None = None,
    api_key: str,
    api_url: str,
    subconscious_model: str | None = None,
) -> dict:
    """
    Phase 1B — Unified Subconscious Cognitive Engine (`Qwen/Qwen3-30B-A3B`):
    Runs asynchronously via `asyncio.create_task` AFTER `channel.send()` (0ms typing latency!).
    - Uses an atomic single-leader lock (`_subconscious_leader_msg[str(trigger_message_id)]`)
      in `shared_image_cache.json` so even if 3 bots reply to the same message in a 50-bot server,
      ONLY 1 shared 30B harvest call runs for room state/lore/user facts, while every responding
      bot ticks its own TTL state and saves its `[THOUGHT:]` subtext!
    """
    ch_key = str(channel_id or 0)
    msg_key = str(trigger_message_id or "")

    # 1. Tick this bot's existing TTL state and record any new [THOUGHT:] subtext
    get_and_tick_bot_internal_state(shared_cache_file, channel_id, my_bot_name, tick_ttl=True)
    if bot_thought_text:
        save_bot_thought_subtext(shared_cache_file, channel_id, my_bot_name, bot_thought_text)

    # 2. Claim Single-Leader Lock for this trigger_message_id
    leader_holder = {"is_leader": False}
    now_val = time.time()

    def _claim_leader(cache: dict) -> dict:
        leaders = cache.setdefault("_subconscious_leader_msg", {})
        # Prune old leader claims (> 10 minutes old)
        stale_keys = [k for k, v in leaders.items() if isinstance(v, dict) and (now_val - float(v.get("ts", 0))) > 600.0]
        for k in stale_keys:
            leaders.pop(k, None)

        if msg_key and msg_key not in leaders:
            leaders[msg_key] = {"bot": my_bot_name, "ts": now_val}
            leader_holder["is_leader"] = True
        elif not msg_key:
            leader_holder["is_leader"] = True
        return cache

    if shared_cache_file:
        update_json_file(shared_cache_file, {}, _claim_leader)
    else:
        leader_holder["is_leader"] = True

    if not leader_holder["is_leader"] or not api_key or not api_url:
        return {"status": "non_leader_skipped", "bot": my_bot_name}

    model_to_use = subconscious_model or (
        getattr(config, "FALLBACK_TEXT_MODELS", [DEFAULT_SUBCONSCIOUS_MODEL])[0]
        if getattr(config, "FALLBACK_TEXT_MODELS", None)
        else DEFAULT_SUBCONSCIOUS_MODEL
    )

    # Build compact recent transcript (last 8 messages) + velocity metrics
    recent_lines = []
    recent_ts = []
    recent_users_set: set[str] = {speaker_name} if speaker_name else set()
    for m in (raw_history or [])[-8:]:
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        a_name = str(m.get("author_name", "User") or "User").strip()
        recent_users_set.add(a_name)
        c_txt = str(m.get("content", "") or "").strip()
        if c_txt:
            recent_lines.append(f"[{a_name}]: {c_txt[:220]}")
        if isinstance(m.get("timestamp"), (int, float)):
            recent_ts.append(float(m["timestamp"]))

    exchange_summary = "\n".join(recent_lines[-6:])
    if bot_reply_text and my_bot_name:
        exchange_summary += f"\n[{my_bot_name} (Just Replied)]: {bot_reply_text[:240]}"

    msg_velocity_desc = "steady chat"
    if len(recent_ts) >= 3:
        avg_gap = (recent_ts[-1] - recent_ts[0]) / float(max(1, len(recent_ts) - 1))
        if avg_gap <= 15.0:
            msg_velocity_desc = "rapid-fire banter (<15s/msg)"
        elif avg_gap >= 300.0:
            msg_velocity_desc = "slow/relaxed pace (>5m/msg)"

    # Check if an episodic session chapter should be synthesized
    cache_snapshot = load_json_file(shared_cache_file, {}) if shared_cache_file else {}
    msg_counter_map = cache_snapshot.get("_episodic_msg_counter", {})
    msgs_since_chapter = int(msg_counter_map.get(ch_key, 0)) + 1
    needs_chapter = msgs_since_chapter >= EPISODIC_MIN_SESSION_MSGS

    sub_prompt = (
        "You are the Subconscious Cognitive Engine for a multi-bot Discord server.\n"
        f"Active Responder Bot: {my_bot_name}\n"
        f"Current Speaker: {speaker_name} ({'Fellow Bot' if is_speaker_bot else 'Human Member'})\n"
        f"Message Velocity: {msg_velocity_desc}\n\n"
        f"Recent Exchange:\n{exchange_summary}\n\n"
        "Analyze this exchange and return ONLY a single valid JSON object with these keys:\n"
        "{\n"
        '  "user_facts": [\n'
        '    {"user": "ExactSpeakerName", "fact": "Enduring personal fact or boundary (NO last names, addresses, emails, phones, cards, or ephemeral chat actions)", "salience": 75, "category": "CORE|BOUNDARY|PROJECT|PREF|REL|HABIT|OPINION|SITUATIONAL", "action": "ADD|SUPERSEDE"}\n'
        "  ],\n"
        '  "server_lore": [{"fact": "Enduring server event/running lore", "salience": 65}],\n'
        '  "general_facts": [{"fact": "Universal trivia/technical fact", "salience": 60}],\n'
        '  "rapport_updates": [\n'
        '    {"target": "SpeakerOrBotName", "delta": 4, "trend": "Warming Up", "catalyst": "Brief reason (<60 chars)"}\n'
        "  ],\n"
        '  "room_state": {\n'
        '    "active_topic": "3-6 word current topic",\n'
        '    "room_vibe": "2-4 word emotional vibe",\n'
        '    "conversational_pace": "rapid|normal|relaxed",\n'
        '    "pace_guidance": "1 short sentence of pacing advice",\n'
        '    "exhausted_jokes": ["any overused gag/word that should be dropped"],\n'
        '    "semantic_tags": ["3 to 6 lowercase topic/domain keywords for RAG"],\n'
        '    "active_props": ["up to 2 physical objects/snacks/items explicitly placed in the room"],\n'
        '    "open_hook": "unresolved user event/goal to check in on later, or empty string"\n'
        "  },\n"
        '  "bot_state": {\n'
        f'    "mood": "2-4 word carryover mood for {my_bot_name}",\n'
        f'    "arc_note": "Optional temporary character callback (<70 chars) or empty string"\n'
        "  }"
        + (
            ',\n  "episodic_chapter": {"title": "Short Chapter Title (3-6 words)", "summary": "1-2 sentence summary of this session storyline"}\n'
            if needs_chapter
            else "\n"
        )
        + "}\n"
        "STRICT PRIVACY & QUALITY RULES:\n"
        "1. NEVER extract last names/surnames, street addresses, emails, phone numbers, credit cards, or passwords (first names are fine).\n"
        "2. Only include `user_facts` if a speaker stated/confirmed a real trait, boundary, project, preference, or milestone. Leave `user_facts: []` if none!\n"
        "3. `salience` is 1-100: CORE/BOUNDARY=90-100, PROJECT/PREF/REL=70-89, HABIT/OPINION/LORE=40-69, SITUATIONAL=10-39."
    )

    payload = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": "Output only valid JSON."},
            {"role": "user", "content": sub_prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 520,
    }

    res = await post_with_backoff(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json_payload=payload,
        timeout=(6, 16),
        max_retries=1,
        bot_name=f"{my_bot_name}:Subconscious",
    )

    if not res or res.status_code != 200:
        return {"status": "api_skipped", "bot": my_bot_name}

    try:
        raw_content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        json_str = _extract_balanced_json(raw_content)
        if not json_str:
            return {"status": "no_json", "bot": my_bot_name}
        parsed = json.loads(json_str)
    except Exception as e:
        print(f"[{my_bot_name}] [SUBCONSCIOUS PARSE WARN] {e}")
        return {"status": "parse_error", "bot": my_bot_name}

    # Lazy import from memory_and_context to avoid circular imports at module load time
    from memory_and_context import (
        consolidate_and_append_fact,
        find_canonical_username,
        is_misattributed_hearsay,
        route_server_vs_general_facts,
        is_duplicate_fact,
    )

    # 3. Apply User Facts & Rapport Updates to `user_profiles.json`
    raw_u_facts = parsed.get("user_facts") or []
    raw_rapport = parsed.get("rapport_updates") or []
    if (raw_u_facts or raw_rapport) and user_profiles_file:
        def _mutate_profiles(profiles: dict) -> dict:
            for item in raw_u_facts:
                if not isinstance(item, dict):
                    continue
                u_raw = str(item.get("user", "") or "").strip()
                f_raw = str(item.get("fact", "") or "").strip().rstrip(".")
                if not u_raw or not f_raw:
                    continue
                canon_u = find_canonical_username(u_raw, profiles, recent_users_set)
                if not canon_u or canon_u.lower() == my_bot_name.lower():
                    continue
                if is_misattributed_hearsay(f_raw, canon_u, speaker_name, user_message):
                    continue
                safe_f = sanitize_or_reject_sensitive_pii(f_raw)
                if not safe_f:
                    continue
                sal = int(item.get("salience", 68) or 68)
                cat = str(item.get("category", "GEN") or "GEN").strip().upper()
                packed_f = pack_salience_fact(safe_f, salience=sal, category=cat)
                u_list = profiles.setdefault(canon_u, [])
                consolidate_and_append_fact(packed_f, u_list)

            for r_item in raw_rapport:
                if not isinstance(r_item, dict):
                    continue
                t_raw = str(r_item.get("target", "") or "").strip()
                if not t_raw:
                    continue
                canon_t = find_canonical_username(t_raw, profiles, recent_users_set | set(other_bot_names or []))
                if not canon_t or canon_t.lower() == my_bot_name.lower():
                    continue
                try:
                    r_delta = max(-15, min(15, int(r_item.get("delta", 0))))
                except ValueError:
                    r_delta = 0
                if r_delta == 0:
                    continue
                r_delta = apply_session_diminishing_returns(channel_id, canon_t, r_delta)
                r_trend = str(r_item.get("trend", "Steady") or "Steady").strip()[:40]
                r_cat = sanitize_or_reject_sensitive_pii(str(r_item.get("catalyst", "") or "").strip()[:75]) or ""
                u_list = profiles.setdefault(canon_t, [])
                consolidate_and_append_fact(
                    f"__RAPPORT_SHIFT__:{r_delta}|{r_trend}|{r_cat}",
                    u_list,
                )
            return profiles

        update_json_file(user_profiles_file, {}, _mutate_profiles)

    # 4. Route & Save Server Lore and General Knowledge Facts
    raw_srv = [
        str(x.get("fact", "") if isinstance(x, dict) else x).strip()
        for x in (parsed.get("server_lore") or [])
        if x
    ]
    raw_gen = [
        str(x.get("fact", "") if isinstance(x, dict) else x).strip()
        for x in (parsed.get("general_facts") or [])
        if x
    ]
    if raw_srv or raw_gen:
        routed_srv, routed_gen = route_server_vs_general_facts(
            raw_srv, raw_gen, recent_users_set, set(other_bot_names or []), my_bot_name
        )
        if routed_srv and long_term_memory_file:
            def _mutate_ltm(ltm: list) -> list:
                if not isinstance(ltm, list):
                    ltm = []
                plain_existing = [unpack_salience_fact(str(x))["text"] for x in ltm]
                for sf in routed_srv:
                    safe_sf = sanitize_or_reject_sensitive_pii(sf)
                    if safe_sf and not is_duplicate_fact(safe_sf, plain_existing, threshold=0.62):
                        ltm.append(pack_salience_fact(safe_sf, salience=65, category="LORE"))
                        plain_existing.append(safe_sf)
                return evict_lowest_retention_facts(ltm, max_items=120)

            update_json_file(long_term_memory_file, [], _mutate_ltm)

        if routed_gen and general_knowledge_file:
            def _mutate_gen(gk: list) -> list:
                if not isinstance(gk, list):
                    gk = []
                plain_existing = [unpack_salience_fact(str(x))["text"] for x in gk]
                for gf in routed_gen:
                    safe_gf = sanitize_or_reject_sensitive_pii(gf)
                    if safe_gf and not is_duplicate_fact(safe_gf, plain_existing, threshold=0.62):
                        gk.append(pack_salience_fact(safe_gf, salience=62, category="GEN"))
                        plain_existing.append(safe_gf)
                return evict_lowest_retention_facts(gk, max_items=120)

            update_json_file(general_knowledge_file, [], _mutate_gen)

    # 5. Update Shared `_room_state`, `_bot_internal_state`, and `_episodic_chapters`
    if shared_cache_file:
        def _mutate_shared(cache: dict) -> dict:
            # Room State Blackboard + 5-turn TTL Room Props
            r_in = parsed.get("room_state")
            if isinstance(r_in, dict):
                room_map = cache.setdefault("_room_state", {})
                cur_room = room_map.get(ch_key, {})
                if not isinstance(cur_room, dict):
                    cur_room = {}

                # Tick existing props TTL (5 turns) and merge any new props
                existing_props = cur_room.get("active_props", [])
                kept_props: list[dict] = []
                if isinstance(existing_props, list):
                    for p in existing_props:
                        if isinstance(p, dict) and p.get("item"):
                            rem_ttl = int(p.get("ttl", 1)) - 1
                            if rem_ttl > 0:
                                kept_props.append({"item": str(p["item"]).strip(), "ttl": rem_ttl})
                for new_p in (r_in.get("active_props") or []):
                    p_str = str(new_p.get("item", "") if isinstance(new_p, dict) else new_p).strip()
                    if p_str and not any(p_str.lower() == kp["item"].lower() for kp in kept_props):
                        kept_props.append({"item": p_str[:65], "ttl": 5})

                cur_room.update({
                    "active_topic": str(r_in.get("active_topic", "") or "")[:80],
                    "room_vibe": str(r_in.get("room_vibe", "") or "")[:50],
                    "conversational_pace": str(r_in.get("conversational_pace", "normal") or "normal")[:24],
                    "pace_guidance": str(r_in.get("pace_guidance", "") or "")[:140],
                    "exhausted_jokes": [
                        str(x).strip()[:45] for x in (r_in.get("exhausted_jokes") or []) if x
                    ][:6],
                    "semantic_tags": [
                        str(x).strip().lower()[:30] for x in (r_in.get("semantic_tags") or []) if x
                    ][:8],
                    "active_props": kept_props[-3:],
                    "updated_at": now_val,
                })
                open_hook = str(r_in.get("open_hook", "") or "").strip()
                if open_hook:
                    cur_room["open_hook"] = {
                        "speaker": speaker_name,
                        "hook": open_hook[:120],
                        "ts": now_val,
                    }
                room_map[ch_key] = cur_room

            # Bot Internal State (TTL Mood & Rotating Character Arc Note)
            b_in = parsed.get("bot_state")
            if isinstance(b_in, dict) and my_bot_name:
                all_b = cache.setdefault("_bot_internal_state", {})
                ch_b = all_b.setdefault(ch_key, {})
                b_st = ch_b.setdefault(my_bot_name.strip(), {})
                mood_val = str(b_in.get("mood", "") or "").strip()
                if mood_val:
                    b_st["mood"] = mood_val[:60]
                    b_st["mood_ttl"] = 3
                arc_val = str(b_in.get("arc_note", "") or "").strip()
                if arc_val:
                    arcs = b_st.get("arc_notes", [])
                    if not isinstance(arcs, list):
                        arcs = []
                    if not any(arc_val.lower() in str(a.get("text", "")).lower() for a in arcs if isinstance(a, dict)):
                        arcs.append({"text": arc_val[:85], "ttl": 8})
                    b_st["arc_notes"] = arcs[-2:]
                b_st["updated_at"] = now_val

            # Episodic Session Chapters
            cnt_map = cache.setdefault("_episodic_msg_counter", {})
            ep_in = parsed.get("episodic_chapter")
            if needs_chapter and isinstance(ep_in, dict) and ep_in.get("summary"):
                ch_map = cache.setdefault("_episodic_chapters", {})
                ch_list = ch_map.setdefault(ch_key, [])
                if not isinstance(ch_list, list):
                    ch_list = []
                ch_list.append({
                    "title": str(ep_in.get("title", "Server Session") or "Server Session").strip()[:65],
                    "summary": str(ep_in.get("summary", "")).strip()[:240],
                    "date": _today_utc_str(),
                    "timestamp": now_val,
                })
                ch_map[ch_key] = ch_list[-MAX_EPISODIC_CHAPTERS:]
                cnt_map[ch_key] = 0
            else:
                cnt_map[ch_key] = msgs_since_chapter

            # Save latest Subconscious X-Ray telemetry entry for `orchestrator.py` Web-UI
            xray_log = cache.setdefault("_subconscious_xray_log", [])
            if not isinstance(xray_log, list):
                xray_log = []
            xray_log.append({
                "ts": now_val,
                "channel_id": int(channel_id or 0),
                "bot": my_bot_name,
                "speaker": speaker_name,
                "thought": (bot_thought_text or "")[:180],
                "mood": str((parsed.get("bot_state") or {}).get("mood", ""))[:50],
                "active_topic": str((parsed.get("room_state") or {}).get("active_topic", ""))[:70],
                "room_vibe": str((parsed.get("room_state") or {}).get("room_vibe", ""))[:40],
                "facts_learned": len(raw_u_facts) + len(raw_srv) + len(raw_gen),
            })
            cache["_subconscious_xray_log"] = xray_log[-40:]
            return cache

        update_json_file(shared_cache_file, {}, _mutate_shared)

    return {"status": "ok", "bot": my_bot_name, "parsed": parsed}