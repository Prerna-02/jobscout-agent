"""Remove personal details before a CV leaves the process.

Matching jobs needs skills, roles and a city, not a phone number. On free LLM
tiers prompts may be used for training, and Langfuse stores every prompt, so
contact details and protected attributes are replaced with placeholders here.

City names are kept on purpose: the job search needs them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

PLACEHOLDER = {
    "name": "[NAME]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "link": "[LINK]",
    "address": "[ADDRESS]",
    "personal": "[REMOVED]",
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LINK = re.compile(
    r"(?i)(?:\bhttps?://|\bwww\.)[^\s)>\]|,;]+"
    r"|\b(?:linkedin\.com|github\.com|gitlab\.com|behance\.net|dribbble\.com|medium\.com"
    r"|kaggle\.com|leetcode\.com)/[^\s)>\]|,;]+"
)
_PHONE_CANDIDATE = re.compile(r"(?<![\w+])\+?\(?\d[\d \-.()]{6,}\d(?!\w)")

# Protected or identifying attributes. The label is kept; the value is removed.
_PERSONAL_LABELS = (
    r"date of birth|d\.?o\.?b\.?|birth ?date|age|gender|sex|marital status|nationality"
    r"|citizenship|religion|caste|father'?s name|mother'?s name|spouse(?:'s)? name"
    r"|passport(?: no\.?| number)?|aadhaar(?: no\.?| number)?|pan(?: no\.?| number)?"
    r"|ssn|social security(?: number)?"
)
_PERSONAL_FIELD = re.compile(rf"(?i)\b(?P<label>{_PERSONAL_LABELS})\s*:\s*(?P<value>[^|\n;]+)")
_ADDRESS_FIELD = re.compile(
    r"(?i)\b(?P<label>(?:permanent |current |residential |home |postal )?address)"
    r"\s*:\s*(?P<value>[^|\n;]+)"
)

_HEADER_WORDS = {"curriculum", "vitae", "resume", "résumé", "cv", "profile", "contact"}
_JOB_WORDS = {
    "analyst", "engineer", "developer", "manager", "scientist", "designer", "consultant",
    "intern", "student", "specialist", "architect", "lead", "officer", "executive",
    "associate", "administrator", "director", "head", "graduate", "fresher",
}  # fmt: skip
_NAME_WORD = re.compile(r"[A-Z][A-Za-z'.\-]*")


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return the text with personal details replaced, plus a count per kind."""
    counts: Counter[str] = Counter()
    name = _guess_name(text)

    def sub(pattern: re.Pattern[str], kind: str, source: str) -> str:
        result, n = pattern.subn(PLACEHOLDER[kind], source)
        counts[kind] += n
        return result

    def sub_value(pattern: re.Pattern[str], kind: str, source: str) -> str:
        def repl(match: re.Match[str]) -> str:
            counts[kind] += 1
            value = match.group("value")
            trailing = value[len(value.rstrip()) :]  # keep the space before a "|" separator
            return f"{match.group('label')}: {PLACEHOLDER[kind]}{trailing}"

        return pattern.sub(repl, source)

    text = sub_value(_PERSONAL_FIELD, "personal", text)
    text = sub_value(_ADDRESS_FIELD, "address", text)
    text = sub(_EMAIL, "email", text)
    text = sub(_LINK, "link", text)
    text = _redact_phones(text, counts)
    if name:
        text = sub(re.compile(re.escape(name), re.IGNORECASE), "name", text)

    return text, {kind: n for kind, n in counts.items() if n}


def mask_patterns(data: Any) -> Any:
    """Mask emails, links and phone numbers anywhere in a JSON-like value.

    Used as a second safety layer for tracing payloads, where the candidate's
    name is unknown, so only pattern-based redaction applies.
    """
    if isinstance(data, str):
        counts: Counter[str] = Counter()
        masked = _EMAIL.sub(PLACEHOLDER["email"], data)
        masked = _LINK.sub(PLACEHOLDER["link"], masked)
        return _redact_phones(masked, counts)
    if isinstance(data, dict):
        return {key: mask_patterns(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_patterns(item) for item in data]
    return data


def _redact_phones(text: str, counts: Counter[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = sum(ch.isdigit() for ch in candidate)
        international = candidate.startswith("+")
        # 10+ digits (or 8+ with a country code) separates phones from years,
        # date ranges like "2019 - 2021" and figures like "10,000".
        if (international and 8 <= digits <= 15) or (10 <= digits <= 15):
            counts["phone"] += 1
            return PLACEHOLDER["phone"]
        return candidate

    return _PHONE_CANDIDATE.sub(repl, text)


def _guess_name(text: str) -> str | None:
    """Most CVs start with the candidate's name. Look at the first few lines only."""
    checked = 0
    for line in text.split("\n"):
        line = line.strip(" |•·-–")
        if not line:
            continue
        checked += 1
        if checked > 3:
            return None
        words = [w for w in re.split(r"\s+", line) if w]
        if all(w.lower().strip(":") in _HEADER_WORDS for w in words):
            continue  # a "CURRICULUM VITAE" banner line
        candidate = re.split(r"\s[|•·–-]\s|,|\s{2,}", line)[0].strip()
        parts = candidate.split()
        if (
            2 <= len(parts) <= 4
            and all(_NAME_WORD.fullmatch(p) for p in parts)
            and not any(p.lower() in _JOB_WORDS | _HEADER_WORDS for p in parts)
        ):
            return candidate
    return None
