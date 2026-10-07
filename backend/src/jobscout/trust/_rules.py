"""Fast, free rules for common job-scam patterns. Every flag says exactly what triggered it.

Rules catch the clear cases cheaply and can't be talked out of a decision by the
ad's own text. They miss reworded scams, which is why uncertain postings also get
an AI review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from jobscout.jobs import JobPosting

Level = Literal["ok", "caution", "high_risk"]
Severity = Literal["high", "caution"]

# A flag phrase right after one of these words is a promise, not a demand: "no registration fee".
_NEGATED = re.compile(r"\b(no|not|never|without|zero|free of|don't|do not|won't)\b[\w\s,-]{0,20}$")

_RULES: tuple[tuple[str, Severity, str, re.Pattern[str]], ...] = (
    (
        "asks_for_payment",
        "high",
        "The ad asks candidates to pay money",
        re.compile(
            r"\b(registration|processing|application|training|visa|security|placement|onboarding"
            r"|joining|interview|medical|admin(?:istration)?|kit)\s+(fees?|charges?|deposits?"
            r"|payments?)\b|\b(pay|send|deposit|transfer)\s+(?:a\s+|an\s+|the\s+)?(?:small\s+)?"
            r"(fee|deposit|amount|money)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unusual_payment",
        "high",
        "Mentions payment methods scammers favour",
        re.compile(
            r"\b(western union|moneygram|gift cards?|bitcoin"
            r"|crypto(?:currency)?\s+(?:wallet|payment))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sensitive_data",
        "caution",
        "Asks for sensitive documents or bank details up front",
        re.compile(
            r"\b(send|share|provide|submit|email)\b[\w\s,]{0,30}"
            r"\b(passport|bank (?:account|details)|credit card|national id|emirates id"
            r"|ssn|aadhaar)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_app_contact",
        "caution",
        "Asks you to make contact through a chat app",
        re.compile(r"\b(whatsapp|telegram|wechat|signal app)\b", re.IGNORECASE),
    ),
    (
        "personal_email",
        "caution",
        "Uses a personal email address instead of a company one",
        re.compile(
            r"[\w.+-]+@(?:gmail|yahoo|hotmail|outlook|aol|protonmail|proton)\.\w+", re.IGNORECASE
        ),
    ),
    (
        "too_good_to_be_true",
        "caution",
        "Promises high pay per day or hour with little explanation",
        re.compile(
            r"\b(earn|make)\s+(?:up to\s+)?(?:\$|usd|aed|inr|€|£)?\s?\d[\d,]*"
            r"\s*(?:\$|usd|aed|dirhams?)?\s*(?:per|a|/)\s*(day|hour)\b",
            re.IGNORECASE,
        ),
    ),
)
_HIDDEN_EMPLOYER = re.compile(
    r"^(confidential|undisclosed|private|hidden|anonymous|company confidential)"
    r"(\s+(company|client|employer|firm))?$",
    re.IGNORECASE,
)
_MIN_AD_CHARS = 300


@dataclass
class TrustFlag:
    kind: str
    severity: Severity
    text: str
    evidence: str | None = None  # the words in the ad that triggered it
    source: Literal["rules", "ai"] = "rules"


@dataclass
class RuleResult:
    level: Level
    flags: list[TrustFlag] = field(default_factory=list)
    good_signs: list[str] = field(default_factory=list)


def check_rules(job: JobPosting) -> RuleResult:
    text = f"{job.title}\n{job.description}"
    flags: list[TrustFlag] = []
    for kind, severity, message, pattern in _RULES:
        for match in pattern.finditer(text):
            if _NEGATED.search(text[max(0, match.start() - 30) : match.start()]):
                continue
            flags.append(TrustFlag(kind, severity, message, _excerpt(text, match)))
            break  # one flag per rule is enough

    if not job.company or _HIDDEN_EMPLOYER.match(job.company.strip()):
        flags.append(TrustFlag("hidden_employer", "caution", "The employer's name is hidden"))
    if len(job.description.strip()) < _MIN_AD_CHARS:
        flags.append(TrustFlag("thin_ad", "caution", "The ad is very short and gives few details"))

    good_signs: list[str] = []
    if job.apply_is_direct or _same_site(job.apply_link, job.employer_website):
        good_signs.append("You apply on the employer's own website")
    if job.employer_website:
        good_signs.append("The employer's website is listed")

    if any(f.severity == "high" for f in flags):
        level: Level = "high_risk"
    elif flags:
        level = "caution"
    else:
        level = "ok"
    return RuleResult(level, flags, good_signs)


def _same_site(link: str, website: str | None) -> bool:
    if not website:
        return False
    a, b = urlparse(link).hostname or "", urlparse(website).hostname or ""
    root = lambda host: ".".join(host.lower().removeprefix("www.").split(".")[-2:])  # noqa: E731
    return bool(a and b) and root(a) == root(b)


def _excerpt(text: str, match: re.Match[str]) -> str:
    start, end = max(0, match.start() - 40), min(len(text), match.end() + 40)
    return re.sub(r"\s+", " ", text[start:end]).strip()
