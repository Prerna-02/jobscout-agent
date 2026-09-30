"""Check whether a skill the LLM listed is actually written in the CV.

LLMs sometimes add skills that "usually go together" (Power BI -> Tableau).
The check is plain string matching, tried in this order:

1. The exact phrase, on word boundaries, so "Java" is not found inside
   "JavaScript" and "C" is not found inside "C++".
2. The phrase with spacing and punctuation removed: "Power BI" = "PowerBI".
3. Every word of the skill on the same CV line (or a line and the next one,
   for PDF line wraps). The CV may say "AWS ECR & ECS (Fargate)" while the LLM
   returns "AWS ECS (Fargate)"; all three words are on that line, so it counts.
4. Aliases: a built-in list ("JS" = "JavaScript", "CRM" = "Customer
   Relationship Management") plus abbreviations the CV defines itself, such as
   "Key Performance Indicators (KPIs)".

No fuzzy matching: a check that is too forgiving would verify invented skills.
"""

from __future__ import annotations

import re
import unicodedata

# Each set lists names for the same skill, lowercase as they appear after normalizing.
_TECH_ALIASES = (
    {"javascript", "js", "ecmascript"},
    {"typescript", "ts"},
    {"kubernetes", "k8s"},
    {"postgresql", "postgres", "psql"},
    {"machine learning", "ml"},
    {"natural language processing", "nlp"},
    {"large language models", "llms", "llm"},
    {"generative ai", "genai", "gen ai"},
    {"artificial intelligence", "ai"},
    {"amazon web services", "aws"},
    {"google cloud platform", "google cloud", "gcp"},
    {"microsoft azure", "azure"},
    {"power bi", "microsoft power bi"},
    {"scikit-learn", "sklearn"},
    {"node.js", "node"},
    {"react", "react.js"},
    {"vue", "vue.js"},
    {"next.js", "nextjs"},
    {"c#", "csharp"},
    {"c++", "cpp"},
    {"golang", "go"},
    {"mongodb", "mongo"},
    {"pytorch", "torch"},
    {"microsoft sql server", "sql server", "mssql"},
    {"rest api", "rest apis", "restful api", "restful apis"},
    {"object oriented programming", "oop"},
    {"continuous integration", "ci/cd"},
    {"elastic container service", "ecs"},
    {"elastic container registry", "ecr"},
)
_BUSINESS_ALIASES = (
    {"customer relationship management", "crm"},
    {"enterprise resource planning", "erp"},
    {"search engine optimization", "search engine optimisation", "seo"},
    {"search engine marketing", "sem"},
    {"pay per click", "pay-per-click", "ppc"},
    {"social media marketing", "smm"},
    {"key performance indicators", "key performance indicator", "kpis", "kpi"},
    {"return on investment", "roi"},
    {"business to business", "business-to-business", "b2b"},
    {"business to consumer", "business-to-consumer", "b2c"},
    {"profit and loss", "p&l"},
    {"point of sale", "pos"},
    {"human resources", "hr"},
    {"quality assurance", "qa"},
    {"standard operating procedures", "standard operating procedure", "sops", "sop"},
    {"customer service", "customer support"},
    {"microsoft excel", "ms excel", "excel"},
    {"microsoft word", "ms word"},
    {"microsoft powerpoint", "ms powerpoint", "powerpoint"},
    {"microsoft office", "ms office", "microsoft 365", "office 365"},
    {"google analytics", "ga4"},
)
ALIAS_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group) for group in (*_TECH_ALIASES, *_BUSINESS_ALIASES)
)

# Words that describe a skill rather than name it. Ignored by the same-line rule,
# so "Communication skills" is found in "Strong communication with clients".
_FILLER = frozenset(
    {
        "a", "an", "and", "the", "of", "in", "on", "for", "with", "to", "using", "via",
        "skill", "skills", "knowledge", "experience", "proficiency", "proficient",
        "basic", "basics", "intermediate", "advanced", "expert", "strong", "tool", "tools",
    }
)  # fmt: skip
_ACRONYM_CONNECTORS = frozenset({"and", "of", "&", "the", "for", "to", "in"})

_TOKEN = re.compile(r"[a-z0-9+#]+")
_MIN_COMPACT_LEN = 4  # shorter compact forms ("go", "r") would match by accident
_EVIDENCE_CHARS = 160

# "Customer Relationship Management (CRM)" and "CRM (Customer Relationship Management)"
_LONG_THEN_SHORT = re.compile(r"([A-Za-z][A-Za-z&\- ]{2,80}?)\s*\(\s*([A-Za-z][A-Za-z&]{1,9})\s*\)")
_SHORT_THEN_LONG = re.compile(r"\b([A-Z][A-Za-z&]{1,9})\s*\(\s*([A-Za-z][A-Za-z&\- ]{2,80}?)\s*\)")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def find_defined_acronyms(cv_text: str) -> list[frozenset[str]]:
    """Alias groups for abbreviations the CV spells out, e.g. "Search Engine Optimization (SEO)".

    A pair only counts when the abbreviation's letters are the initials of the long
    form, so "ECS (Fargate)" or "Python (Pandas)" are not treated as definitions.
    """
    groups: list[frozenset[str]] = []
    for pattern, long_first in ((_LONG_THEN_SHORT, True), (_SHORT_THEN_LONG, False)):
        for match in pattern.finditer(cv_text):
            before, inside = match.group(1), match.group(2)
            short, long_text = (inside, before) if long_first else (before, inside)
            long_form = _expansion_for(short, long_text, from_end=long_first)
            if long_form:
                groups.append(frozenset({_normalize(short), _normalize(long_form)}))
    return groups


def _expansion_for(short: str, text: str, *, from_end: bool) -> str | None:
    # A lowercase trailing "s" is a plural ("KPIs"); an uppercase one is a letter ("AWS").
    singular = short[:-1] if short.endswith("s") and short[:-1].isupper() else short
    letters = re.sub(r"[^a-z]", "", singular.lower())
    words = text.split()
    # The long form is the last few words before "(ABBR)", or all the words inside "ABBR (...)".
    for size in range(len(letters), len(letters) + 3):
        chunk = words[-size:] if from_end else words[:size]
        if len(chunk) != size:
            break
        initials = "".join(w[0] for w in chunk if w.lower() not in _ACRONYM_CONNECTORS)
        if initials.lower() == letters:
            return " ".join(chunk)
    return None


class SkillMatcher:
    """Index a CV once, then look up many skills quickly."""

    def __init__(self, cv_text: str, alias_groups: tuple[frozenset[str], ...] = ALIAS_GROUPS):
        self._lines = cv_text.split("\n")
        self._normalized = [_normalize(line) for line in self._lines]
        self._line_tokens = [frozenset(_TOKEN.findall(line)) for line in self._normalized]
        self._aliases = (*alias_groups, *find_defined_acronyms(cv_text))
        # "power bi" is indexed as "power", "bi" and "powerbi" (up to 3 joined tokens),
        # pointing at the first line where it appears.
        self._compact_index: dict[str, int] = {}
        for i, line in enumerate(self._normalized):
            tokens = _TOKEN.findall(line)
            for n in (1, 2, 3):
                for j in range(len(tokens) - n + 1):
                    self._compact_index.setdefault("".join(tokens[j : j + n]), i)

    def find(self, skill: str) -> str | None:
        """Return the CV line that mentions `skill` (or an alias), or None."""
        for variant in self._variants(skill):
            line_no = self._find_variant(variant)
            if line_no is not None:
                return self._lines[line_no].strip()[:_EVIDENCE_CHARS]
        return None

    def _variants(self, skill: str) -> list[str]:
        norm = _normalize(skill)
        variants = [norm]
        for group in self._aliases:
            if norm in group:
                variants.extend(sorted(group - {norm}))
        return variants

    def _find_variant(self, variant: str) -> int | None:
        if not variant:
            return None
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(variant) + r"(?![a-z0-9+#])")
        for i, line in enumerate(self._normalized):
            if pattern.search(line):
                return i

        tokens = _TOKEN.findall(variant)
        compact = "".join(tokens)
        if len(compact) >= _MIN_COMPACT_LEN and compact in self._compact_index:
            return self._compact_index[compact]

        return self._all_words_nearby([t for t in tokens if t not in _FILLER])

    def _all_words_nearby(self, words: list[str]) -> int | None:
        """First line where every word appears on that line or the next (PDFs wrap lines)."""
        if not words:
            return None
        last = len(self._line_tokens) - 1
        for i, tokens in enumerate(self._line_tokens):
            window = tokens | self._line_tokens[i + 1] if i < last else tokens
            if all(word in window for word in words):
                return i
        return None
