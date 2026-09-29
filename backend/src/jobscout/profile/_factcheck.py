"""Check whether a skill the LLM listed is actually written in the CV.

LLMs sometimes add skills that "usually go together" (Power BI -> Tableau).
The check is plain string matching with three tolerances:

1. Word-boundary match, so "Java" is not found inside "JavaScript" and "C" is
   not found inside "C++".
2. Spacing and punctuation differences: "Power BI" = "PowerBI", "Node.js" = "NodeJS".
3. Known aliases: "JS" = "JavaScript", "k8s" = "Kubernetes".

No fuzzy matching: a check that is too forgiving would verify invented skills.
"""

from __future__ import annotations

import re
import unicodedata

# Each set lists names for the same skill. Lowercase, as they appear after normalizing.
ALIAS_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group)
    for group in (
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
        {"microsoft excel", "ms excel", "excel"},
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
        {"user experience", "ux"},
        {"user interface", "ui"},
        {"continuous integration", "ci/cd"},
    )
)

_TOKEN = re.compile(r"[a-z0-9+#]+")
_MIN_COMPACT_LEN = 4  # shorter compact forms ("go", "r") would match by accident
_EVIDENCE_CHARS = 160


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


class SkillMatcher:
    """Index a CV once, then look up many skills quickly."""

    def __init__(self, cv_text: str, alias_groups: tuple[frozenset[str], ...] = ALIAS_GROUPS):
        self._lines = cv_text.split("\n")
        self._normalized = [_normalize(line) for line in self._lines]
        self._aliases = alias_groups
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
        compact = "".join(_TOKEN.findall(variant))
        if len(compact) >= _MIN_COMPACT_LEN:
            return self._compact_index.get(compact)
        return None
