"""Check that a quote an LLM gave as proof really appears in a text.

Used wherever an LLM must back a judgement with evidence (match scoring, trust
checks). Matching is word-based: case, punctuation and spacing differences are
ignored, and "..." marks skipped words, but every quoted word must be there.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["fold_text", "quote_in_text"]


def fold_text(text: str) -> str:
    """Words only: lowercase, punctuation removed (except + and # for C++ and C#)."""
    text = unicodedata.normalize("NFKC", text).lower().replace("&", " and ")
    return " " + " ".join(re.findall(r"[a-z0-9+#]+", text)) + " "


def quote_in_text(quote: str | None, folded_text: str) -> bool:
    """True when every part of the quote (split at "...") is in the already-folded text."""
    if not quote:
        return False
    parts = [fold_text(part) for part in re.split(r"\.\.\.|…", quote)]
    parts = [part for part in parts if part.strip()]
    return bool(parts) and all(part in folded_text for part in parts)
