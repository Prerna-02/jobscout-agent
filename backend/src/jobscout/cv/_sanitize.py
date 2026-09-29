"""Normalize extracted text and strip characters that are invisible to a reader."""

from __future__ import annotations

import re
import unicodedata

# Zero-width, bidi-override and soft-hyphen characters: invisible on screen but
# read by an LLM, so they can hide instructions or split keywords.
_INVISIBLE = re.compile("[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def sanitize(text: str) -> tuple[str, int]:
    """Return clean text and the number of invisible characters removed.

    NFKC folds PDF ligatures ("ﬁ" -> "fi") and full-width letters, which keeps
    later keyword matching honest.
    """
    text = unicodedata.normalize("NFKC", text)
    text, invisible = _INVISIBLE.subn("", text)
    # Icon fonts put bullets and symbols in the Private Use Area.
    text = "".join(" " if unicodedata.category(ch) == "Co" else ch for ch in text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return text, invisible
