"""CV intake: bytes in, LLM-safe text out.

    from jobscout.cv import read_cv
    cv = read_cv(uploaded_bytes)      # raises CVRejected with a user-facing message
    cv.text                           # sanitized, hidden text removed, personal details redacted

`read_cv` hides file-type detection (by content, not extension), PDF and DOCX
parsing, hidden-text removal, Unicode cleanup, size limits and redaction.
"""

from __future__ import annotations

import hashlib

from jobscout.settings import get_settings

from ._extract import detect_format, extract
from ._redact import mask_patterns
from ._redact import redact as _redact
from ._sanitize import sanitize
from ._types import CVDocument, CVLimits, CVRejected

__all__ = ["CVDocument", "CVLimits", "CVRejected", "mask_patterns", "read_cv"]

_MIN_WORDS = 30


def read_cv(data: bytes, *, limits: CVLimits | None = None) -> CVDocument:
    """Validate, parse, clean and redact an uploaded CV."""
    limits = limits or _default_limits()

    if not data:
        raise CVRejected("no_text", "The file is empty.")
    if len(data) > limits.max_bytes:
        raise CVRejected(
            "too_large",
            f"The file is {len(data) / 1_048_576:.1f} MB. "
            f"The limit is {limits.max_bytes / 1_048_576:.0f} MB.",
        )

    fmt = detect_format(data)
    if fmt is None:
        raise CVRejected("unsupported_type", "Upload a PDF or a Word (.docx) file.")

    raw = extract(data, fmt, limits)
    text, invisible = sanitize(raw.text)

    if len(text.split()) < _MIN_WORDS:
        raise CVRejected(
            "no_text",
            "Almost no readable text was found. If this is a scanned PDF, "
            "upload a text-based PDF or a Word file instead.",
        )
    if len(text) > limits.max_chars:
        raise CVRejected(
            "too_long",
            f"The CV is {len(text):,} characters long. The limit is {limits.max_chars:,}.",
        )

    redacted, counts = _redact(text)
    return CVDocument(
        text=redacted,
        fingerprint=hashlib.sha256(data).hexdigest()[:16],
        source_format=fmt,
        page_count=raw.page_count,
        redactions=counts,
        dropped_hidden_chars=raw.dropped_hidden_chars + invisible,
    )


def _default_limits() -> CVLimits:
    s = get_settings()
    return CVLimits(max_bytes=s.cv_max_bytes, max_pages=s.cv_max_pages, max_chars=s.cv_max_chars)
