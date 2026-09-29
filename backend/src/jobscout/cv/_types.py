from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceFormat = Literal["pdf", "docx"]

RejectReason = Literal[
    "unsupported_type",
    "too_large",
    "too_many_pages",
    "too_long",
    "no_text",
    "unreadable",
]


@dataclass(frozen=True)
class CVLimits:
    max_bytes: int = 4 * 1024 * 1024
    max_pages: int = 10
    max_chars: int = 40_000


@dataclass(frozen=True)
class CVDocument:
    """A CV that is safe to send to an LLM.

    `text` has been sanitized and redacted. It is the only form of the CV that
    should ever leave the process. The original bytes are not kept.
    """

    text: str
    fingerprint: str
    source_format: SourceFormat
    page_count: int | None
    redactions: dict[str, int] = field(default_factory=dict)
    dropped_hidden_chars: int = 0


class CVRejected(ValueError):
    """The upload cannot be turned into a CVDocument. `message` is safe to show the user."""

    def __init__(self, reason: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason: RejectReason = reason
        self.message = message
