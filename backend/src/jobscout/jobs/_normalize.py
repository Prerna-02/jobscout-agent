"""Turn one raw JSearch job into a JobPosting, or drop it if it's unusable."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from ._models import JobPosting

_MAX_DESCRIPTION_CHARS = 6000  # plenty for scoring; some ads run past 10k characters
_UNITS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
}
_RELATIVE = re.compile(r"(\d+|an?)\+?\s*(minute|hour|day|week|month)s?\s+ago")


def normalize_job(raw: dict[str, Any], *, now: datetime) -> JobPosting | None:
    """None when the job has no id, title or apply link: it can't be shown or applied to."""
    job_id, title, link = raw.get("job_id"), _text(raw.get("job_title")), raw.get("job_apply_link")
    if not (job_id and title and link):
        return None

    return JobPosting(
        job_id=str(job_id),
        title=title,
        company=_text(raw.get("employer_name")),
        location=_location(raw),
        country=_text(raw.get("job_country")),
        is_remote=bool(raw.get("job_is_remote")),
        employment_types=[str(t) for t in raw.get("job_employment_types") or []],
        posted_at=_posted_at(raw, now),
        posted_text=_text(raw.get("job_posted_at")),
        description=(_text(raw.get("job_description")) or "")[:_MAX_DESCRIPTION_CHARS],
        apply_link=str(link),
        publisher=_text(raw.get("job_publisher")),
        salary=_salary(raw),
    )


def parse_relative_age(text: str | None, *, now: datetime) -> datetime | None:
    """ "3 days ago" -> now minus 3 days. "Just posted"/"today" -> now. Else None."""
    if not text:
        return None
    s = text.strip().lower()
    if s in {"just posted", "today", "just now", "new"}:
        return now
    if s == "yesterday":
        return now - timedelta(days=1)
    if m := _RELATIVE.search(s):
        count = 1 if m[1] in {"a", "an"} else int(m[1])
        return now - count * _UNITS[m[2]]
    return None


def _posted_at(raw: dict[str, Any], now: datetime) -> datetime | None:
    # Exact fields first; in practice JSearch often leaves them empty.
    if isinstance(ts := raw.get("job_posted_at_timestamp"), (int, float)):
        return datetime.fromtimestamp(ts, tz=UTC)
    if isinstance(iso := raw.get("job_posted_at_datetime_utc"), str):
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            pass
    return parse_relative_age(raw.get("job_posted_at"), now=now)


def _location(raw: dict[str, Any]) -> str | None:
    parts = [_text(raw.get(key)) for key in ("job_city", "job_state", "job_country")]
    joined = ", ".join(dict.fromkeys(p for p in parts if p))
    return joined or _text(raw.get("job_location"))


def _salary(raw: dict[str, Any]) -> str | None:
    if text := _text(raw.get("job_salary_string")):
        return text
    low, high = raw.get("job_min_salary"), raw.get("job_max_salary")
    if low is None and high is None:
        return None
    period = (_text(raw.get("job_salary_period")) or "").lower()
    amount = f"{low:,.0f}-{high:,.0f}" if low and high else f"{low or high:,.0f}"
    return f"{amount} per {period}" if period else amount


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None
