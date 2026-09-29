"""Dates and years of experience, computed in code rather than guessed by the LLM.

The LLM copies dates as the CV writes them ("Sept 2021", "03/2020", "2019",
"Present"). This module turns them into dates, merges overlapping roles so
parallel jobs aren't double-counted, and derives a seniority level.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

from ._schema import Role, Seniority

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    "spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12,
}  # fmt: skip
_PRESENT = ("present", "current", "now", "ongoing", "till date", "to date", "today")
_EARLIEST_YEAR = 1950


def parse_cv_date(text: str | None, *, today: date, is_end: bool) -> date | None:
    """Parse a CV date. End dates are exclusive: "Dec 2021" ends on 1 Jan 2022.

    Returns None for anything unrecognised rather than guessing.
    """
    if not text:
        return None
    s = re.sub(r"\s+", " ", text.strip().lower().rstrip("."))
    if any(word in s for word in _PRESENT):
        return today if is_end else None

    year: int | None = None
    month: int | None = None
    if m := re.fullmatch(r"(?:\d{1,2}(?:st|nd|rd|th)? )?([a-z]+)\.?,? '?(\d{4}|\d{2})", s):
        month, year = _MONTHS.get(m[1]), _year(m[2], today)
    elif m := re.fullmatch(r"(\d{1,2}) ?[/.-] ?(\d{4})", s):
        month, year = int(m[1]), int(m[2])
    elif m := re.fullmatch(r"(\d{4}) ?[/.-] ?(\d{1,2})", s):
        year, month = int(m[1]), int(m[2])
    elif m := re.fullmatch(r"(\d{4})", s):
        year = int(m[1])
        month = 12 if is_end else 1

    if year is None or month is None or not 1 <= month <= 12:
        return None
    if not _EARLIEST_YEAR <= year <= today.year + 1:
        return None

    result = _next_month(year, month) if is_end else date(year, month, 1)
    return min(result, today) if is_end else result


def total_years(roles: Iterable[Role]) -> float | None:
    """Years across job roles (not internships), counting overlaps once.

    None when there are job roles but none have readable dates. 0.0 when there
    are no job roles at all, which is correct for a student.
    """
    jobs = [r for r in roles if r.kind == "job"]
    if not jobs:
        return 0.0
    intervals = sorted((r.start, r.end) for r in jobs if r.start and r.end and r.end > r.start)
    if not intervals:
        return None

    merged: list[tuple[date, date]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    days = sum((end - start).days for start, end in merged)
    return round(days / 365.25, 1)


def derive_seniority(
    current_title: str | None, roles: list[Role], years: float | None
) -> Seniority:
    """Title keywords win; otherwise years decide."""
    title = (current_title or "").lower()
    if re.search(r"\b(intern|trainee|apprentice)\b", title):
        return "intern"
    if re.search(r"\b(lead|principal|staff|head|director|vp)\b", title):
        return "lead"
    if re.search(r"\b(senior|sr)\b", title):
        return "senior"
    if re.search(r"\b(junior|jr|graduate|entry)\b", title):
        return "junior"

    if not any(r.kind == "job" for r in roles):
        return "junior"  # students and graduates search for entry-level roles
    if years is None:
        return "mid"
    if years < 2:
        return "junior"
    if years < 5:
        return "mid"
    return "senior"


def _year(text: str, today: date) -> int:
    value = int(text)
    if value >= 100:
        return value
    # Two-digit years: "'21" is 2021, "'98" is 1998.
    return 2000 + value if value <= today.year % 100 + 1 else 1900 + value


def _next_month(year: int, month: int) -> date:
    return date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
