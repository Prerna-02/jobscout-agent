"""Remove duplicates and postings that can't be a match, and say why each was removed."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ._models import JobPosting


@dataclass(frozen=True)
class FilterRules:
    max_age_days: int | None = None
    remote_only: bool = False
    countries: frozenset[str] = frozenset()  # empty means any country
    exclude_ids: frozenset[str] = frozenset()  # jobs already seen in this run


@dataclass
class FilterResult:
    kept: list[JobPosting]
    dropped: dict[str, int] = field(default_factory=dict)


def filter_jobs(jobs: Iterable[JobPosting], rules: FilterRules, *, now: datetime) -> FilterResult:
    """Keep order, drop the rest. Unknown posting dates and countries are kept, not guessed."""
    kept: list[JobPosting] = []
    dropped: Counter[str] = Counter()
    seen_ids: set[str] = set(rules.exclude_ids)
    seen_ads: set[tuple[str, str]] = set()
    countries = {c.upper() for c in rules.countries}
    oldest = now - timedelta(days=rules.max_age_days) if rules.max_age_days else None

    for job in jobs:
        # The same ad is often syndicated by several job boards under different ids.
        ad = (_fold(job.title), _fold(job.company or ""))
        if job.job_id in seen_ids or ad in seen_ads:
            reason = "duplicate"
        elif rules.remote_only and not job.is_remote:
            reason = "not_remote"
        elif countries and not job.is_remote and _other_country(job, countries):
            reason = "wrong_country"
        elif oldest and job.posted_at and job.posted_at < oldest:
            reason = "too_old"
        else:
            kept.append(job)
            seen_ids.add(job.job_id)
            seen_ads.add(ad)
            continue
        dropped[reason] += 1

    return FilterResult(kept=kept, dropped=dict(dropped))


def _other_country(job: JobPosting, countries: set[str]) -> bool:
    # A blank country is kept: JSearch was already asked for this country, and it
    # often leaves the field empty on ads that only name a city.
    return bool(job.country) and job.country.upper() not in countries


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
