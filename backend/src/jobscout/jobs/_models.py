"""The shapes that go into and come out of a job search."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobscout.cv import mask_patterns

DatePosted = Literal["all", "today", "3days", "week", "month"]
EmploymentType = Literal["FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"]

_PLACEHOLDER = re.compile(r"\[(?:NAME|EMAIL|PHONE|LINK|ADDRESS|REMOVED)\]")
_CACHE_VERSION = "v1"  # bump when JobPosting or normalization changes


class SearchQuery(BaseModel):
    """What the agent is allowed to ask JSearch. Validated before any network call.

    The LLM fills this model; it never writes URLs or HTTP parameters itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(
        min_length=2,
        max_length=100,
        description="Job title and optional place, e.g. 'data analyst in Dubai'.",
    )
    country: str = Field(default="us", description="Two-letter country code, e.g. 'ae'.")
    date_posted: DatePosted = "week"
    remote_only: bool = False
    employment_types: tuple[EmploymentType, ...] = ()

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        if _PLACEHOLDER.search(value) or mask_patterns(value) != value:
            raise ValueError("search text must not contain personal details")
        if any(ch in value for ch in "{}<>\\"):
            raise ValueError("search text contains characters a job search never needs")
        return value

    @field_validator("country", mode="before")
    @classmethod
    def _clean_country(cls, value: str) -> str:
        value = str(value).strip().lower()
        if not re.fullmatch(r"[a-z]{2}", value):
            raise ValueError("country must be a two-letter code such as 'us' or 'ae'")
        return value

    @field_validator("employment_types")
    @classmethod
    def _sort_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))  # same filters in any order share one cache entry

    def to_params(self) -> dict[str, str]:
        params = {
            "query": self.text,
            "country": self.country,
            # Without this JSearch answers in the country's language (Arabic for "ae").
            "language": "en",
            "date_posted": self.date_posted,
        }
        if self.remote_only:
            params["work_from_home"] = "true"
        if self.employment_types:
            params["employment_types"] = ",".join(self.employment_types)
        return params

    def cache_key(self) -> str:
        canonical = "&".join(f"{k}={v}" for k, v in sorted(self.to_params().items()))
        digest = hashlib.sha256(canonical.lower().encode()).hexdigest()[:32]
        return f"jobscout:jsearch:{_CACHE_VERSION}:{digest}"


class JobPosting(BaseModel):
    """One real job returned by JSearch, trimmed to what matching needs."""

    job_id: str
    title: str
    company: str | None = None
    location: str | None = None
    country: str | None = None
    is_remote: bool = False
    employment_types: list[str] = Field(default_factory=list)
    posted_at: datetime | None = None
    posted_text: str | None = None
    description: str = ""
    apply_link: str
    apply_is_direct: bool = False  # True when the link goes to the employer's own site
    employer_website: str | None = None
    publisher: str | None = None
    salary: str | None = None


class SearchResult(BaseModel):
    query: SearchQuery
    jobs: list[JobPosting]
    cache_hit: bool
    fetched_at: datetime
