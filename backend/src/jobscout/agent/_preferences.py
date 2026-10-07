"""What the user confirms at the review step, before any search runs."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobscout.jobs import DatePosted, EmploymentType
from jobscout.profile import CandidateProfile
from jobscout.scoring import PRESETS, Rubric, suggest_preset

_SAFE_TEXT = re.compile(r"[\w .,&+#/()'-]+")


class SearchPreferences(BaseModel):
    """The user's answer to the review step. It overrides what the CV implies."""

    model_config = ConfigDict(extra="forbid")

    target_titles: list[str] = Field(min_length=1, max_length=3)
    country: str = Field(description="Two-letter country code, e.g. 'ae'.")
    city: str | None = Field(default=None, max_length=40)
    remote_only: bool = False
    date_posted: DatePosted = "week"
    employment_types: tuple[EmploymentType, ...] = ()
    skills: list[str] = Field(default_factory=list, max_length=80)  # same cap as the profile
    rubric: Rubric | None = Field(
        default=None, description="Scoring weights; None means the suggested preset."
    )

    @field_validator("target_titles", "skills")
    @classmethod
    def _clean_items(cls, values: list[str]) -> list[str]:
        cleaned = [re.sub(r"\s+", " ", v).strip() for v in values]
        for item in cleaned:
            if not item or len(item) > 60 or not _SAFE_TEXT.fullmatch(item):
                raise ValueError(f"'{item}' is not a valid job title or skill")
        return list(dict.fromkeys(cleaned))

    @field_validator("country", mode="before")
    @classmethod
    def _clean_country(cls, value: str) -> str:
        value = str(value).strip().lower()
        if not re.fullmatch(r"[a-z]{2}", value):
            raise ValueError("country must be a two-letter code such as 'ae' or 'us'")
        return value

    @field_validator("city")
    @classmethod
    def _clean_city(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not _SAFE_TEXT.fullmatch(value):
            raise ValueError("city contains characters a place name never needs")
        return value


def suggest_preferences(profile: CandidateProfile) -> dict[str, object]:
    """Starting values for the review screen. Titles may be empty: then the user must add one."""
    city = profile.locations[0].split(",")[0].strip() if profile.locations else None
    return {
        "target_titles": profile.target_titles[:3],
        "country": None,
        "city": city,
        "remote_only": profile.work_mode == "remote",
        "date_posted": "week",
        "skills": profile.confirmed_skills,
        "preset": suggest_preset(profile),
    }


def resolve_rubric(prefs: SearchPreferences, profile: CandidateProfile) -> Rubric:
    """The user's weights if they set any, otherwise the preset that fits the profile."""
    return prefs.rubric or PRESETS[suggest_preset(profile)]
