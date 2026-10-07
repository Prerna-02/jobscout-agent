"""Scoring presets: what matters for a match, and how much.

A rubric is data, not code: users can pick a preset, change weights, or add their
own criteria (up to 10). Weights are relative; totals are normalized to 100.
"""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jobscout.profile import CandidateProfile

_SAFE_TEXT = re.compile(r"[\w .,&+#/()'?:;-]+")
# Criteria about protected traits are refused: they must never influence a ranking.
_PROTECTED = re.compile(
    r"\b(age|aged|gender|sex|male|female|religio\w*|caste|race|racial|ethnic\w*|nationality"
    r"|citizenship|marital|married|pregnan\w*|disabilit\w*|photo|appearance|looks|beauty)\b",
    re.IGNORECASE,
)


class Criterion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(pattern=r"^[a-z][a-z0-9_]{1,29}$")
    label: str = Field(min_length=2, max_length=40)
    compare: str = Field(min_length=5, max_length=200, description="What the scorer compares.")
    weight: int = Field(ge=0, le=100)

    @field_validator("label", "compare")
    @classmethod
    def _safe(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        if not _SAFE_TEXT.fullmatch(value):
            raise ValueError("use letters, numbers and basic punctuation only")
        if _PROTECTED.search(value):
            raise ValueError("criteria about protected personal traits are not allowed")
        return value


class Rubric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=2, max_length=40)
    version: int = 1
    search_hints: tuple[str, ...] = Field(default=(), max_length=5)
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def _check(self) -> Rubric:
        keys = [c.key for c in self.criteria]
        if len(keys) != len(set(keys)):
            raise ValueError("each criterion needs a different key")
        if not any(c.weight > 0 for c in self.criteria):
            raise ValueError("at least one criterion needs a weight above 0")
        return self

    def fingerprint(self) -> str:
        """Changes whenever anything that affects scores changes; used in cache keys."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _c(key: str, label: str, compare: str, weight: int) -> Criterion:
    return Criterion(key=key, label=label, compare=compare, weight=weight)


PRESETS: dict[str, Rubric] = {
    "professional": Rubric(
        name="Professional",
        criteria=(
            _c("skills", "Skills", "Skills the job asks for vs the candidate's skills", 40),
            _c("experience", "Experience", "Years and seniority asked for vs the candidate's", 20),
            _c("role_fit", "Role fit", "Job title and duties vs the candidate's target roles", 15),
            _c("location", "Location", "Job location and work mode vs the candidate's choice", 10),
            _c("domain", "Domain", "Industry or product area vs the candidate's domains", 10),
            _c("education", "Education", "Degree or field required vs the candidate's", 5),
        ),
    ),
    "student": Rubric(
        name="Student",
        search_hints=("internship", "entry level", "graduate"),
        criteria=(
            _c("skills", "Skills", "Skills the job asks for vs the candidate's skills", 30),
            _c("projects", "Projects", "Projects using similar tools or problems", 20),
            _c("certifications", "Certifications", "Certificates the job asks for or values", 15),
            _c("education", "Education", "Degree and field required vs the candidate's", 15),
            _c("internships", "Internships", "Internship work relevant to this job", 10),
            _c("extracurriculars", "Activities", "Clubs, hackathons, volunteering for the role", 5),
            _c("location", "Location", "Job location and work mode vs the candidate's choice", 5),
        ),
    ),
}  # fmt: skip


def suggest_preset(profile: CandidateProfile) -> str:
    """Students and graduates without job roles get the Student preset."""
    return "professional" if any(r.kind == "job" for r in profile.roles) else "student"
