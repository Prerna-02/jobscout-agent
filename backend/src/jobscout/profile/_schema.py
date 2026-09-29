"""The candidate profile the rest of the agent works with."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["intern", "junior", "mid", "senior", "lead"]
WorkMode = Literal["onsite", "hybrid", "remote", "any"]
RoleKind = Literal["job", "internship"]


class Skill(BaseModel):
    name: str
    in_cv: bool = Field(description="True when the skill is written in the CV text.")
    evidence: str | None = Field(default=None, description="The CV line where the skill was found.")


class Role(BaseModel):
    title: str
    employer: str | None = None
    kind: RoleKind = "job"
    start_text: str | None = None
    end_text: str | None = None
    start: date | None = None
    end: date | None = None
    is_current: bool = False


class CandidateProfile(BaseModel):
    """A structured view of a CV.

    Produced by the profile builder, then shown to the user for review. After
    review, `target_titles`, `work_mode` and the kept skills reflect what the
    user wants, not only what the CV says.
    """

    current_title: str | None = None
    summary: str = ""
    seniority: Seniority
    years_experience: float | None = Field(
        default=None, description="Computed from role dates. None when dates can't be read."
    )
    skills: list[Skill] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    extracurriculars: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    work_mode: WorkMode = "any"
    target_titles: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def flagged_skills(self) -> list[str]:
        """Skills the model listed that the CV never mentions. The user decides on these."""
        return [s.name for s in self.skills if not s.in_cv]

    @property
    def confirmed_skills(self) -> list[str]:
        return [s.name for s in self.skills if s.in_cv]
