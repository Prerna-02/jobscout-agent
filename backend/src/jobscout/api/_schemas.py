"""What the website sends and receives. These models also generate the docs at /docs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jobscout.agent import FinishedRun, PausedRun, SearchPreferences
from jobscout.jobs import DatePosted, EmploymentType
from jobscout.profile import CandidateProfile
from jobscout.scoring import PRESETS, JobMatch, Rubric, build_rubric
from jobscout.trust import TrustReport


class StartResponse(BaseModel):
    run_id: str
    status: Literal["paused"] = "paused"
    profile: CandidateProfile
    suggested: dict[str, Any] = Field(description="Starting values for the review screen.")
    presets: dict[str, Rubric] = Field(description="Scorecards the user can choose from.")


class ResumeRequest(BaseModel):
    """The user's answers on the review screen."""

    model_config = ConfigDict(extra="forbid")

    target_titles: list[str] = Field(min_length=1, max_length=3, examples=[["Data Scientist"]])
    country: str = Field(examples=["ae"])
    city: str | None = Field(default=None, examples=["Dubai"])
    remote_only: bool = False
    date_posted: DatePosted = "week"
    employment_types: tuple[EmploymentType, ...] = ()
    skills: list[str] | None = Field(
        default=None, description="Skills to search with. Omit to use the skills found in the CV."
    )
    preset: str | None = Field(default=None, examples=["student"])
    weights: dict[str, int] | None = Field(
        default=None, description="Weight changes for the preset.", examples=[{"skills": 50}]
    )
    rubric: Rubric | None = Field(default=None, description="Fully custom criteria.")

    def to_preferences(self, profile: CandidateProfile) -> SearchPreferences:
        """Raises ValueError (or pydantic's ValidationError) with a message fit for the user."""
        return SearchPreferences(
            target_titles=self.target_titles,
            country=self.country,
            city=self.city,
            remote_only=self.remote_only,
            date_posted=self.date_posted,
            employment_types=self.employment_types,
            skills=self.skills if self.skills is not None else profile.confirmed_skills,
            rubric=build_rubric(self.preset, self.weights, self.rubric),
        )


class ResultResponse(BaseModel):
    run_id: str
    status: Literal["finished"] = "finished"
    outcome: str
    rubric_name: str | None
    matches: list[JobMatch] = Field(description="Ranked best first.")
    jobs_found: int
    searches: list[dict[str, Any]]
    removed_as_risky: list[dict[str, str]]
    dropped: dict[str, int]
    warnings: list[str]
    trust: dict[str, TrustReport]
    llm_tokens: dict[str, dict[str, int]]


def start_response(paused: PausedRun) -> StartResponse:
    return StartResponse(
        run_id=paused.run_id,
        profile=paused.profile,
        suggested=paused.suggested,
        presets=PRESETS,
    )


def result_response(result: FinishedRun) -> ResultResponse:
    return ResultResponse(
        run_id=result.run_id,
        outcome=result.outcome,
        rubric_name=result.rubric_name,
        matches=result.matches,
        jobs_found=len(result.jobs),
        searches=result.searches,
        removed_as_risky=result.removed_as_risky,
        dropped=result.dropped,
        warnings=result.warnings,
        trust=result.trust,
        llm_tokens=result.llm_tokens,
    )
