"""CV text -> CandidateProfile: one LLM call, then code checks and derives the rest."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from jobscout.cv import CVDocument
from jobscout.llm import structured_llm

from ._experience import derive_seniority, parse_cv_date, total_years
from ._extraction import PROMPT_VERSION, ProfileExtraction, build_messages
from ._factcheck import SkillMatcher
from ._schema import CandidateProfile, Role, Skill

_PLACEHOLDER = re.compile(r"\[(?:NAME|EMAIL|PHONE|LINK|ADDRESS|REMOVED)\]")
_MAX_ITEM_CHARS = 200
_MAX_SKILL_CHARS = 60
_MAX_SKILLS = 80


class ProfileBuilder:
    """Builds a reviewed-ready profile from a redacted CV.

    `extractor` is any runnable that takes chat messages and returns a
    `ProfileExtraction`. It defaults to the configured LLM chain; tests pass a fake.
    """

    def __init__(
        self,
        extractor: Runnable[Any, ProfileExtraction] | None = None,
        *,
        today: date | None = None,
    ) -> None:
        self._extractor = extractor
        self._today = today
        # Wrapping the whole build in one runnable gives Langfuse a parent span
        # ("build_profile") with the LLM call nested inside and the final,
        # fact-checked profile as its output.
        self._runnable: Runnable[CVDocument, CandidateProfile] = RunnableLambda(
            self._build, name="build_profile"
        )

    def build(self, cv: CVDocument, *, config: RunnableConfig | None = None) -> CandidateProfile:
        config = dict(config or {})
        config["metadata"] = {**config.get("metadata", {}), "prompt_version": PROMPT_VERSION}
        return self._runnable.invoke(cv, config=config)  # type: ignore[arg-type]

    def _build(self, cv: CVDocument, config: RunnableConfig) -> CandidateProfile:
        extractor = self._extractor or structured_llm(ProfileExtraction)
        raw = extractor.invoke(build_messages(cv.text), config=config)
        extraction = (
            raw if isinstance(raw, ProfileExtraction) else ProfileExtraction.model_validate(raw)
        )
        return assemble_profile(extraction, cv.text, today=self._today or date.today())


def assemble_profile(
    extraction: ProfileExtraction, cv_text: str, *, today: date
) -> CandidateProfile:
    """Pure function: clean the LLM output, fact-check it and compute the derived fields."""
    warnings: list[str] = []
    matcher = SkillMatcher(cv_text)

    skills = [
        Skill(name=name, in_cv=(evidence := matcher.find(name)) is not None, evidence=evidence)
        for name in _clean_list(extraction.skills, max_chars=_MAX_SKILL_CHARS)[:_MAX_SKILLS]
    ]
    flagged = [s.name for s in skills if not s.in_cv]
    if flagged:
        warnings.append(
            f"{len(flagged)} skill(s) are not written in your CV: {', '.join(flagged)}. "
            "Keep them only if you really have them."
        )

    roles = [_to_role(r, today) for r in extraction.roles if _clean(r.title)]
    unreadable = [r.title for r in roles if r.kind == "job" and not (r.start and r.end)]
    if unreadable:
        warnings.append(
            f"Dates could not be read for: {', '.join(unreadable)}. "
            "Years of experience may be too low."
        )

    current_title = _clean(extraction.current_title) or next(
        (r.title for r in roles if r.is_current), None
    )
    if current_title and matcher.find(current_title) is None:
        warnings.append(f'The title "{current_title}" is not written in your CV word for word.')

    years = total_years(roles)
    return CandidateProfile(
        current_title=current_title,
        summary=_clean(extraction.summary) or "",
        seniority=derive_seniority(current_title, roles, years),
        years_experience=years,
        skills=skills,
        roles=roles,
        education=_clean_list(extraction.education),
        certifications=_clean_list(extraction.certifications),
        projects=_clean_list(extraction.projects),
        extracurriculars=_clean_list(extraction.extracurriculars),
        domains=_clean_list(extraction.domains),
        locations=_clean_list(extraction.locations),
        work_mode=extraction.work_mode,
        target_titles=[current_title] if current_title else [],
        warnings=warnings,
    )


def _to_role(raw: Any, today: date) -> Role:
    start = parse_cv_date(raw.start, today=today, is_end=False)
    end = parse_cv_date(raw.end, today=today, is_end=True)
    return Role(
        title=_clean(raw.title) or "",
        employer=_clean(raw.employer),
        kind=raw.kind,
        start_text=raw.start,
        end_text=raw.end,
        start=start,
        end=end,
        is_current=end == today,
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", _PLACEHOLDER.sub("", value)).strip(" ,;|-")
    return value or None


def _clean_list(values: list[str], *, max_chars: int = _MAX_ITEM_CHARS) -> list[str]:
    """Strip placeholders, drop empties and over-long items, remove case-insensitive duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = _clean(value)
        if not item or len(item) > max_chars or item.lower() in seen:
            continue
        seen.add(item.lower())
        result.append(item)
    return result
