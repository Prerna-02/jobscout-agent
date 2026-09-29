"""CV text -> a fact-checked candidate profile.

    from jobscout.profile import ProfileBuilder
    profile = ProfileBuilder().build(cv, config=trace_config(...))
    profile.flagged_skills     # listed by the LLM but not written in the CV

The LLM only extracts what is written. Code then checks every skill against the
CV text, parses role dates, computes years of experience (overlaps counted once)
and derives seniority.
"""

from ._builder import ProfileBuilder, assemble_profile
from ._extraction import PROMPT_VERSION, ProfileExtraction
from ._schema import CandidateProfile, Role, Seniority, Skill, WorkMode

__all__ = [
    "PROMPT_VERSION",
    "CandidateProfile",
    "ProfileBuilder",
    "ProfileExtraction",
    "Role",
    "Seniority",
    "Skill",
    "WorkMode",
    "assemble_profile",
]
