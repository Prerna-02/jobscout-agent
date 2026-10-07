"""Scoring: shortlist by meaning, score with evidence, rank by your weights.

    shortlist = Shortlister(embeddings, store, model_name=...).shortlist(profile_text, jobs)
    matches = MatchScorer(store=store).score(profile, prefs, PRESETS["student"], shortlist)
    top10 = rank_matches(matches, 10)

Hidden inside: embedding calls and their cache, the scoring prompt, batching
(5 jobs per call), Claude prompt caching, quote verification, the weighted maths
with "not applicable" criteria, and per-job score caching.
"""

from ._compare import ScorerReport, agreement, compare_scorers
from ._pricing import PRICES_PER_MTOK, claude_cost
from ._rubric import PRESETS, Criterion, Rubric, build_rubric, suggest_preset
from ._scorer import (
    PROMPT_VERSION,
    BatchAssessment,
    CriterionResult,
    JobMatch,
    MatchScorer,
    build_match,
    rank_matches,
)
from ._shortlist import Shortlister

__all__ = [
    "PRESETS",
    "PRICES_PER_MTOK",
    "PROMPT_VERSION",
    "BatchAssessment",
    "Criterion",
    "CriterionResult",
    "JobMatch",
    "MatchScorer",
    "Rubric",
    "ScorerReport",
    "Shortlister",
    "agreement",
    "build_match",
    "build_rubric",
    "claude_cost",
    "compare_scorers",
    "rank_matches",
    "suggest_preset",
]
