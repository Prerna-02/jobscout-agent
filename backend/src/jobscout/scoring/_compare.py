"""Score one shortlist with several Claude models: compare tokens, cost, time, agreement."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig

from jobscout.jobs import JobPosting
from jobscout.llm import LLMNotConfigured, structured_llm
from jobscout.observability import UsageCounter
from jobscout.profile import CandidateProfile
from jobscout.settings import Settings, get_settings

from ._pricing import claude_cost
from ._rubric import Rubric
from ._scorer import BatchAssessment, JobMatch, MatchScorer


@dataclass
class ScorerReport:
    model: str
    seconds: float = 0.0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    matches: list[JobMatch] = field(default_factory=list)
    error: str | None = None


def compare_scorers(
    profile: CandidateProfile,
    preferences: dict[str, Any],
    rubric: Rubric,
    jobs: Sequence[tuple[JobPosting, float | None]],
    models: Sequence[str],
    *,
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> list[ScorerReport]:
    """Each model scores the same jobs with no cache and no fallback, so the numbers are fair."""
    s = settings or get_settings()
    if not (s.anthropic_api_key and s.anthropic_api_key.get_secret_value()):
        raise LLMNotConfigured("Set ANTHROPIC_API_KEY in .env to compare Claude models.")

    reports: list[ScorerReport] = []
    for model in models:
        report = ScorerReport(model=model)
        counter = UsageCounter()
        run_config: RunnableConfig = {
            **(config or {}),
            "callbacks": [*(config or {}).get("callbacks", []), counter],
            "run_name": f"compare/{model}",
        }
        scorer = MatchScorer(
            structured_llm(
                BatchAssessment, task="scoring", model=model, fallbacks=False, settings=s
            ),
            store=None,
            batch_size=s.scorer_batch_size,
            job_text_max_chars=s.job_text_max_chars,
        )
        started = time.monotonic()
        try:
            report.matches = scorer.score(profile, preferences, rubric, jobs, config=run_config)
        except Exception as exc:  # one model failing shouldn't hide the other's numbers
            report.error = f"{type(exc).__name__}: {exc}"[:300]
        report.seconds = round(time.monotonic() - started, 1)

        total_cost = 0.0
        for model_name, usage in counter.usage_metadata.items():
            details = usage.get("input_token_details") or {}
            report.input_tokens += usage.get("input_tokens", 0)
            report.cache_read_tokens += details.get("cache_read", 0) or 0
            report.cache_write_tokens += details.get("cache_creation", 0) or 0
            report.output_tokens += usage.get("output_tokens", 0)
            total_cost += claude_cost(model_name, usage) or 0.0
        report.cost_usd = round(total_cost, 4) if counter.usage_metadata else None
        reports.append(report)
    return reports


def agreement(a: Sequence[JobMatch], b: Sequence[JobMatch], top: int = 5) -> dict[str, float]:
    """How similarly two scorers judged the same jobs."""
    totals_a = {m.job.job_id: m.total for m in a}
    totals_b = {m.job.job_id: m.total for m in b}
    both = [j for j in totals_a if totals_a[j] is not None and totals_b.get(j) is not None]
    best_a = {m.job.job_id for m in sorted(a, key=lambda m: -(m.total or -1))[:top]}
    best_b = {m.job.job_id for m in sorted(b, key=lambda m: -(m.total or -1))[:top]}
    mean_gap = (
        sum(abs(totals_a[j] - totals_b[j]) for j in both) / len(both)  # type: ignore[operator]
        if both
        else float("nan")
    )
    return {"same_top": len(best_a & best_b), "top": top, "mean_score_gap": round(mean_gap, 1)}
