"""The LangGraph state machine. Nodes are thin: each calls one deep module.

State holds plain JSON (Pydantic models dumped with mode="json"), so every
checkpoint can be saved to Postgres or SQLite and read back by any later version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from jobscout.cv import CVDocument
from jobscout.jobs import FilterRules, JobPosting, JobSearch, JSearchError, SearchQuery, filter_jobs
from jobscout.profile import CandidateProfile, ProfileBuilder
from jobscout.scoring import JobMatch, MatchScorer, Shortlister, build_match, rank_matches
from jobscout.trust import TrustChecker, TrustReport

from ._planner import LADDER, QueryPlanner
from ._preferences import SearchPreferences, resolve_rubric, suggest_preferences

Outcome = Literal["enough_jobs", "retries_used", "no_new_search", "search_failed"]


class AgentState(TypedDict, total=False):
    cv_text: str
    cv_fingerprint: str
    cv_format: str
    owner_hash: str  # hash of the run's secret key; the key itself is never stored
    profile: dict[str, Any]  # CandidateProfile
    preferences: dict[str, Any]  # SearchPreferences, set at the review step
    query: dict[str, Any]  # the SearchQuery to run next
    past_queries: list[str]  # cache keys already searched, so a rewrite can't repeat one
    rewrites: int
    searches: list[dict[str, Any]]  # one entry per search: what was asked and what came back
    new_jobs: list[dict[str, Any]]  # the last search's postings, before filtering
    jobs: list[dict[str, Any]]  # kept postings from all searches
    dropped: dict[str, int]
    warnings: list[str]
    outcome: Outcome
    trust: dict[str, dict[str, Any]]  # job_id -> TrustReport
    removed_as_risky: list[dict[str, str]]  # {"title", "company", "reason"}
    shortlist: list[dict[str, Any]]  # {"job": JobPosting, "similarity": float | None}
    matches: list[dict[str, Any]]  # JobMatch for every shortlisted job
    ranked: list[dict[str, Any]]  # the top N JobMatch, best first
    rubric_name: str


@dataclass
class AgentDeps:
    profile_builder: ProfileBuilder
    planner: QueryPlanner
    job_search: Callable[[], JobSearch]  # created on first use: one search budget per run
    trust: Callable[[], TrustChecker] | None = None
    shortlister: Callable[[], Shortlister] | None = None
    scorer: Callable[[], MatchScorer] | None = None
    top_n: int = 10
    min_jobs: int = 10
    max_rewrites: int = 3
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    search_budget: list[JobSearch] = []  # one JobSearch per compiled graph, i.e. per request

    def job_search() -> JobSearch:
        if not search_budget:
            search_budget.append(deps.job_search())
        return search_budget[0]

    def extract_profile(state: AgentState, config: RunnableConfig) -> AgentState:
        cv = CVDocument(
            text=state["cv_text"],
            fingerprint=state["cv_fingerprint"],
            source_format=state["cv_format"],  # type: ignore[arg-type]
            page_count=None,
        )
        profile = deps.profile_builder.build(cv, config=config)
        return {"profile": profile.model_dump(mode="json"), "warnings": list(profile.warnings)}

    def confirm_profile(state: AgentState) -> AgentState:
        # Pauses the run here. Everything before this line must be free of side
        # effects, because LangGraph re-runs the node from the top on resume.
        profile = CandidateProfile.model_validate(state["profile"])
        answer = interrupt({"profile": state["profile"], "suggested": suggest_preferences(profile)})
        prefs = SearchPreferences.model_validate(answer)
        # Profile warnings were for the review step; the user has now answered them.
        return {"preferences": prefs.model_dump(mode="json"), "warnings": []}

    def plan_query(state: AgentState, config: RunnableConfig) -> AgentState:
        prefs = SearchPreferences.model_validate(state["preferences"])
        query = deps.planner.first(prefs, config=config)
        return {"query": query.model_dump(mode="json"), "rewrites": 0, "past_queries": []}

    def search_jobs(state: AgentState, config: RunnableConfig) -> AgentState:
        query = SearchQuery.model_validate(state["query"])
        rung = LADDER[state["rewrites"] - 1] if state.get("rewrites") else "first"
        record = {"text": query.text, "date_posted": query.date_posted, "rung": rung}
        try:
            result = job_search().search(query, config=config)
        except JSearchError as exc:
            return {
                "outcome": "search_failed",
                "new_jobs": [],
                "warnings": [*state.get("warnings", []), exc.message],
                "searches": [*state.get("searches", []), {**record, "error": exc.message}],
            }
        return {
            "new_jobs": [job.model_dump(mode="json") for job in result.jobs],
            "past_queries": [*state.get("past_queries", []), query.cache_key()],
            "searches": [
                *state.get("searches", []),
                {**record, "found": len(result.jobs), "cache_hit": result.cache_hit},
            ],
        }

    def filter_dedupe(state: AgentState) -> AgentState:
        prefs = SearchPreferences.model_validate(state["preferences"])
        kept_before = [JobPosting.model_validate(j) for j in state.get("jobs", [])]
        new = [JobPosting.model_validate(j) for j in state.get("new_jobs", [])]
        rules = FilterRules(countries=frozenset({prefs.country}), remote_only=prefs.remote_only)
        # Earlier keepers go first, so a new posting is checked against them for duplicates.
        result = filter_jobs([*kept_before, *new], rules, now=deps.clock())
        dropped = dict(state.get("dropped", {}))
        for reason, count in result.dropped.items():
            dropped[reason] = dropped.get(reason, 0) + count
        searches = [*state["searches"]]
        searches[-1] = {**searches[-1], "kept": len(result.kept) - len(kept_before)}
        return {
            "jobs": [job.model_dump(mode="json") for job in result.kept],
            "dropped": dropped,
            "searches": searches,
        }

    def rewrite_query(state: AgentState, config: RunnableConfig) -> AgentState:
        prefs = SearchPreferences.model_validate(state["preferences"])
        previous = SearchQuery.model_validate(state["query"])
        rung = LADDER[state["rewrites"]]
        query = deps.planner.broaden(previous, prefs, rung, config=config)
        if query.cache_key() in state.get("past_queries", []):
            return {"outcome": "no_new_search"}
        return {"query": query.model_dump(mode="json"), "rewrites": state["rewrites"] + 1}

    def finish(state: AgentState) -> AgentState:
        warnings = list(state.get("warnings", []))
        found = len(state.get("jobs", []))
        outcome: Outcome = state.get("outcome") or (
            "enough_jobs" if found >= deps.min_jobs else "retries_used"
        )
        if outcome != "enough_jobs" and outcome != "search_failed":
            warnings.append(
                f"Only {found} matching jobs found after {state.get('rewrites', 0)} broader "
                "searches. Try another job title, a wider location, or an older date range."
            )
        return {"outcome": outcome, "warnings": warnings}

    def shortlist(state: AgentState) -> AgentState:
        profile = CandidateProfile.model_validate(state["profile"])
        prefs = SearchPreferences.model_validate(state["preferences"])
        jobs = [JobPosting.model_validate(j) for j in state["jobs"]]
        text = "\n".join(
            [", ".join(prefs.target_titles), ", ".join(prefs.skills[:30]), profile.summary]
        )
        assert deps.shortlister is not None
        pairs = deps.shortlister().shortlist(text, jobs)
        return {
            "shortlist": [
                {"job": job.model_dump(mode="json"), "similarity": sim} for job, sim in pairs
            ]
        }

    def score_jobs(state: AgentState, config: RunnableConfig) -> AgentState:
        profile = CandidateProfile.model_validate(state["profile"])
        prefs = SearchPreferences.model_validate(state["preferences"])
        rubric = resolve_rubric(prefs, profile)
        pairs = [(JobPosting.model_validate(p["job"]), p["similarity"]) for p in state["shortlist"]]
        warnings = list(state.get("warnings", []))
        assert deps.scorer is not None
        try:
            matches = deps.scorer().score(
                profile, prefs.model_dump(mode="json"), rubric, pairs, config=config
            )
        except Exception as exc:  # every model failed: still show the jobs, unscored
            warnings.append(f"Scoring failed, jobs are shown unranked ({type(exc).__name__}).")
            matches = [build_match(job, sim, None, rubric, 0) for job, sim in pairs]
        return {
            "matches": [m.model_dump(mode="json") for m in matches],
            "rubric_name": rubric.name,
            "warnings": warnings,
        }

    def rank_jobs(state: AgentState) -> AgentState:
        matches = [JobMatch.model_validate(m) for m in state["matches"]]
        return {"ranked": [m.model_dump(mode="json") for m in rank_matches(matches, deps.top_n)]}

    def trust_check(state: AgentState, config: RunnableConfig) -> AgentState:
        jobs = [JobPosting.model_validate(j) for j in state.get("jobs", [])]
        if deps.trust is None or not jobs:
            return {"trust": {}, "removed_as_risky": []}
        reports = deps.trust().check(jobs, config=config)
        # Strong scam signs remove a posting before any scoring tokens are spent on it.
        risky = [j for j in jobs if reports[j.job_id].level == "high_risk"]
        kept = [j for j in jobs if reports[j.job_id].level != "high_risk"]
        dropped = dict(state.get("dropped", {}))
        if risky:
            dropped["scam_risk"] = dropped.get("scam_risk", 0) + len(risky)
        return {
            "jobs": [j.model_dump(mode="json") for j in kept],
            "dropped": dropped,
            "trust": {k: r.model_dump(mode="json") for k, r in reports.items()},
            "removed_as_risky": [
                {
                    "title": j.title,
                    "company": j.company or "-",
                    "reason": _first_high_flag(reports[j.job_id]),
                }
                for j in risky
            ],
        }

    def after_finish(state: AgentState) -> str:
        return "trust_check" if state.get("jobs") else END

    def after_trust(state: AgentState) -> str:
        scoring_enabled = deps.shortlister is not None and deps.scorer is not None
        return "shortlist" if scoring_enabled and state.get("jobs") else END

    def after_filter(state: AgentState) -> str:
        if state.get("outcome") == "search_failed":
            return "finish"
        if len(state.get("jobs", [])) >= deps.min_jobs:
            return "finish"
        if state.get("rewrites", 0) >= min(deps.max_rewrites, len(LADDER)):
            return "finish"
        return "rewrite_query"

    def after_rewrite(state: AgentState) -> str:
        return "finish" if state.get("outcome") == "no_new_search" else "search_jobs"

    graph = StateGraph(AgentState)
    graph.add_node("extract_profile", extract_profile)
    graph.add_node("confirm_profile", confirm_profile)
    graph.add_node("plan_query", plan_query)
    graph.add_node("search_jobs", search_jobs)
    graph.add_node("filter_dedupe", filter_dedupe)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("finish", finish)
    graph.add_node("trust_check", trust_check)
    graph.add_node("shortlist", shortlist)
    graph.add_node("score_jobs", score_jobs)
    graph.add_node("rank_jobs", rank_jobs)

    graph.add_edge(START, "extract_profile")
    graph.add_edge("extract_profile", "confirm_profile")
    graph.add_edge("confirm_profile", "plan_query")
    graph.add_edge("plan_query", "search_jobs")
    graph.add_edge("search_jobs", "filter_dedupe")
    graph.add_conditional_edges("filter_dedupe", after_filter, ["rewrite_query", "finish"])
    graph.add_conditional_edges("rewrite_query", after_rewrite, ["search_jobs", "finish"])
    graph.add_conditional_edges("finish", after_finish, ["trust_check", END])
    graph.add_conditional_edges("trust_check", after_trust, ["shortlist", END])
    graph.add_edge("shortlist", "score_jobs")
    graph.add_edge("score_jobs", "rank_jobs")
    graph.add_edge("rank_jobs", END)
    return graph.compile(checkpointer=checkpointer)


def _first_high_flag(report: TrustReport) -> str:
    return next((f.text for f in report.flags if f.severity == "high"), "High scam risk")
