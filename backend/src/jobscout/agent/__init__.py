"""The JobScout agent: CV in, pause for review, real jobs out.

    with JobScoutAgent.open() as agent:
        paused = agent.start(cv_bytes)                # profile built, run saved, waiting for you
        result = agent.resume(paused.run_id, prefs)   # any later process can resume by run id

Hidden inside: the LangGraph graph, where paused runs are saved (Postgres or a
local SQLite file), thread ids, recursion limits and Langfuse tracing.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from jobscout.cv import read_cv
from jobscout.jobs import JobPosting, JobSearch
from jobscout.observability import (
    UsageCounter,
    flush,
    record_scores,
    trace_config,
    usage_summary,
)
from jobscout.profile import CandidateProfile, ProfileBuilder
from jobscout.scoring import JobMatch, MatchScorer, Rubric, Shortlister
from jobscout.settings import Settings, get_settings
from jobscout.storage import open_store
from jobscout.trust import TrustChecker, TrustReport

from ._graph import AgentDeps, Outcome, build_graph
from ._planner import QueryPlanner
from ._preferences import SearchPreferences, resolve_rubric
from ._progress import Progress, describe_step

__all__ = [
    "AgentDeps",
    "FinishedRun",
    "JobScoutAgent",
    "PausedRun",
    "Progress",
    "RunNotFound",
    "SearchPreferences",
]

_RECURSION_LIMIT = 30  # 7 nodes, at most 3 loops of rewrite -> search -> filter


class RunNotFound(LookupError):
    """No paused run with that id (wrong id, already finished, or saved elsewhere)."""


@dataclass
class PausedRun:
    run_id: str
    profile: CandidateProfile
    suggested: dict[str, Any]


@dataclass
class FinishedRun:
    run_id: str
    outcome: Outcome
    jobs: list[JobPosting]
    searches: list[dict[str, Any]]
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    llm_tokens: dict[str, dict[str, int]] = field(default_factory=dict)  # per model
    matches: list[JobMatch] = field(default_factory=list)  # ranked, best first
    trust: dict[str, TrustReport] = field(default_factory=dict)  # by job_id
    removed_as_risky: list[dict[str, str]] = field(default_factory=list)
    rubric_name: str | None = None


class JobScoutAgent:
    def __init__(self, checkpointer: BaseCheckpointSaver, deps: Callable[[], AgentDeps]) -> None:
        self._checkpointer = checkpointer
        self._deps = deps

    @classmethod
    @contextmanager
    def open(cls, settings: Settings | None = None) -> Iterator[JobScoutAgent]:
        """Connect to where paused runs are saved, for as long as the `with` block lasts."""
        s = settings or get_settings()
        with _checkpointer(s) as saver:
            yield cls(saver, lambda: _default_deps(s))

    def start(self, cv_bytes: bytes, *, user_id: str | None = None) -> PausedRun:
        """Read the CV and build the profile, then pause for review. Raises CVRejected."""
        cv = read_cv(cv_bytes)  # the raw file never enters the saved state
        run_id = uuid.uuid4().hex[:12]
        graph = build_graph(self._deps(), self._checkpointer)
        config, usage = self._config(run_id, cv.fingerprint, "agent_start", user_id)
        try:
            graph.invoke(
                {
                    "cv_text": cv.text,
                    "cv_fingerprint": cv.fingerprint,
                    "cv_format": cv.source_format,
                },
                config=config,
            )
            record_scores(config, {"llm_tokens": _total_tokens(usage_summary(usage))})
        finally:
            flush()
        paused = graph.get_state(config).interrupts[0].value
        return PausedRun(
            run_id=run_id,
            profile=CandidateProfile.model_validate(paused["profile"]),
            suggested=paused["suggested"],
        )

    def saved_profile(self, run_id: str) -> CandidateProfile:
        """The profile of a paused run, e.g. to prefill the review screen again."""
        snapshot = build_graph(self._deps(), self._checkpointer).get_state(
            {"configurable": {"thread_id": run_id}}
        )
        if not snapshot.interrupts:
            raise RunNotFound(f"No paused run with id '{run_id}'.")
        return CandidateProfile.model_validate(snapshot.values["profile"])

    def resume(
        self, run_id: str, preferences: SearchPreferences, *, user_id: str | None = None
    ) -> FinishedRun:
        """Continue a paused run with the user's answers and return the final result."""
        result: FinishedRun | None = None
        for event in self.resume_events(run_id, preferences, user_id=user_id):
            if isinstance(event, FinishedRun):
                result = event
        assert result is not None
        return result

    def resume_events(
        self, run_id: str, preferences: SearchPreferences, *, user_id: str | None = None
    ) -> Iterator[Progress | FinishedRun]:
        """Like resume(), but yields a Progress message after each step, then the result.

        Raises RunNotFound before yielding anything if there is no paused run with that id.
        """
        graph = build_graph(self._deps(), self._checkpointer)
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
        if not snapshot.interrupts:
            raise RunNotFound(f"No paused run with id '{run_id}'.")
        fingerprint = snapshot.values["cv_fingerprint"]
        return self._resume_stream(graph, run_id, fingerprint, preferences, user_id)

    def _resume_stream(
        self,
        graph: Any,
        run_id: str,
        fingerprint: str,
        preferences: SearchPreferences,
        user_id: str | None,
    ) -> Iterator[Progress | FinishedRun]:
        config, usage = self._config(run_id, fingerprint, "agent_resume", user_id)
        try:
            for update in graph.stream(
                Command(resume=preferences.model_dump(mode="json")),
                config=config,
                stream_mode="updates",
            ):
                for node, values in update.items():
                    message = (
                        describe_step(node, values or {}) if isinstance(values, dict) else None
                    )
                    if message:
                        yield Progress(node, message)
            state = graph.get_state(config).values
            result = _finished(run_id, state, usage_summary(usage))
            record_scores(config, _run_scores(state, result), comment=f"outcome: {result.outcome}")
            yield result
        finally:
            flush()

    def status(self, run_id: str) -> PausedRun | FinishedRun:
        """Where a run stands: waiting for review, or finished with its results."""
        snapshot = build_graph(self._deps(), self._checkpointer).get_state(
            {"configurable": {"thread_id": run_id}}
        )
        if snapshot.interrupts:
            paused = snapshot.interrupts[0].value
            return PausedRun(
                run_id=run_id,
                profile=CandidateProfile.model_validate(paused["profile"]),
                suggested=paused["suggested"],
            )
        if snapshot.values.get("outcome"):
            return _finished(run_id, snapshot.values, {})
        raise RunNotFound(f"No run with id '{run_id}'.")

    def scoring_inputs(
        self, run_id: str
    ) -> tuple[CandidateProfile, dict[str, Any], Rubric, list[tuple[JobPosting, float | None]]]:
        """What a finished run scored, so other models can score exactly the same jobs."""
        values = (
            build_graph(self._deps(), self._checkpointer)
            .get_state({"configurable": {"thread_id": run_id}})
            .values
        )
        if not values.get("shortlist"):
            raise RunNotFound(f"Run '{run_id}' has no scored shortlist (unknown or not finished).")
        profile = CandidateProfile.model_validate(values["profile"])
        prefs = SearchPreferences.model_validate(values["preferences"])
        pairs = [
            (JobPosting.model_validate(p["job"]), p["similarity"]) for p in values["shortlist"]
        ]
        return profile, prefs.model_dump(mode="json"), resolve_rubric(prefs, profile), pairs

    @staticmethod
    def _config(
        run_id: str, fingerprint: str, name: str, user_id: str | None = None
    ) -> tuple[RunnableConfig, UsageCounter]:
        # Every request of one run shares a Langfuse session, the CV fingerprint.
        config = trace_config(
            run_name=name,
            session_id=fingerprint,
            user_id=user_id,
            tags=["agent"],
            metadata={"run_id": run_id},
        )
        config["configurable"] = {"thread_id": run_id}
        config["recursion_limit"] = _RECURSION_LIMIT
        usage = UsageCounter()  # adds up tokens per model across every LLM call in the run
        config["callbacks"] = [*config.get("callbacks", []), usage]
        return config, usage


def _finished(run_id: str, state: dict[str, Any], tokens: dict[str, dict[str, int]]) -> FinishedRun:
    return FinishedRun(
        run_id=run_id,
        outcome=state["outcome"],
        jobs=[JobPosting.model_validate(job) for job in state.get("jobs", [])],
        searches=state.get("searches", []),
        dropped=state.get("dropped", {}),
        warnings=state.get("warnings", []),
        llm_tokens=tokens,
        matches=[JobMatch.model_validate(m) for m in state.get("ranked", [])],
        rubric_name=state.get("rubric_name"),
        trust={k: TrustReport.model_validate(v) for k, v in state.get("trust", {}).items()},
        removed_as_risky=state.get("removed_as_risky", []),
    )


def _run_scores(state: dict[str, Any], result: FinishedRun) -> dict[str, float]:
    searches = result.searches
    scores: dict[str, float] = {
        "jobs_found": len(result.jobs),
        "searches": len(searches),
        "jsearch_requests": sum(1 for x in searches if x.get("cache_hit") is False),
        "cache_hits": sum(1 for x in searches if x.get("cache_hit")),
        "rewrites": state.get("rewrites", 0),
        "llm_tokens": _total_tokens(result.llm_tokens),
        "scam_risk_removed": len(result.removed_as_risky),
        "caution_jobs": sum(1 for t in result.trust.values() if t.level == "caution"),
    }
    if result.matches and result.matches[0].total is not None:
        scores["top_match"] = result.matches[0].total
    return scores


def _total_tokens(tokens: dict[str, dict[str, int]]) -> int:
    return sum(model["total"] for model in tokens.values())


@contextmanager
def _checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver]:
    database_url = settings.database_url.get_secret_value() if settings.database_url else ""
    if database_url.strip():
        with PostgresSaver.from_conn_string(database_url.strip().strip('"')) as saver:
            saver.setup()  # creates the checkpoint tables on first use; safe to repeat
            yield saver
        return
    path = Path(settings.checkpoint_sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def _default_deps(settings: Settings) -> AgentDeps:
    store = open_store(settings)
    return AgentDeps(
        profile_builder=ProfileBuilder(
            store=store, cache_ttl_seconds=settings.profile_cache_ttl_seconds
        ),
        planner=QueryPlanner(),
        job_search=lambda: JobSearch.from_settings(settings),
        trust=lambda: TrustChecker(
            store=store,
            batch_size=settings.scorer_batch_size,
            cache_ttl_seconds=settings.trust_cache_ttl_seconds,
        ),
        shortlister=lambda: Shortlister(
            _embeddings(settings),
            store,
            model_name=settings.embedding_model,
            size=settings.shortlist_size,
        ),
        scorer=lambda: MatchScorer(
            store=store,
            batch_size=settings.scorer_batch_size,
            job_text_max_chars=settings.job_text_max_chars,
            cache_ttl_seconds=settings.score_cache_ttl_seconds,
            cache_namespace=settings.scorer_model,
        ),
        top_n=settings.top_matches,
        min_jobs=settings.agent_min_jobs,
        max_rewrites=settings.agent_max_rewrites,
    )


def _embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings | None:
    key = settings.google_api_key.get_secret_value() if settings.google_api_key else ""
    if not key:
        return None
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=key)
