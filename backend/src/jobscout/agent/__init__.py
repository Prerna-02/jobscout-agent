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
from jobscout.settings import Settings, get_settings

from ._graph import AgentDeps, Outcome, build_graph
from ._planner import QueryPlanner
from ._preferences import SearchPreferences

__all__ = [
    "AgentDeps",
    "FinishedRun",
    "JobScoutAgent",
    "PausedRun",
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

    def start(self, cv_bytes: bytes) -> PausedRun:
        """Read the CV and build the profile, then pause for review. Raises CVRejected."""
        cv = read_cv(cv_bytes)  # the raw file never enters the saved state
        run_id = uuid.uuid4().hex[:12]
        graph = build_graph(self._deps(), self._checkpointer)
        config, usage = self._config(run_id, cv.fingerprint, "agent_start")
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

    def resume(self, run_id: str, preferences: SearchPreferences) -> FinishedRun:
        """Continue a paused run with the user's answers: plan, search, broaden, filter."""
        graph = build_graph(self._deps(), self._checkpointer)
        probe: RunnableConfig = {"configurable": {"thread_id": run_id}}
        snapshot = graph.get_state(probe)
        if not snapshot.interrupts:
            raise RunNotFound(f"No paused run with id '{run_id}'.")

        config, usage = self._config(run_id, snapshot.values["cv_fingerprint"], "agent_resume")
        try:
            graph.invoke(Command(resume=preferences.model_dump(mode="json")), config=config)
            state = graph.get_state(config).values
            tokens = usage_summary(usage)
            searches = state.get("searches", [])
            record_scores(
                config,
                {
                    "jobs_found": len(state.get("jobs", [])),
                    "searches": len(searches),
                    "jsearch_requests": sum(1 for x in searches if x.get("cache_hit") is False),
                    "cache_hits": sum(1 for x in searches if x.get("cache_hit")),
                    "rewrites": state.get("rewrites", 0),
                    "llm_tokens": _total_tokens(tokens),
                },
                comment=f"outcome: {state['outcome']}",
            )
        finally:
            flush()
        return FinishedRun(
            run_id=run_id,
            outcome=state["outcome"],
            jobs=[JobPosting.model_validate(job) for job in state.get("jobs", [])],
            searches=state.get("searches", []),
            dropped=state.get("dropped", {}),
            warnings=state.get("warnings", []),
            llm_tokens=tokens,
        )

    @staticmethod
    def _config(run_id: str, fingerprint: str, name: str) -> tuple[RunnableConfig, UsageCounter]:
        # Every request of one run shares a Langfuse session, the CV fingerprint.
        config = trace_config(
            run_name=name, session_id=fingerprint, tags=["agent"], metadata={"run_id": run_id}
        )
        config["configurable"] = {"thread_id": run_id}
        config["recursion_limit"] = _RECURSION_LIMIT
        usage = UsageCounter()  # adds up tokens per model across every LLM call in the run
        config["callbacks"] = [*config.get("callbacks", []), usage]
        return config, usage


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
    return AgentDeps(
        profile_builder=ProfileBuilder(),
        planner=QueryPlanner(),
        job_search=lambda: JobSearch.from_settings(settings),
        min_jobs=settings.agent_min_jobs,
        max_rewrites=settings.agent_max_rewrites,
    )
