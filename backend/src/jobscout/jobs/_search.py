"""JobSearch: cache first, then budget checks, then one JSearch call."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from jobscout.settings import Settings, get_settings

from ._client import JSearchClient, JSearchError
from ._models import JobPosting, SearchQuery, SearchResult
from ._normalize import normalize_job
from ._store import KeyValueStore, MemoryStore, RedisStore

logger = logging.getLogger(__name__)
_MONTH_KEY_TTL = 40 * 24 * 60 * 60  # outlives the month it counts


class JSearchNotConfigured(RuntimeError):
    """JSEARCH_API_KEY is missing."""


class JobSearch:
    """One instance per agent run: `calls_made` is the run's search budget.

    A search is answered from the cache when possible. Only a cache miss spends
    quota, and only after both limits pass: the per-run budget (stops a rewrite
    loop from burning the plan) and the monthly counter shared through Redis.
    """

    def __init__(
        self,
        client: JSearchClient,
        store: KeyValueStore,
        *,
        cache_ttl_seconds: int = 24 * 60 * 60,
        max_calls_per_run: int = 4,
        monthly_limit: int = 190,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._client = client
        self._store = store
        self._ttl = cache_ttl_seconds
        self._max_calls = max_calls_per_run
        self._monthly_limit = monthly_limit
        self._clock = clock
        self.calls_made = 0
        # A named runnable so each search shows up as a "search_jobs" step in Langfuse.
        self._runnable: Runnable[SearchQuery, SearchResult] = RunnableLambda(
            self._search, name="search_jobs"
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> JobSearch:
        s = settings or get_settings()
        api_key = s.jsearch_api_key.get_secret_value() if s.jsearch_api_key else ""
        if not api_key:
            raise JSearchNotConfigured("Set JSEARCH_API_KEY in .env to search for jobs.")
        redis_url = s.redis_url.get_secret_value() if s.redis_url else ""
        if redis_url:
            store: KeyValueStore = RedisStore.from_url(redis_url)
        else:
            logger.warning("REDIS_URL is not set: searches are cached in memory only.")
            store = MemoryStore()
        client = JSearchClient(
            api_key, base_url=s.jsearch_base_url, timeout=s.jsearch_timeout_seconds
        )
        return cls(
            client,
            store,
            cache_ttl_seconds=s.search_cache_ttl_seconds,
            max_calls_per_run=s.jsearch_max_calls_per_run,
            monthly_limit=s.jsearch_monthly_limit,
        )

    def search(self, query: SearchQuery, *, config: RunnableConfig | None = None) -> SearchResult:
        return self._runnable.invoke(query, config=config)

    def _search(self, query: SearchQuery) -> SearchResult:
        key = query.cache_key()
        if (cached := self._store.get_json(key)) is not None:
            return SearchResult(
                query=query,
                jobs=[JobPosting.model_validate(job) for job in cached["jobs"]],
                cache_hit=True,
                fetched_at=datetime.fromisoformat(cached["fetched_at"]),
            )

        now = self._clock()
        month_used = self._spend_quota(now)
        raw_jobs = self._request(query.to_params(), month_used)
        jobs = [job for raw in raw_jobs if (job := normalize_job(raw, now=now))]

        # Empty results are cached too: repeating a search that found nothing wastes quota.
        self._store.set_json(
            key,
            {"fetched_at": now.isoformat(), "jobs": [j.model_dump(mode="json") for j in jobs]},
            ttl_seconds=self._ttl,
        )
        return SearchResult(query=query, jobs=jobs, cache_hit=False, fetched_at=now)

    def _request(self, params: dict[str, str], month_used: int | None) -> list[dict[str, Any]]:
        """Call JSearch inside its own "jsearch_request" step, so Langfuse shows the exact
        parameters sent, the attempts, the HTTP status and the quota used.

        The step's output is a small summary; the raw jobs (often 30+ KB) are handed
        back through `received` instead, to keep traces light.
        """
        received: list[dict[str, Any]] = []

        def call(sent: dict[str, str]) -> dict[str, Any]:
            jobs, info = self._client.search(sent)
            received.extend(jobs)
            return {
                "jobs_returned": len(jobs),
                "attempts": info.attempts,
                "http_status": info.status,
                "seconds": round(info.seconds, 2),
                "requests_used_this_month": month_used,
                "monthly_limit": self._monthly_limit,
            }

        RunnableLambda(call, name="jsearch_request").invoke(params)
        return received

    def _spend_quota(self, now: datetime) -> int | None:
        if self.calls_made >= self._max_calls:
            raise JSearchError(f"This run has already used its {self._max_calls} job searches.")
        used = self._store.incr(f"jobscout:jsearch:calls:{now:%Y-%m}", ttl_seconds=_MONTH_KEY_TTL)
        if used is not None and used > self._monthly_limit:
            raise JSearchError(
                f"This month's job search limit ({self._monthly_limit}) is reached. "
                "Cached searches still work."
            )
        self.calls_made += 1
        return used
