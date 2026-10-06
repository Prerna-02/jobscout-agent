"""The only code that talks to JSearch over HTTP."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx


class JSearchError(RuntimeError):
    """A search that failed. `message` is safe to show the user."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class RequestInfo:
    """How a search request went, for tracing: attempts include retries."""

    attempts: int
    status: int
    seconds: float


class JSearchClient:
    """Calls the search endpoint and returns the list of raw jobs.

    Retries only failures that are likely to pass on a second try (network
    errors, 5xx). A 429 means the plan's quota or rate limit is used up, and
    retrying would only waste time.
    """

    _RETRY_DELAYS = (1.0, 3.0)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )
        self._sleep = sleep

    def search(self, params: dict[str, str]) -> tuple[list[dict[str, Any]], RequestInfo]:
        started = time.monotonic()
        last_error: JSearchError | None = None
        for attempt in range(len(self._RETRY_DELAYS) + 1):
            if attempt:
                self._sleep(self._RETRY_DELAYS[attempt - 1])
            try:
                response = self._http.get("/search-v2", params=params)
            except httpx.TransportError as exc:  # timeouts, DNS, connection resets
                last_error = JSearchError("Could not reach the job search service.")
                last_error.__cause__ = exc
                continue
            if response.status_code >= 500:
                last_error = JSearchError(
                    "The job search service is having problems.", status=response.status_code
                )
                continue
            info = RequestInfo(attempt + 1, response.status_code, time.monotonic() - started)
            return _jobs_from(response), info
        assert last_error is not None
        raise last_error


def _jobs_from(response: httpx.Response) -> list[dict[str, Any]]:
    status = response.status_code
    if status in (401, 403):
        raise JSearchError("JSearch rejected the API key. Check JSEARCH_API_KEY.", status=status)
    if status == 429:
        raise JSearchError(
            "The JSearch request limit is used up. Try again later or upgrade the plan.",
            status=status,
        )
    if status >= 400:
        raise JSearchError(f"JSearch refused the search (HTTP {status}).", status=status)

    try:
        body = response.json()
        jobs = body["data"]["jobs"]
    except (ValueError, KeyError, TypeError) as exc:
        raise JSearchError("JSearch returned a response in an unexpected format.") from exc
    if body.get("status") != "OK" or not isinstance(jobs, list):
        raise JSearchError("JSearch returned an error instead of results.")
    return jobs
