"""Langfuse tracing behind two functions.

    config = trace_config(run_name="build_profile", session_id=cv.fingerprint)
    runnable.invoke(inputs, config=config)   # traced if Langfuse keys are set
    flush()                                   # before a short-lived process exits

Without Langfuse keys, `trace_config` returns a plain config and nothing is sent,
so tests and offline development need no setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from jobscout.cv import mask_patterns
from jobscout.settings import get_settings

__all__ = ["flush", "trace_config"]


def trace_config(
    *,
    run_name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RunnableConfig:
    """Build a LangChain/LangGraph run config that reports to Langfuse.

    `session_id` groups traces in Langfuse (we use the CV fingerprint, so every
    run for one CV lands in one session).
    """
    settings = get_settings()
    all_tags = [f"env:{settings.app_env}", *tags]
    meta: dict[str, Any] = {**(metadata or {}), "langfuse_tags": all_tags}
    if session_id:
        meta["langfuse_session_id"] = session_id
    if user_id:
        meta["langfuse_user_id"] = user_id

    config: RunnableConfig = {"run_name": run_name, "tags": all_tags, "metadata": meta}
    if _client() is not None:
        config["callbacks"] = [CallbackHandler()]
    return config


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()


@lru_cache(maxsize=1)
def _client() -> Langfuse | None:
    settings = get_settings()
    if not settings.tracing_enabled:
        return None
    assert settings.langfuse_secret_key is not None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
        environment=settings.app_env,
        mask=_mask,
    )


def _mask(*, data: Any, **_: Any) -> Any:
    # Second layer after CV redaction: catch any email, link or phone number
    # that reaches a trace some other way (a job ad, an error message).
    return mask_patterns(data)
