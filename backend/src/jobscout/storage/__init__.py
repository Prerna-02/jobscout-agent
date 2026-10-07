"""Shared key-value storage: caches (searches, profiles, embeddings, scores) and counters.

    store = open_store()               # Redis when REDIS_URL is set, else in-memory
    store.get_json(key); store.set_json(key, value, ttl_seconds=...); store.incr(key, ...)


Redis in real use, an in-memory dict when REDIS_URL isn't set (and in tests).
A storage failure must never break a search, so Redis errors are logged and
treated as "not cached".
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

import redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

logger = logging.getLogger(__name__)

__all__ = ["KeyValueStore", "MemoryStore", "RedisStore", "cache_key", "open_store"]


class KeyValueStore(Protocol):
    def get_json(self, key: str) -> Any | None: ...
    def set_json(self, key: str, value: Any, *, ttl_seconds: int) -> None: ...
    def incr(self, key: str, *, ttl_seconds: int) -> int | None: ...


class RedisStore:
    def __init__(self, client: redis.Redis) -> None:
        self._redis = client

    @classmethod
    def from_url(cls, url: str) -> RedisStore:
        url = url.strip().strip('"').strip("'")
        if not url.startswith(("redis://", "rediss://")):
            raise ValueError(
                "REDIS_URL must look like rediss://default:<password>@<host>:6379 "
                "(Upstash: copy the 'Redis URL', not the redis-cli command)."
            )
        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=5,
            # A brief network blip (seen in practice as a failed DNS lookup) should not
            # turn a cached search into a paid one, so reconnect twice before giving up.
            retry=Retry(ExponentialBackoff(cap=1.0, base=0.2), retries=2),
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )
        return cls(client)

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._redis.get(key)
        except redis.RedisError as exc:
            logger.warning("Redis read failed, treating as a cache miss: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def set_json(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        try:
            self._redis.set(key, json.dumps(value), ex=ttl_seconds)
        except redis.RedisError as exc:
            logger.warning("Redis write failed, result not cached: %s", exc)

    def incr(self, key: str, *, ttl_seconds: int) -> int | None:
        try:
            pipe = self._redis.pipeline()
            pipe.set(key, 0, ex=ttl_seconds, nx=True)  # create with an expiry on first use
            pipe.incr(key)
            _, count = pipe.execute()
            return int(count)
        except redis.RedisError as exc:
            logger.warning("Redis counter failed, quota not tracked for this call: %s", exc)
            return None


class MemoryStore:
    """Same behaviour as RedisStore, kept in this process only."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}

    def get_json(self, key: str) -> Any | None:
        expires, value = self._data.get(key, (0.0, None))
        return value if expires > time.monotonic() else None

    def set_json(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        self._data[key] = (time.monotonic() + ttl_seconds, json.loads(json.dumps(value)))

    def incr(self, key: str, *, ttl_seconds: int) -> int | None:
        current = self.get_json(key)
        if current is None:
            self._data[key] = (time.monotonic() + ttl_seconds, 1)
            return 1
        expires = self._data[key][0]
        self._data[key] = (expires, current + 1)
        return current + 1


def open_store(settings: Any = None) -> KeyValueStore:
    """Redis when REDIS_URL is set, otherwise an in-memory store for this process."""
    from jobscout.settings import get_settings

    s = settings or get_settings()
    redis_url = s.redis_url.get_secret_value() if s.redis_url else ""
    if redis_url:
        return RedisStore.from_url(redis_url)
    logger.warning("REDIS_URL is not set: caches live in memory for this process only.")
    return MemoryStore()


def cache_key(namespace: str, *parts: object) -> str:
    """A short, stable key from any JSON-able parts, e.g. cache_key("score", job_id, rubric)."""
    import hashlib

    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return f"jobscout:{namespace}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"
