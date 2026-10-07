"""Shortlist: keep the jobs whose meaning is closest to the candidate, before careful scoring.

Embeddings turn text into vectors; cosine similarity measures how close two
vectors are. It's cheap and needs no reasoning, so the expensive scorer only sees
the most promising jobs. Job vectors are cached, because the same posting comes
back across runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from jobscout.jobs import JobPosting
from jobscout.storage import KeyValueStore, cache_key

logger = logging.getLogger(__name__)
_EMBED_CHARS = 2000  # the start of an ad (title, summary, key requirements) carries its meaning
_VECTOR_TTL = 30 * 24 * 60 * 60


class Shortlister:
    def __init__(
        self,
        embeddings: Embeddings | None,
        store: KeyValueStore,
        *,
        model_name: str,
        size: int = 15,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._model = model_name
        self._size = size

    def shortlist(
        self, profile_text: str, jobs: Sequence[JobPosting]
    ) -> list[tuple[JobPosting, float | None]]:
        """Top `size` jobs by similarity, best first. Similarity is None when not computed."""
        if len(jobs) <= self._size:
            return [(job, None) for job in jobs]  # nothing to cut: save the embedding calls
        try:
            query = self._embedder().embed_query(profile_text)
            vectors = self._job_vectors(jobs)
        except Exception as exc:  # embeddings are an optimization; never fail the run on them
            logger.warning("Embeddings failed, keeping the first %d jobs: %s", self._size, exc)
            return [(job, None) for job in jobs[: self._size]]
        ranked = sorted(
            ((job, _cosine(query, vectors[job.job_id])) for job in jobs),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(job, round(score, 4)) for job, score in ranked[: self._size]]

    def _job_vectors(self, jobs: Sequence[JobPosting]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        missing: list[JobPosting] = []
        for job in jobs:
            cached = self._store.get_json(cache_key("embedding", self._model, job.job_id))
            if cached is not None:
                vectors[job.job_id] = cached
            else:
                missing.append(job)
        if missing:  # one batched call for every job not seen before
            fresh = self._embedder().embed_documents([job_text(job) for job in missing])
            for job, vector in zip(missing, fresh, strict=True):
                vectors[job.job_id] = vector
                key = cache_key("embedding", self._model, job.job_id)
                self._store.set_json(key, vector, ttl_seconds=_VECTOR_TTL)
        return vectors

    def _embedder(self) -> Embeddings:
        if self._embeddings is None:
            raise RuntimeError("no embedding model configured")
        return self._embeddings


def job_text(job: JobPosting) -> str:
    return f"{job.title}\n{job.company or ''}\n{job.description[:_EMBED_CHARS]}"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
