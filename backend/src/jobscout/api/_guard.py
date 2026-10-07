"""Who is calling, how much they may use, and which runs are theirs.

Once JobScout is online, anyone could call it and spend the JSearch, Gemini and
Claude budgets. The guard requires an invite code (when configured), counts
uploads and searches per user per day, caps all users together, and makes sure a
user can only see or continue their own runs.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException, Request

from jobscout.settings import Settings
from jobscout.storage import KeyValueStore

Action = Literal["start", "resume"]
_DAY = 24 * 60 * 60
_RUN_OWNER_TTL = 30 * _DAY


class Guard:
    def __init__(self, settings: Settings, store: KeyValueStore) -> None:
        raw = settings.api_invite_codes.get_secret_value() if settings.api_invite_codes else ""
        self._codes = {code.strip() for code in raw.split(",") if code.strip()}
        self._limits = {
            "start": settings.api_starts_per_day,
            "resume": settings.api_resumes_per_day,
        }
        self._global_resumes = settings.api_global_resumes_per_day
        self._store = store

    def identity(self, request: Request) -> str:
        """A stable, anonymous id: from the invite code, or else from the caller's IP address."""
        if self._codes:
            code = request.headers.get("x-invite-code", "").strip()
            if code not in self._codes:
                raise HTTPException(401, "A valid invite code is required (X-Invite-Code header).")
            return "invite-" + _short_hash(code)
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        address = forwarded or (request.client.host if request.client else "unknown")
        return "ip-" + _short_hash(address)

    def spend(self, identity: str, action: Action) -> None:
        """Count one use; raise 429 once today's limit is passed."""
        day = datetime.now(UTC).date().isoformat()
        used = self._store.incr(f"jobscout:limit:{action}:{identity}:{day}", ttl_seconds=2 * _DAY)
        limit = self._limits[action]
        if used is not None and used > limit:
            what = "CV uploads" if action == "start" else "searches"
            raise HTTPException(
                429, f"Daily limit reached: {limit} {what} per day. Try again tomorrow."
            )
        if action == "resume":
            total = self._store.incr(f"jobscout:limit:resume:all:{day}", ttl_seconds=2 * _DAY)
            if total is not None and total > self._global_resumes:
                raise HTTPException(
                    429, "JobScout has reached today's search limit. Try again tomorrow."
                )

    def claim(self, run_id: str, identity: str) -> None:
        self._store.set_json(_owner_key(run_id), identity, ttl_seconds=_RUN_OWNER_TTL)

    def require_owner(self, run_id: str, identity: str) -> None:
        # Someone else's run looks exactly like a run that doesn't exist: 404 either way.
        if self._store.get_json(_owner_key(run_id)) != identity:
            raise HTTPException(404, f"No run with id '{run_id}'.")


def _owner_key(run_id: str) -> str:
    return f"jobscout:run-owner:{run_id}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]
