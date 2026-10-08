"""Who may do what, how often, and how many searches run at once.

Three separate questions, answered separately:

1. Whose run is this?  Each run gets a secret key when the CV is uploaded. Only someone
   who sends that key (X-Run-Token) can see or continue the run, from any network.
2. How much has this person used today?  Counted per browser (X-Client-Id, a random id
   the website keeps), plus a looser count per network so a new id doesn't reset limits.
3. Is JobScout busy?  At most N searches run at the same moment across all users; the
   rest get "busy, try again in a minute" instead of every search failing together.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import HTTPException, Request

from jobscout.settings import Settings
from jobscout.storage import KeyValueStore

Action = Literal["start", "resume"]
_DAY = 24 * 60 * 60
_SLOT_SECONDS = 10 * 60  # longer than any search; frees the slot if a server crashes mid-run
_CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def new_run_key() -> tuple[str, str]:
    """A fresh secret key for a run, and the hash we store. The key itself is never saved."""
    key = secrets.token_urlsafe(32)  # 256 random bits: impossible to guess
    return key, _hash(key)


class Guard:
    def __init__(self, settings: Settings, store: KeyValueStore) -> None:
        raw = settings.api_invite_codes.get_secret_value() if settings.api_invite_codes else ""
        self._codes = {code.strip() for code in raw.split(",") if code.strip()}
        self._limits = {
            "start": settings.api_starts_per_day,
            "resume": settings.api_resumes_per_day,
        }
        self._network_factor = settings.api_ip_limit_multiplier
        self._global_resumes = settings.api_global_resumes_per_day
        self._max_searches = settings.api_max_concurrent_searches
        self._store = store

    # --- who is calling -------------------------------------------------------------

    def check_invite(self, request: Request) -> None:
        if self._codes and request.headers.get("x-invite-code", "").strip() not in self._codes:
            raise HTTPException(401, "A valid invite code is required (X-Invite-Code header).")

    @staticmethod
    def client_id(request: Request) -> str | None:
        """The browser's anonymous id, if it sent a well-formed one."""
        value = request.headers.get("x-client-id", "").strip()
        return "client-" + _hash(value)[:12] if _CLIENT_ID.fullmatch(value) else None

    @staticmethod
    def network_id(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        address = forwarded or (request.client.host if request.client else "unknown")
        return "net-" + _hash(address)[:12]

    def user_id(self, request: Request) -> str:
        """For Langfuse: the browser id when there is one, else the network."""
        return self.client_id(request) or self.network_id(request)

    # --- 1. whose run is this --------------------------------------------------------

    @staticmethod
    def require_run_key(request: Request, owner_hash: str | None, run_id: str) -> None:
        """404 unless the request carries this run's key. A wrong key looks like no run at all."""
        sent = request.headers.get("x-run-token", "")
        if not owner_hash or not sent or not hmac.compare_digest(_hash(sent), owner_hash):
            raise HTTPException(404, f"No run with id '{run_id}'.")

    # --- 2. how much has this person used today --------------------------------------

    def spend(self, request: Request, action: Action) -> None:
        """Count one use for the browser and for the network; 429 once a limit is passed."""
        limit = self._limits[action]
        what = "CV uploads" if action == "start" else "searches"
        client = self.client_id(request)
        if client and self._count(action, client) > limit:
            raise HTTPException(429, f"Daily limit reached: {limit} {what} per day.")
        # Shared networks get more room; without a browser id, the network gets the plain limit.
        network_limit = limit * self._network_factor if client else limit
        if self._count(action, self.network_id(request)) > network_limit:
            raise HTTPException(429, f"Too many {what} from this network today. Try tomorrow.")
        if action == "resume" and self._count("resume", "all") > self._global_resumes:
            raise HTTPException(429, "JobScout has reached today's search limit. Try tomorrow.")

    def _count(self, action: Action, who: str) -> int:
        day = datetime.now(UTC).date().isoformat()
        used = self._store.incr(f"jobscout:limit:{action}:{who}:{day}", ttl_seconds=2 * _DAY)
        return used or 0  # if Redis is down we can't count; let the request through

    # --- 3. is JobScout busy ---------------------------------------------------------

    @contextmanager
    def search_slot(self) -> Iterator[None]:
        """Hold one of N search slots for the length of a search, or raise 503 if all are taken.

        Slots live in Redis, so the limit holds across every server instance. Each slot
        expires by itself after 10 minutes, so a crashed server can't block a slot forever.
        """
        ticket = uuid.uuid4().hex
        slot = None
        for number in range(self._max_searches):
            key = f"jobscout:search-slot:{number}"
            taken = self._store.claim(key, ticket, ttl_seconds=_SLOT_SECONDS)
            if taken is None:  # Redis unreachable: don't block searches because of it
                break
            if taken:
                slot = key
                break
        else:
            raise HTTPException(
                503,
                "JobScout is busy right now. Please try again in a minute.",
                headers={"Retry-After": "60"},
            )
        try:
            yield
        finally:
            if slot is not None:
                self._store.release(slot, ticket)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
