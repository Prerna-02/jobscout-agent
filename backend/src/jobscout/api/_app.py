"""HTTP routes. Thin on purpose: each route checks the caller, then calls the agent."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from jobscout import __version__
from jobscout.agent import FinishedRun, JobScoutAgent, PausedRun, Progress, RunNotFound
from jobscout.cv import CVRejected
from jobscout.jobs import JSearchError, JSearchNotConfigured
from jobscout.llm import LLMNotConfigured
from jobscout.scoring import PRESETS, Rubric
from jobscout.settings import Settings, get_settings
from jobscout.storage import KeyValueStore, open_store

from ._guard import Guard, new_run_key
from ._schemas import ResultResponse, ResumeRequest, StartResponse, result_response, start_response

logger = logging.getLogger(__name__)

AgentOpener = Callable[[], AbstractContextManager[JobScoutAgent]]

_DESCRIPTION = """
Upload a CV, review the profile, then let the agent search, check and score real jobs.

1. `POST /runs` with the CV file. You get the profile and a `run_id`; the run is paused.
   Keep the returned `run_token`: send it as `X-Run-Token` on every call about this run.
2. `POST /runs/{run_id}/resume` with your choices. Progress streams live as Server-Sent
   Events (`progress`, then `result`). Add `?stream=false` to get one JSON answer instead.
3. `GET /runs/{run_id}` shows a run again, paused or finished.
"""


def create_app(
    settings: Settings | None = None,
    *,
    open_agent: AgentOpener | None = None,
    store: KeyValueStore | None = None,
) -> FastAPI:
    s = settings or get_settings()
    guard = Guard(s, store or open_store(s))
    opener: AgentOpener = open_agent or (lambda: JobScoutAgent.open(s))

    app = FastAPI(title="JobScout API", version=__version__, description=_DESCRIPTION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in s.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Invite-Code", "X-Client-Id", "X-Run-Token"],
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/presets", tags=["scoring"])
    def presets() -> dict[str, Rubric]:
        return PRESETS

    @app.post("/runs", status_code=201, tags=["runs"])
    def start_run(request: Request, cv: Annotated[UploadFile, File()]) -> StartResponse:
        guard.check_invite(request)
        data = cv.file.read(s.cv_max_bytes + 1)  # never read more than the limit allows
        if len(data) > s.cv_max_bytes:
            raise HTTPException(413, f"The file is larger than {s.cv_max_bytes // 1_048_576} MB.")
        guard.spend(request, "start")
        run_key, owner_hash = new_run_key()
        try:
            with opener() as agent:
                paused = agent.start(data, user_id=guard.user_id(request), owner_hash=owner_hash)
        except CVRejected as exc:
            raise HTTPException(422, {"reason": exc.reason, "message": exc.message}) from exc
        except LLMNotConfigured as exc:
            raise HTTPException(503, str(exc)) from exc
        # The key is shown exactly once, here. Only its hash is stored, with the run.
        return start_response(paused, run_key)

    @app.get("/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, request: Request) -> StartResponse | ResultResponse:
        guard.check_invite(request)
        run = _owned_run(opener, guard, request, run_id)
        return start_response(run) if isinstance(run, PausedRun) else result_response(run)

    @app.post("/runs/{run_id}/resume", tags=["runs"])
    def resume_run(run_id: str, body: ResumeRequest, request: Request, stream: bool = True) -> Any:
        guard.check_invite(request)
        run = _owned_run(opener, guard, request, run_id)
        if isinstance(run, FinishedRun):
            raise HTTPException(409, "This run has finished. Upload the CV again to search again.")
        try:
            prefs = body.to_preferences(run.profile)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors()[0]["msg"]) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        # Take a search slot first (503 if JobScout is busy), so a busy moment doesn't
        # use up the person's daily searches. Then count the search.
        slot = guard.search_slot()
        slot.__enter__()
        try:
            guard.spend(request, "resume")
        except HTTPException:
            slot.__exit__(None, None, None)
            raise
        user = guard.user_id(request)

        if not stream:
            try:
                with opener() as agent:
                    return result_response(agent.resume(run_id, prefs, user_id=user))
            except (JSearchNotConfigured, LLMNotConfigured) as exc:
                raise HTTPException(503, str(exc)) from exc
            finally:
                slot.__exit__(None, None, None)

        def events() -> Iterator[str]:
            try:
                with opener() as agent:
                    for event in agent.resume_events(run_id, prefs, user_id=user):
                        if isinstance(event, Progress):
                            yield _sse("progress", {"step": event.step, "message": event.message})
                        else:
                            yield _sse("result", result_response(event).model_dump(mode="json"))
            except (JSearchNotConfigured, LLMNotConfigured) as exc:
                yield _sse("error", {"message": str(exc)})
            except JSearchError as exc:
                yield _sse("error", {"message": exc.message})
            except Exception:  # never leak internals into the stream
                logger.exception("Run %s failed", run_id)
                yield _sse("error", {"message": "Something went wrong. Please try again."})
            finally:
                slot.__exit__(None, None, None)  # free the slot as soon as the search ends

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _owned_run(
    opener: AgentOpener, guard: Guard, request: Request, run_id: str
) -> PausedRun | FinishedRun:
    """The run, if it exists and the request carries its key; otherwise 404 either way."""
    try:
        with opener() as agent:
            run = agent.status(run_id)
    except RunNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    guard.require_run_key(request, run.owner_hash, run_id)
    return run


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
