"""The JobScout web API (FastAPI).

    uv run jobscout-api                     # http://127.0.0.1:8000/docs
    uvicorn jobscout.api:app --reload       # the same, with auto-reload while coding

`app` is created on first access, so importing this package (e.g. in tests) doesn't
connect to Redis or read settings.
"""

from __future__ import annotations

from typing import Any

from ._app import create_app

__all__ = ["create_app", "main"]

_app: Any = None


def __getattr__(name: str) -> Any:
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)


def main() -> None:
    import uvicorn

    uvicorn.run("jobscout.api:app", host="127.0.0.1", port=8000)
