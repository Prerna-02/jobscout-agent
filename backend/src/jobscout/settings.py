"""Application settings, loaded once from environment variables and `.env` files."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value can be set as an environment variable of the same name (case-insensitive).

    `.env` is read from the current directory and its parent, so running from
    `backend/` still picks up the repository-root `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLMs: Gemini first, OpenRouter as the last fallback.
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_seconds: float = 60.0

    # Langfuse tracing. Tracing is off when the keys are missing.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    app_env: str = "dev"

    # CV intake limits. Vercel rejects request bodies above 4.5 MB.
    cv_max_bytes: int = 4 * 1024 * 1024
    cv_max_pages: int = 10
    cv_max_chars: int = 40_000

    @property
    def tracing_enabled(self) -> bool:
        secret = self.langfuse_secret_key.get_secret_value() if self.langfuse_secret_key else ""
        return bool(self.langfuse_public_key and secret)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
