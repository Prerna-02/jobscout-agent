"""Application settings, loaded once from environment variables and `.env` files."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
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
    gemini_second_model: str | None = "gemini-3.7-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    # Retries per Gemini model on 5xx "high demand" errors. Rate limits (429) skip
    # straight to the next model, because waiting a few seconds doesn't lift a quota.
    llm_server_error_attempts: int = 2
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Seconds to wait for one model answer before moving on. Light tasks (a search
    # phrase) are short; a 60 s wait there once cost a 2-minute run.
    llm_timeout_seconds: float = 45.0
    llm_light_timeout_seconds: float = 15.0
    llm_light_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"

    # Langfuse tracing. Tracing is off when the keys are missing.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # Langfuse calls this LANGFUSE_BASE_URL; older docs use LANGFUSE_HOST. Either works.
    # US region: https://us.cloud.langfuse.com   EU region: https://cloud.langfuse.com
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )
    app_env: str = "dev"

    # Job search (JSearch via OpenWeb Ninja) and its Redis cache.
    jsearch_api_key: SecretStr | None = None
    jsearch_base_url: str = "https://api.openwebninja.com/jsearch"
    jsearch_timeout_seconds: float = 15.0
    jsearch_max_calls_per_run: int = 4
    # The free plan stops at 200 requests a month; keep a margin for manual testing.
    jsearch_monthly_limit: int = 190
    redis_url: SecretStr | None = None
    search_cache_ttl_seconds: int = 24 * 60 * 60

    # Agent (Phase 3). Paused runs are saved to Postgres when DATABASE_URL is set,
    # otherwise to a local SQLite file so pause/resume works without a database.
    database_url: SecretStr | None = None
    checkpoint_sqlite_path: str = ".jobscout/checkpoints.sqlite"
    agent_min_jobs: int = 10
    agent_max_rewrites: int = 3

    # Scoring (Phase 4). Claude scores the shortlist; Gemini is the fallback.
    anthropic_api_key: SecretStr | None = None
    scorer_model: str = "claude-sonnet-5"  # chosen after the Sonnet vs Opus comparison
    scorer_effort: Literal["low", "medium", "high"] = "medium"
    scorer_timeout_seconds: float = 120.0
    scorer_batch_size: int = 5
    shortlist_size: int = 15
    top_matches: int = 10
    embedding_model: str = "gemini-embedding-2"
    job_text_max_chars: int = 4000  # about 1,000 tokens per job ad sent for scoring
    profile_cache_ttl_seconds: int = 30 * 24 * 60 * 60
    # Trust check: Haiku reviews only postings the rules are unsure about.
    trust_model: str = "claude-haiku-4-5"
    trust_cache_ttl_seconds: int = 30 * 24 * 60 * 60
    score_cache_ttl_seconds: int = 7 * 24 * 60 * 60

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
