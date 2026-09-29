"""Model gateway: the one place that knows which LLM providers exist.

    from jobscout.llm import structured_llm
    extractor = structured_llm(MySchema)          # Runnable that returns MySchema
    result = extractor.invoke(messages, config=trace_config(...))

Callers ask for a schema and get a validated object back. Provider choice,
fallback order (Gemini Flash -> Flash-Lite -> OpenRouter), retries and timeouts
stay in here, so swapping a model never touches agent code.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from jobscout.settings import Settings, get_settings

__all__ = ["LLMNotConfigured", "structured_llm"]

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMNotConfigured(RuntimeError):
    """No provider has credentials. The message says which variables to set."""


def structured_llm(
    schema: type[SchemaT], *, settings: Settings | None = None
) -> Runnable[LanguageModelInput, SchemaT]:
    """A runnable that returns `schema` instances, trying each configured model in turn.

    A provider error, rate limit or schema-invalid reply moves on to the next model.
    Each model makes a single attempt (no internal retries): on free tiers a
    rate-limited model rarely recovers within seconds, so falling back is faster.
    """
    s = settings or get_settings()
    candidates: list[Runnable[LanguageModelInput, SchemaT]] = []

    if _is_set(s.google_api_key):
        for model in dict.fromkeys([s.gemini_model, s.gemini_fallback_model]):
            # Temperature is left unset on purpose: Gemini 3 models are tuned for
            # their default sampling and the integration applies it.
            chat = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=s.google_api_key,
                max_retries=1,  # 1 means "no retries" in the Google SDK
                request_timeout=s.llm_timeout_seconds,
            )
            candidates.append(
                chat.with_structured_output(schema).with_config(run_name=f"gemini/{model}")
            )

    if _is_set(s.openrouter_api_key) and s.openrouter_model:
        chat_or = ChatOpenAI(
            model=s.openrouter_model,
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key,
            max_retries=0,
            timeout=s.llm_timeout_seconds,
        )
        # Free OpenRouter models support tool calling far more often than JSON schema mode.
        candidates.append(
            chat_or.with_structured_output(schema, method="function_calling").with_config(
                run_name=f"openrouter/{s.openrouter_model}"
            )
        )

    if not candidates:
        raise LLMNotConfigured(
            "No LLM is configured. Set GOOGLE_API_KEY in .env, "
            "or OPENROUTER_API_KEY together with OPENROUTER_MODEL."
        )

    primary, *fallbacks = candidates
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value())
