"""Model gateway: the one place that knows which LLM providers exist.

    from jobscout.llm import structured_llm
    extractor = structured_llm(MySchema)                  # careful work: profile, scoring
    planner = structured_llm(Plan, task="light")          # short, simple answers
    result = extractor.invoke(messages, config=trace_config(...))

Callers say what kind of task it is and get a validated object back. Model
order, thinking depth, retries and timeouts stay in here.

- "reasoning": Gemini 3.8 Flash -> 3.7 Flash -> Flash-Lite -> OpenRouter
- "light":     Flash-Lite (low thinking) -> 3.8 Flash (low thinking) -> OpenRouter
- "scoring":   Claude (SCORER_MODEL, prompt-cached system prompt) -> the "reasoning" chain
- "screening": Claude Haiku (TRUST_MODEL) -> the "light" chain
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeVar

from google.genai.errors import ServerError
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from jobscout.settings import Settings, get_settings

__all__ = ["LLMNotConfigured", "Task", "structured_llm"]

SchemaT = TypeVar("SchemaT", bound=BaseModel)
Task = Literal["reasoning", "light", "scoring", "screening"]


class LLMNotConfigured(RuntimeError):
    """No provider has credentials. The message says which variables to set."""


class _HideAFCNotice(logging.Filter):
    # LangChain sends our output schema to Gemini as a "function", which makes the
    # Google SDK print a one-time notice about Automatic Function Calling. Gemini never
    # runs any of our code here, so the notice is noise.
    def filter(self, record: logging.LogRecord) -> bool:
        return "automatic function calling (AFC)" not in record.getMessage()


logging.getLogger("google_genai.models").addFilter(_HideAFCNotice())


def structured_llm(
    schema: type[SchemaT],
    *,
    task: Task = "reasoning",
    model: str | None = None,
    fallbacks: bool = True,
    settings: Settings | None = None,
) -> Runnable[LanguageModelInput, SchemaT]:
    """A runnable that returns `schema` instances, trying each configured model in turn.

    Each Gemini model is retried on server errors (503 "high demand"); any other
    error, a rate limit, a timeout or a schema-invalid reply moves to the next model.
    For "scoring", `model` picks the Claude model and `fallbacks=False` uses only it
    (for side-by-side model comparisons).
    """
    s = settings or get_settings()
    light = task in ("light", "screening")
    timeout = s.llm_light_timeout_seconds if light else s.llm_timeout_seconds
    candidates: list[Runnable[LanguageModelInput, SchemaT]] = []

    if task in ("scoring", "screening") and _is_set(s.anthropic_api_key):
        scoring = task == "scoring"
        claude_model = model or (s.scorer_model if scoring else s.trust_model)
        options: dict[str, Any] = {}
        if scoring:  # Haiku rejects the effort setting, so only the scorer sets it
            options["effort"] = s.scorer_effort  # judgement, not the deepest reasoning
        claude = ChatAnthropic(
            model=claude_model,
            api_key=s.anthropic_api_key,
            max_tokens=16_000 if scoring else 4_000,
            timeout=s.scorer_timeout_seconds,
            max_retries=2,
            **options,
        )
        structured = claude.with_structured_output(schema, method="json_schema")
        # The scorer's system prompt is long and repeated across batches: worth caching.
        prepared = (RunnableLambda(_cache_system_prompt) | structured) if scoring else structured
        candidates.append(prepared.with_config(run_name=f"claude/{claude_model}"))
        if not fallbacks:
            return candidates[0]

    if _is_set(s.google_api_key):
        if light:  # writing a search phrase doesn't need the strongest model or deep thinking
            models = [s.gemini_fallback_model, s.gemini_model]
        else:
            models = [s.gemini_model, s.gemini_second_model, s.gemini_fallback_model]
        for model in dict.fromkeys(m for m in models if m):
            # Temperature is left unset on purpose: Gemini 3 models are tuned for
            # their default sampling and the integration applies it.
            chat = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=s.google_api_key,
                max_retries=1,  # 1 means "no retries" in the Google SDK
                request_timeout=timeout,
                thinking_level=s.llm_light_thinking_level if light else None,
            )
            structured = chat.with_structured_output(schema).with_retry(
                # 5xx only: "model is experiencing high demand" often clears within seconds.
                retry_if_exception_type=(ServerError,),
                stop_after_attempt=s.llm_server_error_attempts,
                wait_exponential_jitter=True,
            )
            candidates.append(structured.with_config(run_name=f"gemini/{model}"))

    if _is_set(s.openrouter_api_key) and s.openrouter_model:
        chat_or = ChatOpenAI(
            model=s.openrouter_model,
            base_url=s.openrouter_base_url,
            api_key=s.openrouter_api_key,
            max_retries=0,
            timeout=timeout,
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


def _cache_system_prompt(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Mark the system prompt as cacheable for Claude.

    The scorer's system prompt (rules, rubric, profile) is identical across its
    batches; only the jobs change. With this marker, batches 2+ read it from
    Claude's prompt cache at a tenth of the input price. Gemini fallbacks get the
    plain messages, because the marker is Claude-specific.
    """
    marked: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            block = {
                "type": "text",
                "text": message.content,
                "cache_control": {"type": "ephemeral"},
            }
            message = SystemMessage(content=[block])
        marked.append(message)
    return marked


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value())
