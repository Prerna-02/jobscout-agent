import pytest
from langchain_core.runnables import RunnableWithFallbacks

from jobscout.llm import LLMNotConfigured, structured_llm
from jobscout.observability import trace_config
from jobscout.profile import ProfileExtraction
from jobscout.settings import Settings


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_no_keys_raises_a_helpful_error():
    with pytest.raises(LLMNotConfigured, match="GOOGLE_API_KEY"):
        structured_llm(ProfileExtraction, settings=_settings(google_api_key=""))


def test_fallback_order_gemini_38_then_37_then_lite_then_openrouter():
    chain = structured_llm(
        ProfileExtraction,
        settings=_settings(
            google_api_key="test-key",
            openrouter_api_key="test-key",
            openrouter_model="some/model:free",
        ),
    )
    assert isinstance(chain, RunnableWithFallbacks)
    names = [chain.runnable.config["run_name"], *(f.config["run_name"] for f in chain.fallbacks)]
    assert names == [
        "gemini/gemini-3.8-flash",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-3.5-flash-lite",
        "openrouter/some/model:free",
    ]


def test_openrouter_needs_a_model_name():
    with pytest.raises(LLMNotConfigured):
        structured_llm(ProfileExtraction, settings=_settings(openrouter_api_key="k"))


def test_trace_config_without_langfuse_keys_has_no_callbacks():
    config = trace_config(run_name="x", session_id="abc", tags=["cli"])
    assert "callbacks" not in config
    assert config["metadata"]["langfuse_session_id"] == "abc"
    assert config["metadata"]["langfuse_tags"] == ["env:dev", "cli"]


def test_langfuse_region_url_accepts_both_names(monkeypatch):
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    assert Settings(_env_file=None).langfuse_base_url == "https://us.cloud.langfuse.com"

    monkeypatch.delenv("LANGFUSE_BASE_URL")
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    assert Settings(_env_file=None).langfuse_base_url == "https://us.cloud.langfuse.com"


def test_light_tasks_start_with_flash_lite_and_low_thinking():
    chain = structured_llm(
        ProfileExtraction, task="light", settings=_settings(google_api_key="test-key")
    )
    names = [chain.runnable.config["run_name"], *(f.config["run_name"] for f in chain.fallbacks)]
    assert names == ["gemini/gemini-3.5-flash-lite", "gemini/gemini-3.8-flash"]


def test_afc_notice_is_hidden_but_other_google_warnings_are_not(caplog):
    import logging

    logger = logging.getLogger("google_genai.models")
    with caplog.at_level(logging.WARNING, logger="google_genai.models"):
        logger.warning("Direct use of automatic function calling (AFC) in Models...")
        logger.warning("Quota almost used up")
    assert [r.getMessage() for r in caplog.records] == ["Quota almost used up"]


def test_usage_summary_splits_thinking_tokens():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from jobscout.observability import UsageCounter, usage_summary

    counter = UsageCounter()
    message = AIMessage(
        content="",
        response_metadata={"model_name": "gemini-3.8-flash"},
        usage_metadata={
            "input_tokens": 209,
            "output_tokens": 507,
            "total_tokens": 716,
            "output_token_details": {"reasoning": 494},
        },
    )
    counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert usage_summary(counter) == {
        "gemini-3.8-flash": {"input": 209, "output": 507, "thinking": 494, "total": 716}
    }
