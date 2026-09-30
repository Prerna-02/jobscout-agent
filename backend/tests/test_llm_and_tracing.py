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
