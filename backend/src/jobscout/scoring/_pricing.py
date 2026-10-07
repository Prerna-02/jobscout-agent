"""List prices for the Claude scorer, used to compare models in dollars.

Anthropic first-party API rates per 1M tokens (from the Claude API reference
cached 2026-06-24; check https://www.anthropic.com/pricing before relying on them).
Prompt-cache writes cost 1.25x input; cache reads cost 0.1x input.
"""

from __future__ import annotations

from typing import Any

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def claude_cost(model: str, usage: dict[str, Any]) -> float | None:
    """Dollar cost of one model's usage from LangChain usage_metadata; None if unpriced."""
    price = next((p for name, p in PRICES_PER_MTOK.items() if model.startswith(name)), None)
    if price is None:
        return None
    input_price, output_price = price
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0) or 0
    cache_write = details.get("cache_creation", 0) or 0
    fresh_input = max(0, usage.get("input_tokens", 0) - cache_read - cache_write)
    dollars = (
        fresh_input * input_price
        + cache_write * input_price * 1.25
        + cache_read * input_price * 0.10
        + usage.get("output_tokens", 0) * output_price
    )
    return dollars / 1_000_000
