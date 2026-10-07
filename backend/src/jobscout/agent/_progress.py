"""Plain-English progress messages, one per finished graph step, for live updates in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._planner import LADDER

_RUNG_MESSAGES = {
    "similar_titles": "Too few jobs, so trying similar job titles",
    "wider_dates": "Too few jobs, so including older postings",
    "wider_area": "Too few jobs, so searching the whole country",
}


@dataclass(frozen=True)
class Progress:
    step: str  # the graph node that just finished, e.g. "search_jobs"
    message: str


def describe_step(node: str, update: dict[str, Any]) -> str | None:
    """What a finished step means to the user, or None for steps not worth announcing."""
    if node == "plan_query":
        return f'Searching for "{update["query"]["text"]}"'
    if node == "search_jobs":
        search = update["searches"][-1]
        if search.get("error"):
            return f"Search failed: {search['error']}"
        source = " (from cache)" if search.get("cache_hit") else ""
        return f'Found {search["found"]} jobs for "{search["text"]}"{source}'
    if node == "filter_dedupe":
        return f"{len(update.get('jobs', []))} matching jobs so far"
    if node == "rewrite_query":
        if update.get("outcome") == "no_new_search":
            return "No broader search left to try"
        return _RUNG_MESSAGES[LADDER[update["rewrites"] - 1]]
    if node == "trust_check":
        checked, removed = len(update.get("trust", {})), len(update.get("removed_as_risky", []))
        tail = f", removed {removed} likely scam(s)" if removed else ""
        return f"Checked {checked} postings for scam signs{tail}" if checked else None
    if node == "shortlist":
        return f"Shortlisted the {len(update['shortlist'])} closest jobs"
    if node == "score_jobs":
        return f"Scored {len(update['matches'])} jobs with the {update['rubric_name']} scorecard"
    if node == "rank_jobs":
        return f"Ranked your top {len(update['ranked'])} jobs"
    return None
