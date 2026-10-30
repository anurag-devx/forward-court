"""Helpers for reading Responses API output (dicts and objects alike)."""
from __future__ import annotations


def field(obj, name: str):
    """Read a field from either a dict or an object (the API mixes both)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def web_search_results(output) -> tuple[list, bool]:
    """Return (results, searched) from a Responses API output list.

    `searched` is True only when at least one web_search_call completed.
    """
    results: list = []
    searched = False
    for item in output or []:
        if field(item, "type") != "web_search_call":
            continue
        if field(item, "status") == "completed":
            searched = True
            results.extend(field(item, "results") or [])
    return results, searched


def output_text(output) -> str:
    """Concatenate output_text blocks from message items."""
    parts: list[str] = []
    for item in output or []:
        if field(item, "type") != "message":
            continue
        for block in field(item, "content") or []:
            if field(block, "type") == "output_text":
                parts.append(field(block, "text") or "")
    return "".join(parts)
