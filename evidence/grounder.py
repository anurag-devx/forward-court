"""Search-grounded evidence collection.

Uses the Responses API with the `web_search` tool plus
`include=["web_search_call.results"]` (per Meta's search-grounding docs).
Evidence URLs come ONLY from `web_search_call.results` metadata; the model's
text output is never used for URLs. Any grounding failure raises
GroundingError -- there is deliberately no ungrounded fallback.
"""
from __future__ import annotations

from config import agent_model

from .dates import extract_date
from .models import EvidenceItem
from .pages import (
    extract_main_text,
    extract_publish_date,
    fetch_html,
    make_excerpt,
)
from .urls import normalize_url

SEARCH_TOOL = {"type": "web_search", "search_context_size": "low"}
INCLUDE_RESULTS = ["web_search_call.results"]
TOP_FETCH_PER_CLAIM = 3
MAX_TOOL_CALLS = 2

SUPPORT_QUERY = "Find web evidence SUPPORTING this claim: {claim}"
REFUTE_QUERY = (
    "Find web evidence REFUTING, debunking, or fact-checking this claim: {claim}"
)


class GroundingError(RuntimeError):
    """Search grounding failed. Stop the run; never use ungrounded URLs."""


def _field(obj, name: str):
    """Read a field from either a dict or an object (the live API mixes both)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _results_from_output(output) -> tuple[list, bool]:
    """Return (results, searched) from a Responses API output list.

    `searched` is True only when at least one web_search_call completed.
    Duck-typed so tests can use simple fake objects.
    """
    results: list = []
    searched = False
    for item in output or []:
        if _field(item, "type") != "web_search_call":
            continue
        if _field(item, "status") == "completed":
            searched = True
            results.extend(_field(item, "results") or [])
    return results, searched


def _interleave(per_query: list[list]) -> list:
    """Round-robin across per-query result lists so both sides rank evenly."""
    ordered: list = []
    for index in range(max((len(q) for q in per_query), default=0)):
        for results in per_query:
            if index < len(results):
                ordered.append(results[index])
    return ordered


def _enrich(url: str, title: str, fallback: str, *, http_client=None) -> tuple[str, str | None]:
    """Fetch a page for a 500-char excerpt + meta date; fall back silently."""
    snippet = fallback or title
    date = extract_date(f"{title}\n{snippet}")
    html = fetch_html(url, client=http_client)
    if not html:
        return snippet, date
    excerpt = make_excerpt(extract_main_text(html))
    if excerpt:
        snippet = excerpt
    return snippet, extract_publish_date(html) or extract_date(f"{title}\n{snippet}")


def gather_evidence(
    claims: list[str], *, client, tracker=None, http_client=None
) -> list[EvidenceItem]:
    """Collect grounded evidence for each claim (supporting + refuting queries).

    The top TOP_FETCH_PER_CLAIM unique URLs per claim are fetched for an
    article excerpt and meta publish date; fetch failures keep the search
    snippet and never abort the run. Raises GroundingError when a query
    performs no grounded search, when the API call fails, or when a claim
    ends up with zero grounded items.
    """
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for number, claim in enumerate(claims, start=1):
        queries = (
            ("support", SUPPORT_QUERY.format(claim=claim)),
            ("refute", REFUTE_QUERY.format(claim=claim)),
        )
        per_query: list[list] = []
        for side, query in queries:
            try:
                response = client.responses.create(
                    model=agent_model(),
                    input=query,
                    tools=[dict(SEARCH_TOOL)],
                    include=list(INCLUDE_RESULTS),
                    max_tool_calls=MAX_TOOL_CALLS,
                )
            except Exception as exc:
                raise GroundingError(
                    f"Search grounding failed for claim {claim!r}: {exc}"
                ) from exc
            if tracker is not None:
                tracker.add(
                    response.usage, stage=f"grounding:claim{number}:{side}"
                )
            results, searched = _results_from_output(response.output)
            if not searched:
                raise GroundingError(
                    f"Model performed no grounded search for claim {claim!r}; stopping."
                )
            per_query.append(results)
        claim_items: list[EvidenceItem] = []
        fetch_attempts = 0
        for result in _interleave(per_query):
            url = _field(result, "url")
            if not url:
                continue
            normalized = normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            title = _field(result, "title") or ""
            search_snippet = _field(result, "snippet") or ""
            if fetch_attempts < TOP_FETCH_PER_CLAIM:
                fetch_attempts += 1
                snippet, date = _enrich(
                    url, title, search_snippet, http_client=http_client
                )
            else:
                snippet = search_snippet or title
                date = extract_date(f"{title}\n{snippet}")
            claim_items.append(
                EvidenceItem(
                    claim=claim, source_url=url, date=date, snippet=snippet
                )
            )
        if not claim_items:
            raise GroundingError(
                f"No grounded evidence found for claim {claim!r}; stopping."
            )
        items.extend(claim_items)
    return items
