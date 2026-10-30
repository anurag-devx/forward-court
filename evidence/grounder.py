"""Search-grounded evidence collection.

Uses the Responses API with the `web_search` tool plus
`include=["web_search_call.results"]` (per Meta's search-grounding docs).
Evidence URLs come ONLY from `web_search_call.results` metadata; the model's
text output is never used for URLs. Any grounding failure raises
GroundingError -- there is deliberately no ungrounded fallback.
"""
from __future__ import annotations

from config import agent_model

from dates import extract_date
from .models import EvidenceItem
from .pages import enrich_from_page
from .responses import field, web_search_results
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


def _interleave(per_query: list[list]) -> list:
    """Round-robin across per-query result lists so both sides rank evenly."""
    ordered: list = []
    for index in range(max((len(q) for q in per_query), default=0)):
        for results in per_query:
            if index < len(results):
                ordered.append(results[index])
    return ordered


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
    if tracker is not None:
        tracker.mark("grounding")
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
            results, searched = web_search_results(response.output)
            if not searched:
                raise GroundingError(
                    f"Model performed no grounded search for claim {claim!r}; stopping."
                )
            per_query.append(results)
        claim_items: list[EvidenceItem] = []
        fetch_attempts = 0
        for result in _interleave(per_query):
            url = field(result, "url")
            if not url:
                continue
            normalized = normalize_url(url)
            if normalized in seen:
                continue
            seen.add(normalized)
            title = field(result, "title") or ""
            search_snippet = field(result, "snippet") or ""
            if fetch_attempts < TOP_FETCH_PER_CLAIM:
                fetch_attempts += 1
                snippet, date, review = enrich_from_page(
                    url, title, search_snippet, http_client=http_client
                )
            else:
                snippet = search_snippet or title
                date = extract_date(f"{title}\n{snippet}")
                review = None
            claim_items.append(
                EvidenceItem(
                    claim=claim, source_url=url, date=date, snippet=snippet,
                    claim_review=review,
                )
            )
        if not claim_items:
            raise GroundingError(
                f"No grounded evidence found for claim {claim!r}; stopping."
            )
        items.extend(claim_items)
    return items
