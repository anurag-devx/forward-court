"""Search-grounded evidence collection.

Uses the Responses API with the `web_search` tool plus
`include=["web_search_call.results"]` (per Meta's search-grounding docs).
Evidence URLs come ONLY from `web_search_call.results` metadata; the model's
text output is never used for URLs. Any grounding failure raises
GroundingError -- there is deliberately no ungrounded fallback.
"""
from __future__ import annotations

from config import AGENT_MODEL

from .dates import extract_date
from .models import EvidenceItem
from .urls import normalize_url

SEARCH_TOOL = {"type": "web_search", "search_context_size": "medium"}
INCLUDE_RESULTS = ["web_search_call.results"]

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


def gather_evidence(claims: list[str], *, client, tracker=None) -> list[EvidenceItem]:
    """Collect grounded evidence for each claim (supporting + refuting queries).

    Raises GroundingError when a query performs no grounded search, when the
    API call fails, or when a claim ends up with zero grounded items.
    """
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for claim in claims:
        claim_items: list[EvidenceItem] = []
        queries = (
            SUPPORT_QUERY.format(claim=claim),
            REFUTE_QUERY.format(claim=claim),
        )
        for query in queries:
            try:
                response = client.responses.create(
                    model=AGENT_MODEL,
                    input=query,
                    tools=[dict(SEARCH_TOOL)],
                    include=list(INCLUDE_RESULTS),
                )
            except Exception as exc:
                raise GroundingError(
                    f"Search grounding failed for claim {claim!r}: {exc}"
                ) from exc
            if tracker is not None:
                tracker.add(response.usage)
            results, searched = _results_from_output(response.output)
            if not searched:
                raise GroundingError(
                    f"Model performed no grounded search for claim {claim!r}; stopping."
                )
            for result in results:
                url = _field(result, "url")
                if not url:
                    continue
                normalized = normalize_url(url)
                if normalized in seen:
                    continue
                seen.add(normalized)
                title = _field(result, "title") or ""
                snippet = _field(result, "snippet") or ""
                claim_items.append(
                    EvidenceItem(
                        claim=claim,
                        source_url=url,
                        date=extract_date(f"{title}\n{snippet}"),
                        snippet=snippet or title,
                    )
                )
        if not claim_items:
            raise GroundingError(
                f"No grounded evidence found for claim {claim!r}; stopping."
            )
        items.extend(claim_items)
    return items
