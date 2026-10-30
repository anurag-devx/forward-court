"""Existing-fact-check lookup: grounded searches restricted to trusted domains.

Queries the claim (plus extracted entity names) with `site:` OR-clauses
over trusted fact-checker hosts -- the Responses API rejects the
`filters.allowed_domains` tool param, so query-side restriction is used
instead. Non-English claims also get an English entity query. Results are
post-filtered to trusted domains/paths and converted to evidence tagged
as existing fact-checks. Best-effort: zero matches or a failed search
yields no items, never an error.
"""
from __future__ import annotations

import re

from config import TRUSTED_FACTCHECK_DOMAINS, agent_model
from dates import extract_date

from .grounder import INCLUDE_RESULTS, MAX_TOOL_CALLS, SEARCH_TOOL
from .models import EvidenceItem
from .pages import enrich_from_page
from .responses import field, web_search_results
from .urls import matches_entries, normalize_url, split_host_path

LOOKUP_QUERY = "fact check: {claim}"
MAX_MATCHES_PER_CLAIM = 3
ENRICH_MATCHES = 3

_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\b")


def trusted_hosts() -> list[str]:
    """Bare trusted hostnames (paths stripped) for site: query clauses."""
    hosts: list[str] = []
    for entry in TRUSTED_FACTCHECK_DOMAINS:
        host = entry.split("/", 1)[0].lower()
        if host not in hosts:
            hosts.append(host)
    return hosts


def site_clause() -> str:
    """OR-clause restricting backend search to trusted hosts."""
    return " ".join(f"site:{host}" if i == 0 else f"OR site:{host}"
                    for i, host in enumerate(trusted_hosts()))


def extract_entities(claim: str) -> list[str]:
    """Proper-noun candidates: quoted spans + Latin capitalized runs."""
    found: list[str] = []
    for pattern in (_QUOTED_RE, _CAPITALIZED_RE):
        for match in pattern.finditer(claim or ""):
            text = match.group(1) if pattern is _QUOTED_RE else match.group(0)
            text = " ".join(text.split())
            if text and text not in found:
                found.append(text)
    return found


def lookup_queries(claim: str, language: str) -> list[tuple[str, str]]:
    """(stage_suffix, query) pairs: claim language + English for entities."""
    clause = site_clause()
    entities = extract_entities(claim)
    base = LOOKUP_QUERY.format(claim=claim)
    if entities:
        base = f"{base} {' '.join(entities)}"
    queries = [("", f"{base} {clause}")]
    if language != "en":
        latin = [entity for entity in entities if entity.isascii()]
        if latin:
            queries.append((":en", f"fact check: {' '.join(latin)} {clause}"))
    return queries


def is_trusted_factcheck(url: str) -> bool:
    """True when the URL is on a trusted fact-checker domain/path."""
    host, path = split_host_path(url)
    if host is None:
        return False
    return matches_entries(host, path, TRUSTED_FACTCHECK_DOMAINS)


def lookup_factchecks(
    claims: list[str], *, client, tracker=None, http_client=None,
    language: str = "en",
) -> list[EvidenceItem]:
    """Find existing fact-checks per claim. Never raises for empty results."""
    tool = dict(SEARCH_TOOL)
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for number, claim in enumerate(claims, start=1):
        kept = 0
        for suffix, query in lookup_queries(claim, language):
            if kept >= MAX_MATCHES_PER_CLAIM:
                break
            try:
                response = client.responses.create(
                    model=agent_model(),
                    input=query,
                    tools=[dict(tool)],
                    include=list(INCLUDE_RESULTS),
                    max_tool_calls=MAX_TOOL_CALLS,
                )
            except Exception:
                continue
            if tracker is not None:
                tracker.add(
                    response.usage, stage=f"factcheck:claim{number}{suffix}"
                )
            results, searched = web_search_results(response.output)
            if not searched:
                continue
            for result in results:
                url = field(result, "url")
                if not url:
                    continue
                normalized = normalize_url(url)
                if normalized in seen or not is_trusted_factcheck(url):
                    continue
                seen.add(normalized)
                kept += 1
                if kept > MAX_MATCHES_PER_CLAIM:
                    break
                title = field(result, "title") or ""
                search_snippet = field(result, "snippet") or ""
                if kept <= ENRICH_MATCHES:
                    snippet, date, review = enrich_from_page(
                        url, title, search_snippet, http_client=http_client
                    )
                else:
                    snippet = search_snippet or title
                    date = extract_date(f"{title}\n{snippet}")
                    review = None
                items.append(
                    EvidenceItem(
                        claim=claim, source_url=url, date=date, snippet=snippet,
                        fact_check=True, claim_review=review,
                    )
                )
    return items
