"""Evidence selection: rank, domain-dedupe, and cap items per claim.

Runs after evidence assembly, before the relevance filter. Items carrying
a ClaimReview are always kept (exempt from cap and domain limit); the rest
rank tier-1 first, then fact-check items, then exact matches, then
grounding order, with max 2 kept per host and MAX_PER_CLAIM total.
"""
from __future__ import annotations

from .urls import split_host_path, tier_of

MAX_PER_CLAIM = 15
MAX_PER_DOMAIN = 2


def _rank(item, index: int) -> tuple:
    return (
        tier_of(item.source_url) != 1,
        not item.fact_check,
        not item.exact_match,
        index,
    )


def select_evidence(
    claims: list[str], evidence: list, tracker=None
) -> list:
    """Cap evidence per claim.

    ClaimReview carriers bypass ranking entirely; everything else is
    ranked, domain-deduped, and capped. Kept order: claims in input
    order, exempt items first, then rank order.
    """
    by_claim: dict[str, list] = {claim: [] for claim in claims}
    for index, item in enumerate(evidence):
        if item.claim in by_claim:
            by_claim[item.claim].append((index, item))
    kept: list = []
    for claim in claims:
        indexed = by_claim[claim]
        exempt = [item for _, item in indexed if item.claim_review is not None]
        ranked = sorted(
            ((index, item) for index, item in indexed
             if item.claim_review is None),
            key=lambda pair: _rank(pair[1], pair[0]),
        )
        seen_hosts: dict[str, int] = {}
        chosen: list = []
        for _, item in ranked:
            if len(chosen) >= MAX_PER_CLAIM:
                break
            host, _ = split_host_path(item.source_url)
            if seen_hosts.get(host or item.source_url, 0) >= MAX_PER_DOMAIN:
                continue
            seen_hosts[host or item.source_url] = (
                seen_hosts.get(host or item.source_url, 0) + 1
            )
            chosen.append(item)
        kept.extend(exempt + chosen)
    if tracker is not None:
        tracker.note(
            f"evidence capped {len(evidence) - len(kept)}/{len(evidence)} items."
        )
    return kept
