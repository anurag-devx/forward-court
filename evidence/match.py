"""Claim-evidence matching shared by the judge and the verdict card."""
from __future__ import annotations

import re

from evidence.urls import is_factcheck_url

CLAIM_MATCH_TOKENS = 3


def significant_tokens(text: str) -> set[str]:
    cleaned = "".join(
        char.lower() if char.isalnum() or char.isspace() else " "
        for char in text or ""
    )
    return {token for token in cleaned.split() if len(token) > 3}


def review_matches_claim(item, claim: str) -> bool:
    """True when the item's ClaimReview applies to the claim.

    Matches on normalized claim text (containment or shared significant
    tokens), or -- for cross-language cases -- when the item came from the
    lookup query for this exact claim with relevance score 2.
    """
    review = item.claim_review
    if review is None:
        return False
    if (
        getattr(item, "fact_check", False)
        and item.claim == claim
        and getattr(item, "relevance", None) == 2
    ):
        return True
    reviewed = (review.claim_text or "").strip()
    if not reviewed:
        return False
    left = " ".join(reviewed.lower().split())
    right = " ".join((claim or "").lower().split())
    if (
        min(len(left), len(right)) >= 8
        and (left in right or right in left)
    ):
        return True
    shared = significant_tokens(reviewed) & significant_tokens(claim)
    return len(shared) >= CLAIM_MATCH_TOKENS


def snippet_states_claim(snippet: str | None, claim: str) -> bool:
    """True when the snippet's lead sentence is about the claim."""
    first = re.split(r"[.!?…]\s*", snippet or "", maxsplit=1)[0]
    shared = significant_tokens(first) & significant_tokens(claim)
    return len(shared) >= CLAIM_MATCH_TOKENS


def key_mismatched(item, claims: list[str]) -> bool:
    """True when a fact-check item is about a different claim.

    Direct reports are always key-eligible; only fact-checks are vetted,
    against their ClaimReview text or (without one) their snippet's lead.
    Outlet URL counts too: grounding hits from fact-check sites arrive
    without the lookup flag.
    """
    if not (
        getattr(item, "fact_check", False)
        or item.claim_review is not None
        or is_factcheck_url(item.source_url)
    ):
        return False
    return not any(
        review_matches_claim(item, claim)
        or snippet_states_claim(item.snippet, claim)
        for claim in claims
    )
