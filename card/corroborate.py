"""Corroborating sources for verdict cards: deterministic, no model calls."""
from __future__ import annotations

from evidence.match import key_mismatched
from evidence.urls import (
    is_factcheck_url,
    normalize_url,
    split_host_path,
    tier_of,
)

MAX_CORROBORATION = 2


def outlet_label(url: str) -> str:
    """Outlet host for card display (www stripped, full URL on failure)."""
    host, _ = split_host_path(url)
    return host or url


def _is_factcheck_item(item) -> bool:
    return bool(
        getattr(item, "fact_check", False)
        or item.claim_review is not None
        or is_factcheck_url(item.source_url)
    )


def _agrees(item, verdict: str) -> bool:
    """True when the item's stance agrees with the verdict."""
    rating = item.claim_review.verdict if item.claim_review else None
    if verdict == "False":
        return rating in ("False", "Misleading")
    if verdict == "True":
        return rating is None or rating == "True"
    if verdict == "Misleading":
        return rating is None or rating in ("True", "Misleading")
    return False


def select_corroboration(
    verdict, claims: list[str], evidence, max_items: int = MAX_CORROBORATION,
) -> list:
    """Up to max_items extra sources backing the verdict.

    Picks from already-gathered evidence: agreeing rated fact-checkers
    first, then other tier-1 outlets. Tier-2/3 never qualify. Each pick is
    on-claim (same veto as True key_sources), from its own domain, and
    never the key source or its domain. Empty when none qualify.
    """
    if verdict.verdict not in ("True", "False", "Misleading"):
        return []
    key_url = normalize_url(verdict.key_source)
    key_host, _ = split_host_path(verdict.key_source)
    candidates = []
    for item in evidence:
        if normalize_url(item.source_url) == key_url:
            continue
        host, _ = split_host_path(item.source_url)
        if host is not None and host == key_host:
            continue
        if tier_of(item.source_url) != 1:
            continue
        if getattr(item, "visual_only", False):
            continue
        if getattr(item, "shared_article", False):
            continue
        if key_mismatched(item, claims):
            continue
        if not _agrees(item, verdict.verdict):
            continue
        candidates.append(item)
    candidates.sort(key=lambda item: 0 if _is_factcheck_item(item) else 1)
    picked, seen = [], set()
    for item in candidates:
        host, _ = split_host_path(item.source_url)
        if host in seen:
            continue
        seen.add(host)
        picked.append(item)
        if len(picked) >= max_items:
            break
    return picked
