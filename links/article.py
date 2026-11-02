"""Safe news-link handling: fetch article, headline claim, evidence tag."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from dates import extract_date
from evidence.models import ClaimReview, EvidenceItem
from evidence.pages import (
    extract_claim_review,
    extract_main_text,
    extract_publish_date,
    extract_title,
    fetch_html,
    make_excerpt,
)
from evidence.urls import impersonates_outlet, tier_of


@dataclass
class ArticleInfo:
    url: str
    headline: str | None
    outlet: str
    date: str | None
    excerpt: str
    review: ClaimReview | None = None


def fetch_article(
    url: str, http_client: httpx.Client | None = None
) -> ArticleInfo | None:
    """Fetch a safety-cleared URL as an article; None when unusable.

    One retry on failure: a single transient network flake must not turn
    a shared article into 'no checkable claims'.
    """
    html = fetch_html(url, client=http_client)
    if not html:
        html = fetch_html(url, client=http_client)
    if not html:
        return None
    from urllib.parse import urlsplit

    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[4:]
    headline = extract_title(html)
    text = extract_main_text(html)
    excerpt = make_excerpt(text) or text[:500]
    pub_date = extract_publish_date(html) or extract_date(
        f"{headline or ''}\n{excerpt}"
    )
    return ArticleInfo(
        url=url,
        headline=headline,
        outlet=host,
        date=pub_date,
        excerpt=excerpt,
        review=extract_claim_review(html),
    )


def is_bare_link(text: str, urls: list[str]) -> bool:
    """True when the text carries no message beyond the URLs themselves."""
    stripped = text or ""
    for url in urls:
        stripped = stripped.replace(url, " ")
    return not stripped.strip()


def article_tier_note(article: ArticleInfo) -> str | None:
    """Outlet tier + impersonation note for the tracker, if noteworthy."""
    tier = tier_of(article.url)
    if impersonates_outlet(article.outlet):
        return f"shared article outlet {article.outlet} impersonates a known outlet."
    if tier == 1:
        return f"shared article from tier-1 outlet {article.outlet}."
    return None


def to_evidence(article: ArticleInfo, claim: str) -> EvidenceItem:
    """Shared article as tagged evidence (never its own key_source)."""
    return EvidenceItem(
        claim=claim,
        source_url=article.url,
        date=article.date,
        snippet=article.excerpt or article.headline or "",
        shared_article=True,
        claim_review=article.review,
    )
