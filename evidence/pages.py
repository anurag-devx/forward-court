"""Fetch article pages and extract publish dates (offline-safe, failure-tolerant).

All functions return None/""/fallbacks on failure: callers treat page
enrichment as best-effort and never abort a run over it.
"""
from __future__ import annotations

import json
from datetime import datetime
from html.parser import HTMLParser

import httpx
import trafilatura

from dates import extract_date

from .models import ClaimReview

FETCH_TIMEOUT = 8.0
MAX_BYTES = 2 * 1024 * 1024
EXCERPT_CHARS = 500
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 ForwardCourt/0.1"
)


def _fetch(
    client: httpx.Client, url: str, max_bytes: int, content_must_contain: tuple[str, ...]
) -> tuple[bytes, str] | None:
    try:
        with client.stream(
            "GET",
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_must_contain and not any(
                s in content_type for s in content_must_contain
            ):
                return None
            size = 0
            chunks: list[bytes] = []
            for chunk in response.iter_bytes(65536):
                size += len(chunk)
                if size > max_bytes:
                    return None
                chunks.append(chunk)
    except (httpx.HTTPError, ValueError):
        return None
    return b"".join(chunks), response.charset_encoding or "utf-8"


def _is_http(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def fetch_html(url: str, *, client: httpx.Client | None = None) -> str | None:
    """GET a page as text. Returns None on any failure or >2MB body."""
    if not _is_http(url):
        return None
    if client is None:
        with httpx.Client() as owned:
            fetched = _fetch(owned, url, MAX_BYTES, ("html", "text"))
    else:
        fetched = _fetch(client, url, MAX_BYTES, ("html", "text"))
    if fetched is None:
        return None
    content, encoding = fetched
    return content.decode(encoding, errors="replace")


def download_bytes(
    url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_BYTES,
    content_must_contain: tuple[str, ...] = (),
) -> bytes | None:
    """GET a URL as bytes (images, files). None on failure or over max_bytes."""
    if not _is_http(url):
        return None
    if client is None:
        with httpx.Client() as owned:
            fetched = _fetch(owned, url, max_bytes, content_must_contain)
    else:
        fetched = _fetch(client, url, max_bytes, content_must_contain)
    if fetched is None:
        return None
    return fetched[0]


def enrich_from_page(
    url: str, title: str, fallback: str, *, http_client=None
) -> tuple[str, str | None, ClaimReview | None]:
    """Fetch a page for excerpt + publish date + ClaimReview; fall back silently."""
    snippet = fallback or title
    date = extract_date(f"{title}\n{snippet}")
    html = fetch_html(url, client=http_client)
    if not html:
        return snippet, date, None
    excerpt = make_excerpt(extract_main_text(html))
    if excerpt:
        snippet = excerpt
    date = extract_publish_date(html) or extract_date(f"{title}\n{snippet}")
    return snippet, date, extract_claim_review(html)


class _MetaParser(HTMLParser):
    """Collects article:published_time and JSON-LD script bodies."""

    def __init__(self) -> None:
        super().__init__()
        self.published_time: str | None = None
        self.ld_json_blocks: list[str] = []
        self._in_ld_json = False
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "meta":
            fields = dict(attrs)
            if (
                fields.get("property") == "article:published_time"
                and fields.get("content")
                and self.published_time is None
            ):
                self.published_time = fields["content"]
        elif tag == "script":
            if dict(attrs).get("type") == "application/ld+json":
                self._in_ld_json = True
                self._current = []

    def handle_data(self, data: str) -> None:
        if self._in_ld_json:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_ld_json:
            self._in_ld_json = False
            self.ld_json_blocks.append("".join(self._current))


def _day(value) -> str | None:
    """Normalize a date value to ISO YYYY-MM-DD, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip()).date().isoformat()
    except ValueError:
        return extract_date(value)


def _ld_date_published(block: str) -> str | None:
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return None
    candidates = data if isinstance(data, list) else [data]
    for node in candidates:
        if not isinstance(node, dict):
            continue
        day = _day(node.get("datePublished"))
        if day is not None:
            return day
        graph = node.get("@graph")
        if isinstance(graph, list):
            for child in graph:
                if isinstance(child, dict) and (day := _day(child.get("datePublished"))) is not None:
                    return day
    return None


RATING_VERDICTS = {
    "false": "False",
    "fake": "False",
    "debunked": "False",
    "misleading": "Misleading",
    "partly false": "Misleading",
    "partially false": "Misleading",
    "missing context": "Misleading",
    "altered": "Misleading",
    "manipulated": "Misleading",
    "satire": "Misleading",
    "true": "True",
    "correct": "True",
    "accurate": "True",
}

_SCHEMA_BEST = 5.0
_SCHEMA_WORST = 1.0


def _ld_nodes(block: str) -> list[dict]:
    """JSON-LD nodes from a script body, unwrapping @graph lists."""
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return []
    candidates = data if isinstance(data, list) else [data]
    nodes: list[dict] = []
    for node in candidates:
        if not isinstance(node, dict):
            continue
        nodes.append(node)
        graph = node.get("@graph")
        if isinstance(graph, list):
            nodes.extend(child for child in graph if isinstance(child, dict))
    return nodes


def _is_claim_review(node: dict) -> bool:
    node_type = node.get("@type")
    if isinstance(node_type, list):
        return "ClaimReview" in node_type
    return node_type == "ClaimReview"


def _rating_verdict(node: dict) -> tuple[str, str] | None:
    """Map a ClaimReview node's reviewRating to (raw, verdict)."""
    rating = node.get("reviewRating")
    if not isinstance(rating, dict):
        return None
    alternate = rating.get("alternateName")
    if isinstance(alternate, str) and alternate.strip():
        raw = alternate.strip()
        verdict = RATING_VERDICTS.get(raw.lower())
        if verdict is not None:
            return raw, verdict
    value = _rating_number(rating.get("ratingValue"))
    if value is None:
        return None
    raw = str(rating.get("ratingValue")).strip()
    best = _rating_number(rating.get("bestRating"), default=_SCHEMA_BEST)
    worst = _rating_number(rating.get("worstRating"), default=_SCHEMA_WORST)
    if best == worst:
        return raw, "Unverified"
    if value <= min(best, worst):
        return raw, "False"
    if value >= max(best, worst):
        return raw, "True"
    return raw, "Unverified"


def _rating_number(value, default: float | None = None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default


def _reviewed_claim(node: dict) -> str:
    reviewed = node.get("claimReviewed", node.get("itemReviewed"))
    if isinstance(reviewed, dict):
        return str(reviewed.get("name", reviewed.get("text", "")) or "")
    return str(reviewed or "")


def extract_claim_review(html: str) -> ClaimReview | None:
    """Parse the first schema.org ClaimReview rating from JSON-LD blocks."""
    if not html:
        return None
    parser = _MetaParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    for block in parser.ld_json_blocks:
        for node in _ld_nodes(block):
            if not _is_claim_review(node):
                continue
            mapped = _rating_verdict(node)
            if mapped is None:
                continue
            raw, verdict = mapped
            return ClaimReview(
                rating_raw=raw, verdict=verdict,
                claim_text=_reviewed_claim(node),
            )
    return None


def extract_publish_date(html: str) -> str | None:
    """Publish date from article:published_time, else JSON-LD datePublished."""
    if not html:
        return None
    parser = _MetaParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    if (day := _day(parser.published_time)) is not None:
        return day
    for block in parser.ld_json_blocks:
        if (day := _ld_date_published(block)) is not None:
            return day
    return None


def extract_main_text(html: str) -> str:
    """Main article text via trafilatura; "" when extraction fails."""
    if not html:
        return ""
    try:
        return trafilatura.extract(html) or ""
    except Exception:
        return ""


def make_excerpt(text: str, chars: int = EXCERPT_CHARS) -> str:
    """Whitespace-collapsed leading excerpt of the article text."""
    return " ".join(text.split())[:chars]
