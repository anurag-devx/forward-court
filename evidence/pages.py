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

from .dates import extract_date

FETCH_TIMEOUT = 8.0
MAX_BYTES = 2 * 1024 * 1024
EXCERPT_CHARS = 500
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 ForwardCourt/0.1"
)


def fetch_html(url: str, *, client: httpx.Client | None = None) -> str | None:
    """GET a page as text. Returns None on any failure or >2MB body."""
    if not url.lower().startswith(("http://", "https://")):
        return None
    if client is None:
        with httpx.Client() as owned:
            return _get(owned, url)
    return _get(client, url)


def _get(client: httpx.Client, url: str) -> str | None:
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
            if "html" not in content_type and "text" not in content_type:
                return None
            size = 0
            chunks: list[bytes] = []
            for chunk in response.iter_bytes(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    return None
                chunks.append(chunk)
    except (httpx.HTTPError, ValueError):
        return None
    encoding = response.charset_encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace")


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
