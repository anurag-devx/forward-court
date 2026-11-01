"""URL detection and shortener expansion (HEAD-only, no body download)."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

import httpx

from config import LINK_SHORTENERS

URL_RE = re.compile(r"https?://[^\s<>\"]+")
_TRAILING = ".,);:!?\"'"
MAX_HOPS = 5
HEAD_TIMEOUT_S = 8.0


def find_urls(text: str) -> list[str]:
    """All http(s) URLs in text, deduplicated, trailing punctuation stripped."""
    urls: list[str] = []
    for match in URL_RE.findall(text or ""):
        url = match.rstrip(_TRAILING)
        if url and url not in urls:
            urls.append(url)
    return urls


def is_shortener(url: str) -> bool:
    """True when the URL host is a known link shortener."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    if host.startswith("www."):
        host = host[4:]
    return host in LINK_SHORTENERS


def expand_url(url: str, client: httpx.Client) -> str:
    """Follow shortener redirects with HEAD only; original on any failure."""
    current = url
    try:
        for _ in range(MAX_HOPS):
            response = client.head(
                current, follow_redirects=False, timeout=HEAD_TIMEOUT_S
            )
            if response.status_code not in (301, 302, 303, 307, 308):
                return current
            location = response.headers.get("location", "")
            if not location:
                return current
            if location.startswith("/"):
                parts = urlsplit(current)
                location = f"{parts.scheme}://{parts.netloc}{location}"
            if not location.startswith(("http://", "https://")):
                return current
            current = location
    except httpx.HTTPError:
        pass
    return current


def expand_shorteners(urls: list[str], client: httpx.Client) -> dict[str, str]:
    """Map each shortener URL to its expanded target (HEAD hops only)."""
    return {
        url: expand_url(url, client) for url in urls if is_shortener(url)
    }
