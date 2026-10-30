"""Collect image-related fact-check article URLs from RSS feeds.

Usage: python scripts/collect_factchecks.py [out_file] [feed_url ...]

Default feeds are Alt News and BOOM (verified RSS 2.0). Writes up to
MAX_URLS article links, one per line. A failing feed warns and is skipped.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence.pages import download_bytes

FEEDS = (
    "https://www.altnews.in/feed/",
    "https://www.boomlive.in/feed",
    "https://newschecker.in/feed/",
    "https://www.vishvasnews.com/feed/",
)
KEYWORDS = (
    "photo",
    "image",
    "picture",
    "viral image",
    "old photo",
    "तस्वीर",
    "फोटो",
    "वायरल",
)
MAX_URLS = 20
DEFAULT_OUT = Path("tests/samples/image_urls.txt")


def parse_feed(xml_text: str) -> list[tuple[str, str, str]]:
    """Parse RSS items into (title, link, description) triples."""
    items: list[tuple[str, str, str]] = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            link = (item.findtext("guid") or "").strip()
        description = (item.findtext("description") or "").strip()
        if link:
            items.append((title, link, description))
    return items


def is_image_related(title: str, description: str) -> bool:
    """True when title/description mentions image-like content."""
    haystack = f"{title}\n{description}".lower()
    return any(keyword in haystack for keyword in KEYWORDS)


def collect(feeds, *, client=None, max_urls: int = MAX_URLS) -> list[str]:
    """Gather deduplicated image-related article links across feeds."""
    urls: list[str] = []
    seen: set[str] = set()
    for feed_url in feeds:
        raw = download_bytes(feed_url, client=client)
        if raw is None:
            print(f"warning: fetch failed for {feed_url}", file=sys.stderr)
            continue
        try:
            items = parse_feed(raw.decode("utf-8", errors="replace"))
        except ET.ParseError:
            print(f"warning: invalid RSS from {feed_url}", file=sys.stderr)
            continue
        for title, link, description in items:
            if link not in seen and is_image_related(title, description):
                seen.add(link)
                urls.append(link)
                if len(urls) >= max_urls:
                    return urls
    return urls


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_file = Path(args[0]) if args else DEFAULT_OUT
    feeds = tuple(args[1:]) if len(args) > 1 else FEEDS
    urls = collect(feeds)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("".join(url + "\n" for url in urls))
    print(f"collected {len(urls)} urls to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
