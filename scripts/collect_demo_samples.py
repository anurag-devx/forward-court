"""Mine original-upload videos behind recycled-video fact-checks (demo set).

Usage: python scripts/collect_demo_samples.py [urls_file] [out_dir]

Reads article URLs (default tests/samples/video_urls.txt), keeps those
about old/recycled video that link a YouTube upload, and downloads the
oldest fitting upload per article as the demo original. Writes kept files
plus manifest.json into out_dir (default tests/samples/demo/). Kept
entries: {file, article_url, original_url, original_date, viral_claim}.
Skips: {file: null, article_url, skipped: true, skip_reason}. Same yt-dlp
limits as collect_video_samples.py (<=480p mp4, <=16MB, <=90s known
duration, no auth). Failures skip with a reason; never abort. Clips are
local testing only, never redistributed.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence.pages import extract_main_text, fetch_html
from scripts.collect_video_samples import YTDLP_TIMEOUT, download_embed
from scripts.fetch_samples import ArticleReading, read_article

DEFAULT_URLS = Path("tests/samples/video_urls.txt")
DEFAULT_OUT = Path("tests/samples/demo")
DEMO_STOP_AFTER_KEPT = 5
MAX_YOUTUBE_CANDIDATES = 4
MAX_DOWNLOAD_ATTEMPTS = 3
OLD_KEYWORDS = ("old", "recycled", "purana", "पुराना")
PAST_YEAR_RE = re.compile(r"\b(201[5-9]|202[0-5])\b")
YOUTUBE_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?[^\"'\s<>]*"
    r"|embed/|shorts/|live/|v/)|youtu\.be/)[\w?=&;%.-]+",
    re.IGNORECASE,
)


class _LinkParser(HTMLParser):
    """Collects page URLs from iframes and anchors."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        fields = dict(attrs)
        if tag == "iframe" and fields.get("src"):
            self.raw_urls.append(fields["src"])
        elif tag == "a" and fields.get("href"):
            self.raw_urls.append(fields["href"])


def extract_youtube_urls(html: str, base_url: str) -> list[str]:
    """YouTube page URLs in document order, deduplicated, capped."""
    if not html:
        return []
    parser = _LinkParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    urls: list[str] = []
    for raw in parser.raw_urls:
        for match in YOUTUBE_RE.findall(urljoin(base_url, raw)):
            cleaned = match.rstrip(").,;")
            if cleaned not in urls:
                urls.append(cleaned)
            if len(urls) >= MAX_YOUTUBE_CANDIDATES:
                return urls
    return urls


def mentions_old_video(text: str) -> bool:
    """True when the article text frames the video as old/recycled.

    Matches explicit old/recycled wording or a past-year (2015-2025)
    frame such as "2017 video shared as new".
    """
    lowered = text or ""
    if "video" not in lowered.lower():
        return False
    if any(keyword in lowered.lower() for keyword in OLD_KEYWORDS):
        return True
    return PAST_YEAR_RE.search(lowered) is not None


def probe_video(url: str) -> dict | None:
    """Info-only probe: {upload_date YYYYMMDD, duration, webpage_url}."""
    import yt_dlp  # lazy: dev-only dependency, scripts use only

    try:
        with yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "socket_timeout": YTDLP_TIMEOUT,
            }
        ) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    if not info:
        return None
    return {
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url") or url,
    }


def _to_iso(raw: str | None) -> str | None:
    if raw and re.fullmatch(r"\d{8}", raw):
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:])).isoformat()
    return None


def _slug(article_url: str, index: int) -> str:
    host = re.sub(
        r"[^a-z0-9.-]+", "", urlparse(article_url).netloc.lower()
    )
    return f"demo{index:02d}-{host or 'video'}.mp4"


def _skip(article_url: str, reason: str) -> dict:
    print(f"warning: skipping {article_url}: {reason}", file=sys.stderr)
    return {"file": None, "article_url": article_url,
            "skipped": True, "skip_reason": reason}


def fetch_demo_samples(
    urls: list[str],
    out_dir: Path,
    *,
    client=None,
    model_client=None,
    probe=None,
    downloader=None,
    stop_after: int = DEMO_STOP_AFTER_KEPT,
) -> list[dict]:
    """Download oldest fitting YouTube original per article."""
    from config import get_client

    out_dir.mkdir(parents=True, exist_ok=True)
    if model_client is None:
        model_client = get_client()
    if probe is None:
        probe = probe_video
    if downloader is None:
        downloader = download_embed
    manifest: list[dict] = []
    kept = 0
    for index, article_url in enumerate(urls):
        if kept >= stop_after:
            manifest.append(
                _skip(article_url, "unattempted: keep target reached"))
            continue
        html = fetch_html(article_url, client=client)
        if not html:
            manifest.append(_skip(article_url, "article fetch failed"))
            continue
        text = extract_main_text(html)
        if not mentions_old_video(text):
            manifest.append(_skip(article_url, "not an old/recycled video"))
            continue
        try:
            reading = read_article("", text, client=model_client)
        except Exception:
            reading = ArticleReading(verdict="Unverified", false_claim="")
        if reading.verdict != "Misleading":
            manifest.append(
                _skip(article_url,
                       f"article verdict {reading.verdict}, not Misleading"))
            continue
        candidates = extract_youtube_urls(html, article_url)
        if not candidates:
            manifest.append(_skip(article_url, "no YouTube link found"))
            continue
        probed: list[tuple[str, dict]] = []
        for candidate in candidates:
            info = probe(candidate)
            if info is not None:
                probed.append((candidate, info))
        if not probed:
            manifest.append(_skip(article_url, "all YouTube probes failed"))
            continue
        probed.sort(key=lambda pair: pair[1].get("upload_date") or "9")
        filename = _slug(article_url, index)
        chosen: tuple[str, dict] | None = None
        for _, info in probed[:MAX_DOWNLOAD_ATTEMPTS]:
            reason = downloader(info["webpage_url"], out_dir / filename)
            if reason is None:
                chosen = (info["webpage_url"], info)
                break
        if chosen is None:
            manifest.append(_skip(article_url, "no original within limits"))
            continue
        original_url, info = chosen
        manifest.append({
            "file": filename, "article_url": article_url,
            "original_url": original_url,
            "original_date": _to_iso(info.get("upload_date")),
            "viral_claim": reading.false_claim,
        })
        kept += 1
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"kept {kept} demo originals from {len(urls)} articles to {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    urls_file = Path(args[0]) if args else DEFAULT_URLS
    out_dir = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    urls = [
        line.strip()
        for line in urls_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    fetch_demo_samples(urls, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
