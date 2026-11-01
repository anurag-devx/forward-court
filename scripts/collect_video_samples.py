"""Mine downloadable videos from fact-check articles (dev/test bench only).

Usage: python scripts/collect_video_samples.py [out_dir]

Phase A pages the collect_factchecks.py feeds for video-related articles
(keywords EN+HI) and writes tests/samples/video_urls.txt. Phase B scans each
article for a direct mp4 (<video>/<source>, og:video, mp4 link) or YouTube/X
embeds, prints a time estimate, then downloads in article order until
STOP_AFTER_KEPT videos are kept. Downloads: direct GET ≤16MB, or yt-dlp
(lowest ≤480p mp4, ≤16MB, ≤90s known duration, no login/cookies) for embeds.
Writes kept files plus manifest.json into out_dir (default
tests/samples/videos/). Kept entries: {file, article_url, video_url,
viral_claim, expected, claimreview, source}. Skips: {file: null, article_url,
video_url, skipped: true, skip_reason}. Failures skip with a reason; never
abort. Downloaded clips are local testing only, never redistributed.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence.pages import download_bytes, extract_claim_review, fetch_html
from scripts.collect_factchecks import FEEDS, MAX_PAGES, MAX_URLS, parse_feed
from scripts.fetch_samples import ArticleReading, read_article

DEFAULT_OUT = Path("tests/samples/videos")
URLS_OUT = Path("tests/samples/video_urls.txt")
MAX_VIDEO_BYTES = 16 * 1024 * 1024
MAX_SECONDS = 90
STOP_AFTER_KEPT = 5
# Single-file only (no ffmpeg merge): https-DASH has direct URLs while the
# throttled m3u8 manifests fail at download time, and bare `worst` can select
# undownloadable storyboards. The 8MB preference keeps downloads ahead of
# throttling (144p video-only worst case, never audio-only); the 16MB
# max_filesize + post-download check still enforce the pipeline cap.
YTDLP_FORMAT = (
    "best[ext=mp4][height<=480][protocol=https][filesize<?8M]"
    "/worstvideo[ext=mp4][protocol=https]"
    "/worstvideo[protocol=https]"
)
YTDLP_TIMEOUT = 30
YTDLP_RETRIES = 2
SECONDS_PER_EMBED_ESTIMATE = 90
KEYWORDS = ("video", "clip", "footage", "वीडियो")
STATUS_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/[^/]+/status/\d+",
    re.IGNORECASE,
)
YOUTUBE_EMBED_RE = re.compile(
    r"(?:youtube\.com/embed/|youtu\.be/)([\w-]{6,})", re.IGNORECASE
)
UNRESOLVABLE_EMBEDS = ("instagram.com", "facebook.com", "fb.watch")


class _VideoParser(HTMLParser):
    """Collects direct video URLs, YouTube/X embed URLs, other embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.embeds: list[str] = []
        self.has_other_embed = False

    def _add_embed(self, url: str) -> None:
        if url not in self.embeds:
            self.embeds.append(url)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        fields = dict(attrs)
        if tag in ("video", "source"):
            src = fields.get("src", "")
            if src:
                self.urls.append(src)
        elif tag == "meta":
            if fields.get("property") == "og:video":
                content = fields.get("content", "")
                if content:
                    self.urls.append(content)
        elif tag == "a":
            href = fields.get("href", "")
            if not href:
                return
            if urlparse(href).path.lower().endswith(".mp4"):
                self.urls.append(href)
            match = STATUS_RE.search(href)
            if match:
                self._add_embed(match.group(0))
        elif tag == "iframe":
            src = fields.get("src", "")
            match = YOUTUBE_EMBED_RE.search(src)
            if match:
                self._add_embed(
                    f"https://www.youtube.com/watch?v={match.group(1)}"
                )
            elif any(hint in src.lower() for hint in UNRESOLVABLE_EMBEDS):
                self.has_other_embed = True


def _is_youtube(url: str) -> bool:
    return "youtube.com" in url.lower()


def extract_video_candidates(
    html: str, base_url: str
) -> tuple[str | None, list[str], bool]:
    """(first direct URL, YouTube/X embed URLs, other-embed seen)."""
    if not html:
        return None, [], False
    parser = _VideoParser()
    try:
        parser.feed(html)
    except Exception:
        return None, [], False
    direct = None
    for raw in parser.urls:
        if raw.startswith("data:") or raw.startswith("blob:"):
            continue
        url = urljoin(base_url, raw)
        if url.lower().startswith(("http://", "https://")):
            direct = url
            break
    # YouTube first: logged-out downloads usually succeed, while X media
    # often stalls; stable within each kind.
    embeds = sorted(parser.embeds, key=_is_youtube, reverse=True)
    return direct, embeds, parser.has_other_embed


def is_video_related(title: str, description: str) -> bool:
    """True when title/description mentions video-like content."""
    haystack = f"{title}\n{description}".lower()
    return any(keyword in haystack for keyword in KEYWORDS)


def _page_url(feed_url: str, page: int) -> str:
    if page == 1:
        return feed_url
    separator = "&" if "?" in feed_url else "?"
    return f"{feed_url}{separator}paged={page}"


def collect_video_urls(feeds, *, client=None) -> list[str]:
    """Gather deduplicated video-related article links across feeds."""
    urls: list[str] = []
    seen: set[str] = set()
    for feed_url in feeds:
        for page in range(1, MAX_PAGES + 1):
            raw = download_bytes(_page_url(feed_url, page), client=client)
            if raw is None:
                if page == 1:
                    print(f"warning: fetch failed for {feed_url}",
                          file=sys.stderr)
                break
            try:
                items = parse_feed(raw.decode("utf-8", errors="replace"))
            except ET.ParseError:
                if page == 1:
                    print(f"warning: invalid RSS from {feed_url}",
                          file=sys.stderr)
                break
            page_links = {link for _, link, _ in items}
            if page > 1 and page_links <= seen:
                break
            for _, link, _ in items:
                seen.add(link)
            for title, link, description in items:
                if link not in urls and is_video_related(title, description):
                    urls.append(link)
                    if len(urls) >= MAX_URLS:
                        return urls
    return urls


def _is_mp4(data: bytes) -> bool:
    return len(data) > 12 and data[4:8] == b"ftyp"


def download_embed(
    url: str,
    dest: Path,
    *,
    max_seconds: int | None = MAX_SECONDS,
    format: str | None = None,
    max_bytes: int | None = MAX_VIDEO_BYTES,
) -> str | None:
    """Fetch a YouTube/X embed via yt-dlp. None on success, else a reason.

    YTDLP_FORMAT (overridable) capped at max_bytes (None skips the size
    cap) and ≤max_seconds known duration (None skips the duration check);
    unknown duration/size is allowed (the post-download check still
    applies when max_bytes is set). No login or cookies, ever.
    """
    import yt_dlp  # lazy: dev-only dependency, scripts use only

    opts = {
        "format": format or YTDLP_FORMAT,
        "outtmpl": str(dest),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": YTDLP_TIMEOUT,
        "retries": YTDLP_RETRIES,
    }
    if max_bytes is not None:
        opts["max_filesize"] = max_bytes
    if max_seconds is not None:
        opts["match_filter"] = yt_dlp.utils.match_filter_func(
            f"duration <=? {max_seconds}"
        )
    opts["socket_timeout"] = max(YTDLP_TIMEOUT, 60)
    reason: str | None = None
    for _ in range(3):  # outer retries for throttled fragment streams
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            reason = None
            break
        except Exception as exc:
            message = str(exc).strip().splitlines()
            first = message[0] if message else ""
            first = first.removeprefix("ERROR: ").strip() or "failed"
            reason = f"yt-dlp: {first[:120]}"
    if reason is not None:
        return reason
    if not dest.exists():
        return "video download produced no file"
    if max_bytes is not None and dest.stat().st_size > max_bytes:
        dest.unlink()
        return "video over 16MB"
    if not _is_mp4(dest.read_bytes()[:16]):
        dest.unlink()
        return "not an mp4 file"
    return None


def _slug(article_url: str, index: int) -> str:
    host = re.sub(r"[^a-z0-9.-]+", "", urlparse(article_url).netloc.lower())
    return f"{index:02d}-{host or 'video'}.mp4"


def _skip(article_url: str, video_url: str | None, reason: str) -> dict:
    print(f"warning: skipping {article_url}: {reason}", file=sys.stderr)
    return {"file": None, "article_url": article_url, "video_url": video_url,
            "skipped": True, "skip_reason": reason}


def _read_article_safe(html: str, model_client) -> ArticleReading:
    from evidence.pages import extract_main_text

    try:
        return read_article("", extract_main_text(html), client=model_client)
    except Exception:
        return ArticleReading(verdict="Unverified", false_claim="")


def fetch_video_samples(
    urls: list[str],
    out_dir: Path,
    *,
    client=None,
    model_client=None,
    downloader=None,
    stop_after: int = STOP_AFTER_KEPT,
) -> list[dict]:
    """Download one video per article; return manifest entries."""
    from config import get_client

    out_dir.mkdir(parents=True, exist_ok=True)
    if model_client is None:
        model_client = get_client()
    if downloader is None:
        downloader = download_embed
    # Pass 1: scan (fast page fetches, no downloads).
    scanned: list[tuple[str, str | None, str | None, list[str], bool]] = []
    for article_url in urls:
        html = fetch_html(article_url, client=client)
        if not html:
            scanned.append((article_url, None, None, [], False))
            continue
        direct, embeds, other = extract_video_candidates(html, article_url)
        scanned.append((article_url, html, direct, embeds, other))
    n_direct = sum(1 for s in scanned if s[2])
    n_embed = sum(len(s[3]) for s in scanned)
    worst_minutes = n_embed * SECONDS_PER_EMBED_ESTIMATE / 60
    print(
        f"scan: {len(urls)} articles, {n_direct} direct, "
        f"{n_embed} YouTube/X embeds; attempting in order, stopping "
        f"after {stop_after} kept (worst-case est. ~{worst_minutes:.0f} min)."
    )
    # Pass 2: download in article order with early stop.
    manifest: list[dict] = []
    kept = 0
    for index, (article_url, html, direct, embeds, other) in enumerate(scanned):
        if kept >= stop_after:
            manifest.append(
                _skip(article_url, None, "unattempted: keep target reached"))
            continue
        if html is None:
            manifest.append(_skip(article_url, None, "article fetch failed"))
            continue
        filename = _slug(article_url, index)
        video_url: str | None = None
        source = "direct"
        if direct is not None:
            data = download_bytes(
                direct, client=client, max_bytes=MAX_VIDEO_BYTES,
                content_must_contain=("video/",),
            )
            if data and _is_mp4(data):
                (out_dir / filename).write_bytes(data)
                video_url = direct
            elif data:
                manifest.append(
                    _skip(article_url, direct, "not an mp4 file"))
                continue
            else:
                manifest.append(
                    _skip(article_url, direct,
                           "video download failed or over 16MB"))
                continue
        else:
            errors: list[str] = []
            for embed_url in embeds:
                reason = downloader(embed_url, out_dir / filename)
                if reason is None:
                    video_url = embed_url
                    source = "ytdlp"
                    break
                errors.append(reason)
            if video_url is None:
                if errors:
                    manifest.append(_skip(article_url, embeds[0], errors[0]))
                elif other:
                    manifest.append(
                        _skip(article_url, None,
                               "platform embed (needs downloader)"))
                else:
                    manifest.append(
                        _skip(article_url, None, "no direct video found"))
                continue
        assert video_url is not None
        reading = _read_article_safe(html, model_client)
        review = extract_claim_review(html)
        manifest.append({
            "file": filename, "article_url": article_url,
            "video_url": video_url, "viral_claim": reading.false_claim,
            "expected": reading.verdict,
            "claimreview": (
                {"verdict": review.verdict, "rating_raw": review.rating_raw}
                if review is not None else None
            ),
            "source": source,
        })
        kept += 1
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"kept {kept} videos from {len(urls)} articles to {out_dir}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args[0]) if args else DEFAULT_OUT
    urls = collect_video_urls(FEEDS)
    URLS_OUT.parent.mkdir(parents=True, exist_ok=True)
    URLS_OUT.write_text("".join(url + "\n" for url in urls))
    print(f"collected {len(urls)} video-related urls to {URLS_OUT}")
    fetch_video_samples(urls, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
