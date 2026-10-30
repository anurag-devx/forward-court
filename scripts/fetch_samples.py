"""Download og:image files from fact-check article URLs.

Usage: python scripts/fetch_samples.py urls.txt [out_dir]

Reads article URLs (one per line, # comments ignored), downloads each
article's og:image into out_dir (default tests/samples/images/), and writes
manifest.json: [{file, article_url, og_image_url}]. Failures skip with a
warning; they never abort the run.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_client
from evidence.pages import download_bytes, fetch_html
from media.vision import describe_image

DEFAULT_OUT = Path("tests/samples/images")


class _OgImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.og_image: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "meta" and self.og_image is None:
            fields = dict(attrs)
            if fields.get("property") == "og:image" and fields.get("content"):
                self.og_image = fields["content"]


def extract_og_image(html: str, base_url: str) -> str | None:
    """First og:image URL resolved against the article URL, else None."""
    if not html:
        return None
    parser = _OgImageParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    if not parser.og_image:
        return None
    return urljoin(base_url, parser.og_image)


def _slug(article_url: str, og_image_url: str, index: int) -> str:
    host = re.sub(r"[^a-z0-9.-]+", "", urlparse(article_url).netloc.lower())
    suffix = Path(urlparse(og_image_url).path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        suffix = ".jpg"
    return f"{index:02d}-{host or 'image'}{suffix}"


def _mime_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def _skip(manifest: list[dict], article_url: str, og_image_url: str | None,
          reason: str) -> None:
    print(f"warning: skipping {article_url}: {reason}", file=sys.stderr)
    manifest.append(
        {"file": None, "article_url": article_url,
         "og_image_url": og_image_url, "skipped": True, "skip_reason": reason}
    )


def fetch_samples(
    urls: list[str], out_dir: Path, *, client=None, vision_client=None
) -> list[dict]:
    """Download og:images, screen out fact-check-marked ones, return manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if vision_client is None:
        vision_client = get_client()
    manifest: list[dict] = []
    for index, article_url in enumerate(urls):
        html = fetch_html(article_url, client=client)
        og_image_url = extract_og_image(html or "", article_url) if html else None
        if not og_image_url:
            _skip(manifest, article_url, None, "no og:image found")
            continue
        data = download_bytes(
            og_image_url, client=client, content_must_contain=("image/",)
        )
        if not data:
            _skip(manifest, article_url, og_image_url, "image download failed")
            continue
        try:
            vision = describe_image(
                data, mime=_mime_from_url(og_image_url), client=vision_client
            )
        except Exception as exc:
            _skip(manifest, article_url, og_image_url,
                   f"vision screening failed: {exc}")
            continue
        if vision.fact_check_marked:
            _skip(manifest, article_url, og_image_url,
                   "fact-check markings detected")
            continue
        filename = _slug(article_url, og_image_url, index)
        (out_dir / filename).write_bytes(data)
        manifest.append(
            {"file": filename, "article_url": article_url,
             "og_image_url": og_image_url}
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: python scripts/fetch_samples.py urls.txt [out_dir]",
              file=sys.stderr)
        return 2
    urls = [
        line.strip()
        for line in Path(args[0]).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    out_dir = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    manifest = fetch_samples(urls, out_dir)
    kept = sum(1 for entry in manifest if not entry.get("skipped"))
    print(f"kept {kept} of {len(urls)} images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
