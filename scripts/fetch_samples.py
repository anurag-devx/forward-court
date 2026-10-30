"""Mine viral-claim images from fact-check articles.

Usage: python scripts/fetch_samples.py urls.txt [out_dir]

Extracts body <img> URLs from each article, drops small/logo images,
classifies the rest with vision, and keeps viral_claim images only. Writes
kept files plus manifest.json into out_dir (default tests/samples/images/).
Manifest entries: kept {file, article_url, image_url, role, expected} or
skipped {file: null, article_url, image_url, skipped: true, skip_reason}.
Failures skip with a reason; they never abort the run.
"""
from __future__ import annotations

import io
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from pydantic import BaseModel

from config import agent_model, get_client
from evidence.pages import download_bytes, extract_main_text, fetch_html
from media.vision import describe_image

DEFAULT_OUT = Path("tests/samples/images")
MIN_DIMENSION = 300
MAX_IMAGES_PER_ARTICLE = 6
SKIP_HINTS = ("logo", "icon", "sprite", "avatar")


class ArticleVerdict(BaseModel):
    verdict: Literal["False", "Misleading", "AI-generated", "Unverified"]


CLASSIFY_INSTRUCTIONS = (
    "Read this fact-check article and report its conclusion about the "
    "viral claim: 'False' when the claim is debunked as untrue, "
    "'Misleading' when real content is shown with wrong context or as "
    "recycled old news, 'AI-generated' when the article says the image or "
    "video was AI-created or synthetic, 'Unverified' when unclear."
)


def classify_article(title: str, text: str, *, client) -> str:
    """Classify the article's own conclusion about the viral claim."""
    response = client.responses.parse(
        model=agent_model(),
        instructions=CLASSIFY_INSTRUCTIONS,
        input=f"{title}\n\n{text[:4000]}",
        text_format=ArticleVerdict,
    )
    return response.output_parsed.verdict


class _ImgParser(HTMLParser):
    """Collects (src, srcset, width, height) for every <img>."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[tuple[str, str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "img":
            fields = dict(attrs)
            self.images.append(
                (
                    fields.get("src", ""),
                    fields.get("srcset", ""),
                    fields.get("width", ""),
                    fields.get("height", ""),
                )
            )


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "title":
            self._in_title = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False


def _srcset_largest(srcset: str) -> str:
    candidates = [entry.strip().split() for entry in srcset.split(",")]
    candidates = [parts[0] for parts in candidates if parts]
    return candidates[-1] if candidates else ""


def _looks_small(width: str, height: str) -> bool:
    try:
        return int(width) < MIN_DIMENSION and int(height) < MIN_DIMENSION
    except ValueError:
        return False


def extract_body_images(html: str, base_url: str) -> list[str]:
    """Body image URLs: resolved, logo/icon pre-filtered, capped."""
    if not html:
        return []
    parser = _ImgParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    urls: list[str] = []
    for src, srcset, width, height in parser.images:
        raw = src or _srcset_largest(srcset)
        if not raw or raw.startswith("data:"):
            continue
        url = urljoin(base_url, raw)
        path = urlparse(url).path.lower()
        if path.endswith(".svg"):
            continue
        if any(hint in path for hint in SKIP_HINTS):
            continue
        if _looks_small(width, height):
            continue
        if url not in urls:
            urls.append(url)
        if len(urls) >= MAX_IMAGES_PER_ARTICLE:
            break
    return urls


def _is_small(data: bytes) -> bool | None:
    """True if both dimensions <300px; None when unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception:
        return None
    return width < MIN_DIMENSION and height < MIN_DIMENSION


def _mime_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def _slug(article_url: str, image_url: str, index: int, img_index: int) -> str:
    host = re.sub(r"[^a-z0-9.-]+", "", urlparse(article_url).netloc.lower())
    suffix = Path(urlparse(image_url).path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        suffix = ".jpg"
    return f"{index:02d}-{img_index}-{host or 'image'}{suffix}"


def _skip(manifest: list[dict], article_url: str, image_url: str | None,
          reason: str) -> None:
    manifest.append(
        {"file": None, "article_url": article_url, "image_url": image_url,
         "skipped": True, "skip_reason": reason}
    )


def fetch_samples(
    urls: list[str], out_dir: Path, *, client=None, vision_client=None
) -> list[dict]:
    """Mine viral-claim images; return manifest entries (kept + skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if vision_client is None:
        vision_client = get_client()
    manifest: list[dict] = []
    for index, article_url in enumerate(urls):
        html = fetch_html(article_url, client=client)
        if not html:
            _skip(manifest, article_url, None, "article fetch failed")
            continue
        images = extract_body_images(html, article_url)
        if not images:
            _skip(manifest, article_url, None, "no body images found")
            continue
        expected: str | None = None
        for img_index, image_url in enumerate(images):
            data = download_bytes(
                image_url, client=client, content_must_contain=("image/",)
            )
            if not data:
                _skip(manifest, article_url, image_url, "image download failed")
                continue
            small = _is_small(data)
            if small is None:
                _skip(manifest, article_url, image_url, "unreadable image")
                continue
            if small:
                _skip(manifest, article_url, image_url, "too small (<300px)")
                continue
            try:
                vision = describe_image(
                    data, mime=_mime_from_url(image_url), client=vision_client
                )
            except Exception as exc:
                _skip(manifest, article_url, image_url,
                       f"vision screening failed: {exc}")
                continue
            if vision.image_role != "viral_claim":
                _skip(manifest, article_url, image_url,
                       f"role: {vision.image_role}")
                continue
            if expected is None:
                title_parser = _TitleParser()
                try:
                    title_parser.feed(html)
                except Exception:
                    pass
                try:
                    expected = classify_article(
                        title_parser.title, extract_main_text(html),
                        client=vision_client,
                    )
                except Exception:
                    expected = "Unverified"
            filename = _slug(article_url, image_url, index, img_index)
            (out_dir / filename).write_bytes(data)
            manifest.append(
                {"file": filename, "article_url": article_url,
                 "image_url": image_url, "role": "viral_claim",
                 "expected": expected}
            )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    kept = sum(1 for entry in manifest if not entry.get("skipped"))
    print(f"kept {kept} images from {len(urls)} articles to {out_dir}")
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
    fetch_samples(urls, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
