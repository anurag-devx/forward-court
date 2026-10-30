import io
import json

import httpx
from PIL import Image

from media.models import VisionResult
from scripts.fetch_samples import (
    ArticleReading,
    extract_body_images,
    fetch_samples,
    read_article,
)
from tests.fakes import FakeClient


def _jpeg(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def _vision(role):
    return VisionResult(
        scene_description="s",
        visible_text="",
        people_places_objects=[],
        claims_from_image=[],
        language="en",
        image_role=role,
    )


def test_extract_body_images_filters_and_resolves():
    html = """
    <img src="/photos/big.jpg" />
    <img srcset="/photos/s-200.jpg 200w, /photos/s-800.jpg 800w" />
    <img src="data:image/png;base64,xx" />
    <img src="/img/v.svg" />
    <img src="/static/logo.png" />
    <img src="/photos/thumb.jpg" width="100" height="100" />
    """
    assert extract_body_images(html, "https://news.com/a") == [
        "https://news.com/photos/big.jpg",
        "https://news.com/photos/s-800.jpg",
    ]


def test_extract_body_images_caps_per_article():
    html = "".join(f'<img src="/p/{i}.jpg" />' for i in range(10))
    assert len(extract_body_images(html, "https://x.com")) == 6


def test_read_article_returns_verdict_and_claim():
    fake = FakeClient(
        parsed_by_model={
            ArticleReading: ArticleReading(
                verdict="Misleading", false_claim="Old flood photo as today."
            )
        }
    )
    reading = read_article("t", "body", client=fake)
    assert (reading.verdict, reading.false_claim) == (
        "Misleading", "Old flood photo as today.",
    )


def test_fetch_samples_keeps_viral_and_original_with_claim(tmp_path):
    article = """
    <html><head><title>Old photo viral</title></head><body>
    <img src="/photos/claim.jpg" />
    <img src="/photos/original.jpg" />
    <img src="/photos/stamp.jpg" />
    <img src="/photos/tiny.jpg" />
    </body></html>
    """

    def handler(request):
        url = str(request.url)
        if url == "https://news.com/a":
            return httpx.Response(
                200, text=article, headers={"content-type": "text/html"}
            )
        if url == "https://news.com/photos/tiny.jpg":
            body = _jpeg(10, 10)
        else:
            body = _jpeg(300, 300)
        return httpx.Response(
            200, content=body, headers={"content-type": "image/jpeg"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    vision_client = FakeClient(
        parse_queue=[
            _vision("viral_claim"),
            _vision("original_source"),
            _vision("fact_check_marked"),
            ArticleReading(verdict="False", false_claim="Flood today."),
        ]
    )
    out = tmp_path / "images"
    manifest = fetch_samples(
        ["https://news.com/a"], out, client=client, vision_client=vision_client
    )
    kept = [entry for entry in manifest if not entry.get("skipped")]
    skipped = [entry for entry in manifest if entry.get("skipped")]
    assert kept == [
        {
            "file": "00-0-news.com.jpg",
            "article_url": "https://news.com/a",
            "image_url": "https://news.com/photos/claim.jpg",
            "role": "viral_claim",
            "expected": "False",
            "claim": "Flood today.",
            "ground_truth": ["https://news.com/photos/original.jpg"],
        },
        {
            "file": "00-1-news.com.jpg",
            "article_url": "https://news.com/a",
            "image_url": "https://news.com/photos/original.jpg",
            "role": "original_source",
        },
    ]
    assert [entry["skip_reason"] for entry in skipped] == [
        "role: fact_check_marked",
        "too small (<300px)",
    ]
    assert (out / "00-0-news.com.jpg").exists()
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_fetch_samples_skips_failed_articles(tmp_path):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, text="gone")
        )
    )
    out = tmp_path / "images"
    manifest = fetch_samples(
        ["https://news.com/a"], out, client=client,
        vision_client=FakeClient(),
    )
    assert manifest[0]["skip_reason"] == "article fetch failed"
