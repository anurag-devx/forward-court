import json

import httpx

from media.models import VisionResult
from scripts.fetch_samples import extract_og_image, fetch_samples
from tests.fakes import FakeClient


def _vision_client(marked):
    return FakeClient(
        parsed_by_model={
            VisionResult: VisionResult(
                scene_description="s",
                visible_text="",
                people_places_objects=[],
                claims_from_image=[],
                language="en",
                fact_check_marked=marked,
            )
        }
    )


def test_extract_og_image_resolves_relative():
    html = '<meta property="og:image" content="/img/a.jpg" />'
    assert extract_og_image(html, "https://news.com/article") == (
        "https://news.com/img/a.jpg"
    )
    assert extract_og_image("<html>none</html>", "https://x.com") is None


def test_fetch_samples_writes_files_and_manifest(tmp_path):
    articles = {
        "https://news.com/a": (
            '<meta property="og:image" content="https://img.com/a.png" />'
        ),
        "https://blog.com/b": "<html>no image here</html>",
    }

    def handler(request):
        url = str(request.url)
        if url in articles:
            return httpx.Response(
                200, text=articles[url], headers={"content-type": "text/html"}
            )
        if url == "https://img.com/a.png":
            return httpx.Response(
                200, content=b"png-bytes", headers={"content-type": "image/png"}
            )
        return httpx.Response(404, text="gone")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tmp_path / "images"
    manifest = fetch_samples(
        ["https://news.com/a", "https://blog.com/b"],
        out,
        client=client,
        vision_client=_vision_client(False),
    )
    assert manifest == [
        {
            "file": "00-news.com.png",
            "article_url": "https://news.com/a",
            "og_image_url": "https://img.com/a.png",
        },
        {
            "file": None,
            "article_url": "https://blog.com/b",
            "og_image_url": None,
            "skipped": True,
            "skip_reason": "no og:image found",
        },
    ]
    assert (out / "00-news.com.png").read_bytes() == b"png-bytes"
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_fetch_samples_skips_fact_check_marked(tmp_path):
    def handler(request):
        url = str(request.url)
        if url == "https://news.com/a":
            return httpx.Response(
                200,
                text='<meta property="og:image" content="https://img.com/a.jpg" />',
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200, content=b"jpg-bytes", headers={"content-type": "image/jpeg"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tmp_path / "images"
    manifest = fetch_samples(
        ["https://news.com/a"],
        out,
        client=client,
        vision_client=_vision_client(True),
    )
    assert manifest[0]["skipped"] is True
    assert manifest[0]["skip_reason"] == "fact-check markings detected"
    assert list(out.iterdir()) == [out / "manifest.json"]
