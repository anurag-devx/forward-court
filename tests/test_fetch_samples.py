import json

import httpx

from scripts.fetch_samples import extract_og_image, fetch_samples


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
        ["https://news.com/a", "https://blog.com/b"], out, client=client
    )
    assert manifest == [
        {
            "file": "00-news.com.png",
            "article_url": "https://news.com/a",
            "og_image_url": "https://img.com/a.png",
        }
    ]
    assert (out / "00-news.com.png").read_bytes() == b"png-bytes"
    assert json.loads((out / "manifest.json").read_text()) == manifest
