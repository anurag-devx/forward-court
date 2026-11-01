import httpx

from scripts.collect_demo_samples import (
    extract_youtube_urls,
    fetch_demo_samples,
    mentions_old_video,
)
from scripts.fetch_samples import ArticleReading
from tests.fakes import FakeClient

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def test_extract_youtube_urls_from_iframes_and_anchors():
    html = """
    <html><body>
    <iframe src="https://www.youtube.com/embed/old111"></iframe>
    <p>See the <a href="https://www.youtube.com/watch?v=new222&t=3s">
    original</a> and <a href="https://youtu.be/new222">mirror</a>.</p>
    <a href="https://x.com/u/status/1">tweet</a>
    </body></html>
    """
    assert extract_youtube_urls(html, "https://n.com/a") == [
        "https://www.youtube.com/embed/old111",
        "https://www.youtube.com/watch?v=new222&t=3s",
        "https://youtu.be/new222",
    ]
    assert extract_youtube_urls("<html></html>", "https://n.com/a") == []


def test_mentions_old_video():
    assert mentions_old_video("This old video is viral again.")
    assert mentions_old_video("Recycled VIDEO from 2015 shared now.")
    assert mentions_old_video("यह पुराना video है।")
    assert not mentions_old_video("This video shows the event.")
    assert not mentions_old_video("Old photo viral.")
    assert mentions_old_video("A 2017 video is shared as new footage.")
    assert not mentions_old_video("A 2017 photo is shared as new.")


def _article() -> str:
    return """
    <html><head><title>Old clip viral</title></head><body>
    <p>This old video from 2015 is being shared as recent footage.</p>
    <iframe src="https://www.youtube.com/embed/new222"></iframe>
    <a href="https://www.youtube.com/watch?v=old111">original upload</a>
    </body></html>
    """


def _client() -> httpx.Client:
    def handler(request):
        return httpx.Response(
            200, text=_article(), headers={"content-type": "text/html"}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _model(verdict="Misleading") -> FakeClient:
    return FakeClient(
        parse_queue=[
            ArticleReading(verdict=verdict, false_claim="Floods hit Patna."),
        ]
    )


def test_oldest_probe_downloaded_first(tmp_path):
    attempted: list[str] = []

    def probe(url):
        return {
            "https://www.youtube.com/embed/new222": {
                "upload_date": "20241001", "duration": 30,
                "webpage_url": "https://www.youtube.com/watch?v=new222",
            },
            "https://www.youtube.com/watch?v=old111": {
                "upload_date": "20150425", "duration": 30,
                "webpage_url": "https://www.youtube.com/watch?v=old111",
            },
        }[url]

    def downloader(url, dest):
        attempted.append(url)
        dest.write_bytes(MP4)
        return None

    out = tmp_path / "demo"
    manifest = fetch_demo_samples(
        ["https://news.com/a"], out, client=_client(), model_client=_model(),
        probe=probe, downloader=downloader,
    )
    assert attempted == ["https://www.youtube.com/watch?v=old111"]
    assert manifest == [
        {
            "file": "demo00-news.com.mp4",
            "article_url": "https://news.com/a",
            "original_url": "https://www.youtube.com/watch?v=old111",
            "original_date": "2015-04-25",
            "viral_claim": "Floods hit Patna.",
        }
    ]


def test_download_falls_through_to_next_oldest(tmp_path):
    def probe(url):
        return {
            "upload_date": "20150425" if "old" in url else "20241001",
            "duration": 30,
            "webpage_url": url,
        }

    def downloader(url, dest):
        if "old" in url:
            return "video over 16MB"
        dest.write_bytes(MP4)
        return None

    out = tmp_path / "demo"
    manifest = fetch_demo_samples(
        ["https://news.com/a"], out, client=_client(), model_client=_model(),
        probe=probe, downloader=downloader,
    )
    assert manifest[0]["original_url"] == (
        "https://www.youtube.com/embed/new222"
    )


def test_non_misleading_verdict_skips_before_download(tmp_path):
    def probe(url):  # pragma: no cover
        raise AssertionError("probe must not run")

    out = tmp_path / "demo"
    manifest = fetch_demo_samples(
        ["https://news.com/a"], out, client=_client(),
        model_client=_model(verdict="False"), probe=probe,
        downloader=lambda url, dest: None,
    )
    assert manifest[0]["skip_reason"] == (
        "article verdict False, not Misleading"
    )
    assert list(out.glob("*.mp4")) == []


def test_no_youtube_link_skips(tmp_path):
    def handler(request):
        return httpx.Response(
            200, text="<html><body><p>Old video viral.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tmp_path / "demo"
    manifest = fetch_demo_samples(
        ["https://news.com/a"], out, client=client, model_client=_model(),
        probe=lambda url: None, downloader=lambda url, dest: None,
    )
    assert manifest[0]["skip_reason"] == "no YouTube link found"


def test_fits_or_trims_keeps_small_and_trims_big(tmp_path, monkeypatch):
    import shutil
    import subprocess

    import scripts.collect_demo_samples as demo

    if shutil.which("ffmpeg") is None:
        import pytest

        pytest.skip("ffmpeg missing")
    clip = tmp_path / "big.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, timeout=60, capture_output=True,
    )
    assert demo._fits_or_trims(clip) is True
    monkeypatch.setattr(demo, "MAX_VIDEO_BYTES", 10)
    assert demo._fits_or_trims(clip) is False
    assert not list(tmp_path.glob("*.trim.mp4"))
