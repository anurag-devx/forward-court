import sys

import httpx

import scripts.collect_video_samples as cvs
from scripts.collect_video_samples import (
    download_embed,
    extract_video_candidates,
    fetch_video_samples,
    is_video_related,
)
from scripts.fetch_samples import ArticleReading
from tests.fakes import FakeClient

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def test_candidates_prefer_first_direct_in_document_order():
    html = """
    <html><head>
    <meta property="og:video" content="https://cdn.com/og.mp4" />
    </head><body>
    <video src="https://cdn.com/v.mp4"></video>
    <a href="https://cdn.com/link.mp4">dl</a>
    </body></html>
    """
    direct, embeds, other = extract_video_candidates(html, "https://n.com/a")
    assert direct == "https://cdn.com/og.mp4"
    assert embeds == []
    assert other is False


def test_candidates_resolve_relative_and_skip_blobs():
    html = """
    <html><body>
    <video src="blob:https://n.com/x"></video>
    <video><source src="/rel/clip.mp4" /></video>
    </body></html>
    """
    direct, _, _ = extract_video_candidates(html, "https://n.com/a")
    assert direct == "https://n.com/rel/clip.mp4"


def test_candidates_collect_youtube_and_x_embeds_in_order():
    html = """
    <html><body>
    <blockquote class="twitter-tweet">
    <a href="https://x.com/u1/status/111?ref_src=x">t</a></blockquote>
    <iframe src="https://www.youtube.com/embed/h4IQdjjVAO8"></iframe>
    <a href="https://twitter.com/u2/status/222">t2</a>
    </body></html>
    """
    direct, embeds, other = extract_video_candidates(html, "https://n.com/a")
    assert direct is None
    assert embeds == [  # YouTube first, stable within each kind
        "https://www.youtube.com/watch?v=h4IQdjjVAO8",
        "https://x.com/u1/status/111",
        "https://twitter.com/u2/status/222",
    ]
    assert other is False


def test_candidates_flag_unresolvable_embeds_only():
    html = """
    <html><body>
    <iframe src="https://www.instagram.com/p/x/embed"></iframe>
    </body></html>
    """
    assert extract_video_candidates(html, "https://n.com/a") == (
        None, [], True)
    assert extract_video_candidates("<html></html>", "https://n.com/a") == (
        None, [], False)


def test_is_video_related_english_and_hindi():
    assert is_video_related("Viral video debunked", "")
    assert is_video_related("", "watch this clip and footage")
    assert is_video_related("वायरल वीडियो की सच्चाई", "")
    assert not is_video_related("Old photo viral", "a still image")


def _article(video_tag: str, claim_review: bool = True) -> str:
    review = """
    <script type="application/ld+json">
    {"@type": "ClaimReview", "claimReviewed": "Floods hit Patna today",
     "reviewRating": {"@type": "Rating", "alternateName": "False"}}
    </script>
    """ if claim_review else ""
    return f"""
    <html><head><title>Clip viral</title></head><body>
    {video_tag}{review}
    <p>The viral clip claims floods hit Patna today. It is old footage.</p>
    </body></html>
    """


def _client(article_html: str, video_body: bytes = MP4) -> httpx.Client:
    def handler(request):
        url = str(request.url)
        if url == "https://news.com/a":
            return httpx.Response(
                200, text=article_html,
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200, content=video_body,
            headers={"content-type": "video/mp4"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _multi_client(pages: dict[str, str]) -> httpx.Client:
    def handler(request):
        url = str(request.url)
        if url in pages:
            return httpx.Response(
                200, text=pages[url],
                headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200, content=MP4, headers={"content-type": "video/mp4"}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _model() -> FakeClient:
    return FakeClient(
        parse_queue=[
            ArticleReading(verdict="False", false_claim="Floods hit Patna."),
        ]
    )


def test_fetch_direct_happy_path(tmp_path):
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(_article('<video src="/v/clip.mp4"></video>')),
        model_client=_model(),
    )
    assert manifest == [
        {
            "file": "00-news.com.mp4",
            "article_url": "https://news.com/a",
            "video_url": "https://news.com/v/clip.mp4",
            "viral_claim": "Floods hit Patna.",
            "expected": "False",
            "claimreview": {"verdict": "False", "rating_raw": "False"},
            "source": "direct",
        }
    ]
    assert (out / "00-news.com.mp4").read_bytes() == MP4
    assert (out / "manifest.json").exists()


def test_embed_falls_back_to_downloader(tmp_path):
    out = tmp_path / "videos"
    seen: list[tuple[str, object]] = []

    def downloader(url, dest):
        seen.append((url, dest))
        dest.write_bytes(MP4)
        return None

    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(_article(
            '<iframe src="https://www.youtube.com/embed/abc1234"></iframe>'
        )),
        model_client=_model(),
        downloader=downloader,
    )
    assert seen[0][0] == "https://www.youtube.com/watch?v=abc1234"
    assert manifest[0]["source"] == "ytdlp"
    assert manifest[0]["video_url"] == "https://www.youtube.com/watch?v=abc1234"


def test_embed_errors_record_first_reason(tmp_path):
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(_article(
            '<a href="https://x.com/u/status/111">t</a>'
        )),
        model_client=_model(),
        downloader=lambda url, dest: "yt-dlp: Login required",
    )
    assert manifest[0]["skipped"] is True
    assert manifest[0]["skip_reason"] == "yt-dlp: Login required"


def test_other_embeds_keep_skip_reason(tmp_path):
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(
            _article('<iframe src="https://www.instagram.com/p/x/"></iframe>')
        ),
        model_client=_model(),
    )
    assert manifest[0]["skip_reason"] == "platform embed (needs downloader)"


def test_early_stop_after_target(tmp_path):
    pages = {
        f"https://news.com/a{i}": _article(
            f'<video src="/v/{i}.mp4"></video>', claim_review=False
        )
        for i in range(4)
    }
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        list(pages), out, client=_multi_client(pages),
        model_client=FakeClient(
            parse_queue=[
                ArticleReading(verdict="False", false_claim=f"c{i}")
                for i in range(2)
            ]
        ),
        stop_after=2,
    )
    kept = [e for e in manifest if not e.get("skipped")]
    assert len(kept) == 2
    assert manifest[2]["skip_reason"] == "unattempted: keep target reached"
    assert manifest[3]["skip_reason"] == "unattempted: keep target reached"


def test_no_direct_video_skips(tmp_path):
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(_article("<p>no media here</p>")),
        model_client=_model(),
    )
    assert manifest[0]["skip_reason"] == "no direct video found"


def test_over_cap_video_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(cvs, "MAX_VIDEO_BYTES", 10)
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(_article('<video src="/v/clip.mp4"></video>')),
        model_client=_model(),
    )
    assert manifest[0]["skip_reason"] == "video download failed or over 16MB"


def test_non_mp4_body_skips(tmp_path):
    out = tmp_path / "videos"
    manifest = fetch_video_samples(
        ["https://news.com/a"], out,
        client=_client(
            _article('<video src="/v/clip.mp4"></video>'),
            video_body=b"not a video file" * 8,
        ),
        model_client=_model(),
    )
    assert manifest[0]["skip_reason"] == "not an mp4 file"


class _StubYoutubeDL:
    seen_opts: dict = {}

    def __init__(self, opts):
        type(self).seen_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        assert download is True


class _StubUtils:
    @staticmethod
    def match_filter_func(spec):
        assert spec == "duration <=? 90"
        return ("filter", spec)


def test_download_embed_options_and_auth_free(tmp_path, monkeypatch):
    import types

    stub = types.ModuleType("yt_dlp")
    stub.YoutubeDL = _StubYoutubeDL
    stub.utils = _StubUtils
    monkeypatch.setitem(sys.modules, "yt_dlp", stub)
    dest = tmp_path / "clip.mp4"
    dest.write_bytes(MP4)
    assert download_embed("https://www.youtube.com/watch?v=x", dest) is None
    opts = _StubYoutubeDL.seen_opts
    assert opts["format"] == cvs.YTDLP_FORMAT
    assert "480" in opts["format"]
    assert opts["max_filesize"] == cvs.MAX_VIDEO_BYTES
    assert opts["match_filter"] == ("filter", "duration <=? 90")
    assert opts["noplaylist"] is True
    assert "cookiefile" not in opts and "username" not in opts


def test_download_embed_no_duration_cap_omits_filter(tmp_path, monkeypatch):
    import types

    stub = types.ModuleType("yt_dlp")
    stub.YoutubeDL = _StubYoutubeDL
    stub.utils = _StubUtils
    monkeypatch.setitem(sys.modules, "yt_dlp", stub)
    dest = tmp_path / "clip.mp4"
    dest.write_bytes(MP4)
    assert download_embed("https://www.youtube.com/watch?v=x", dest,
                          max_seconds=None) is None
    assert "match_filter" not in _StubYoutubeDL.seen_opts


def test_download_embed_maps_errors(tmp_path, monkeypatch):
    import types

    class Failing:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise RuntimeError("Login required\nsecond line")

    stub = types.ModuleType("yt_dlp")
    stub.YoutubeDL = Failing
    stub.utils = _StubUtils
    monkeypatch.setitem(sys.modules, "yt_dlp", stub)
    reason = download_embed("https://x.com/u/status/1", tmp_path / "c.mp4")
    assert reason == "yt-dlp: Login required"
