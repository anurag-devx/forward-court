import io
import random

import httpx
import pytest
from PIL import Image

from evidence.reverse_image import (
    ReverseMatch,
    ReverseSearchError,
    SerpApiLensProvider,
    fit_for_upload,
    matches_to_evidence,
    search_images,
)
from tests.fakes import dead_http_client

LENS_JSON = {
    "exact_matches": [
        {"link": "https://old.com/quake", "title": "2015 quake",
         "thumbnail": "https://t.com/1"},
        {"link": "https://news.com/flood", "title": "flood report",
         "thumbnail": None},
    ],
    "visual_matches": [
        {"link": "https://blog.com/similar", "title": "similar scene",
         "thumbnail": "https://t.com/2"},
        {"link": "not-a-url", "title": "junk"},
    ],
}


def _lens_client(seen, lens_json=None, upload_json=None):
    def handler(request):
        seen.append((request.method, str(request.url)))
        if request.url.path == "/image":
            return httpx.Response(200, json=upload_json or {"image_id": "abc123"})
        return httpx.Response(200, json=lens_json if lens_json is not None else LENS_JSON)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_lens_provider_uploads_then_searches():
    seen = []
    provider = SerpApiLensProvider("key", http_client=_lens_client(seen))
    matches = provider.search(b"fake-jpeg-bytes")
    assert [(m.source_url, m.exact) for m in matches] == [
        ("https://old.com/quake", True),
        ("https://news.com/flood", True),
        ("https://blog.com/similar", False),
    ]
    assert matches[0].thumbnail == "https://t.com/1"
    methods = [method for method, _ in seen]
    assert methods == ["POST", "GET"]
    get_url = seen[1][1]
    assert "engine=google_lens" in get_url
    assert "type=all" in get_url
    assert "image_id=abc123" in get_url


def test_lens_provider_raises_on_error_key():
    seen = []
    provider = SerpApiLensProvider(
        "key", http_client=_lens_client(seen, lens_json={"error": "bad key"})
    )
    with pytest.raises(ReverseSearchError, match="bad key"):
        provider.search(b"x")


def test_lens_provider_raises_on_failed_upload():
    seen = []
    provider = SerpApiLensProvider(
        "key",
        http_client=_lens_client(seen, upload_json={"error": "too big"}),
    )
    with pytest.raises(ReverseSearchError, match="upload failed"):
        provider.search(b"x")


def _noisy_jpeg(size=1200):
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    rng = random.Random(7)
    for x in range(0, size, 4):
        for y in range(0, size, 4):
            pixels[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def test_fit_for_upload_shrinks_large_images():
    big = _noisy_jpeg()
    assert len(big) > 480 * 1024
    fitted = fit_for_upload(big)
    assert len(fitted) <= 480 * 1024
    Image.open(io.BytesIO(fitted)).verify()


def test_fit_for_upload_keeps_small_images():
    small = b"x" * 100
    assert fit_for_upload(small) == small


class _SpyProvider:
    def __init__(self, by_image):
        self.by_image = by_image
        self.searched = []

    def search(self, image):
        self.searched.append(image)
        return list(self.by_image.get(image, []))


def _match(url, exact=True):
    return ReverseMatch(source_url=url, page_title="t", exact=exact)


def test_crops_searched_only_when_few_exact_matches():
    full = b"full"
    provider = _SpyProvider(
        {full: [_match("https://a.com/1"), _match("https://a.com/2")]}
    )
    matches = search_images(full, [b"crop1", b"crop2"], provider=provider)
    assert provider.searched == [full, b"crop1", b"crop2"]
    assert len(matches) == 2


def test_crops_skipped_when_enough_exact_matches():
    full = b"full"
    provider = _SpyProvider(
        {full: [_match(f"https://a.com/{i}") for i in range(3)]}
    )
    search_images(full, [b"crop1"], provider=provider)
    assert provider.searched == [full]


def test_failing_crop_search_is_skipped():
    full = b"full"

    class Flaky:
        def search(self, image):
            if image != full:
                raise ReverseSearchError("crop failed")
            return [_match("https://a.com/1")]

    matches = search_images(full, [b"crop1"], provider=Flaky())
    assert [m.source_url for m in matches] == ["https://a.com/1"]


def test_failing_full_search_raises():
    class Broken:
        def search(self, image):
            raise ReverseSearchError("down")

    with pytest.raises(ReverseSearchError):
        search_images(b"full", [b"crop1"], provider=Broken())


def test_dedup_prefers_exact_over_visual():
    full = b"full"
    provider = _SpyProvider(
        {
            full: [
                _match("https://a.com/x?utm_s=1", exact=False),
                _match("https://a.com/x", exact=True),
            ]
        }
    )
    matches = search_images(full, [], provider=provider)
    assert len(matches) == 1
    assert matches[0].exact is True


def test_matches_to_evidence_labels_visual_only():
    matches = [
        _match("https://a.com/exact"),
        ReverseMatch(source_url="https://b.com/vis", page_title="t", exact=False),
    ]
    items = matches_to_evidence(
        matches, "claim", http_client=dead_http_client()
    )
    assert [i.visual_only for i in items] == [False, True]
    assert items[0].snippet == "t"


def test_matches_to_evidence_caps_counts():
    matches = [_match(f"https://a.com/{i}") for i in range(20)]
    matches += [
        ReverseMatch(source_url=f"https://b.com/{i}", page_title="t", exact=False)
        for i in range(20)
    ]
    items = matches_to_evidence(
        matches, "claim", http_client=dead_http_client()
    )
    exact = [i for i in items if not i.visual_only]
    visual = [i for i in items if i.visual_only]
    assert len(exact) == 8
    assert len(visual) == 4
