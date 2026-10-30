import httpx

from scripts.collect_factchecks import collect, is_image_related, main, parse_feed

FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Old photo viral as recent flood</title>
<link>https://x.com/a</link><description>check</description></item>
<item><title>Unrelated claim about taxes</title>
<link>https://x.com/b</link><description>no media words</description></item>
<item><title>Video shows speech</title>
<link>https://x.com/c</link><description>a viral image attached</description></item>
<item><title>Old photo viral as recent flood</title>
<link>https://x.com/a</link><description>dup</description></item>
</channel></rss>"""


def test_parse_feed_reads_items():
    items = parse_feed(FEED_XML)
    assert len(items) == 4
    assert items[0][1] == "https://x.com/a"


def test_keyword_filter_matches_spec_words():
    assert is_image_related("Old photo viral", "") is True
    assert is_image_related("", "picture from 2015") is True
    assert is_image_related("taxes explained", "no media") is False


def test_keyword_filter_matches_hindi():
    assert is_image_related("वायरल तस्वीर की सच्चाई", "") is True
    assert is_image_related("", "फोटो पुरानी है") is True
    assert is_image_related("मौसम का हाल", "बारिश होगी") is False


def _client(routes):
    def handler(request):
        body, status = routes.get(str(request.url), ("", 404))
        return httpx.Response(status, content=body.encode())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_collect_filters_dedupes_and_caps():
    client = _client({"https://f.com/rss": (FEED_XML, 200)})
    assert collect(["https://f.com/rss"], client=client) == [
        "https://x.com/a",
        "https://x.com/c",
    ]
    assert collect(["https://f.com/rss"], client=client, max_urls=1) == [
        "https://x.com/a"
    ]


def _feed(*links):
    items = "".join(
        f"<item><title>old photo {i}</title><link>{link}</link>"
        "<description>x</description></item>"
        for i, link in enumerate(links)
    )
    return f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>'


def test_collect_pages_back_until_cap():
    client = _client(
        {
            "https://f.com/rss": (_feed("https://x.com/1"), 200),
            "https://f.com/rss?paged=2": (_feed("https://x.com/2"), 200),
        }
    )
    assert collect(["https://f.com/rss"], client=client) == [
        "https://x.com/1",
        "https://x.com/2",
    ]


def test_collect_stops_when_page_repeats():
    feed = _feed("https://x.com/1")
    client = _client(
        {
            "https://f.com/rss": (feed, 200),
            "https://f.com/rss?paged=2": (feed, 200),
        }
    )
    assert collect(["https://f.com/rss"], client=client) == ["https://x.com/1"]


def test_recycled_keywords_match():
    assert is_image_related("recycled clips viral again", "") is True
    assert is_image_related("10-year-old boy missing", "") is True


def test_collect_skips_failed_feeds(capsys):
    client = _client({"https://good.com/rss": (FEED_XML, 200)})
    urls = collect(["https://bad.com/rss", "https://good.com/rss"], client=client)
    assert urls == ["https://x.com/a", "https://x.com/c"]
    assert "warning" in capsys.readouterr().err


def test_collect_skips_invalid_xml(capsys):
    client = _client({"https://f.com/rss": ("not xml", 200)})
    assert collect(["https://f.com/rss"], client=client) == []
    assert "invalid RSS" in capsys.readouterr().err


def test_main_writes_url_file(tmp_path, monkeypatch):
    out = tmp_path / "urls.txt"
    monkeypatch.setattr(
        "scripts.collect_factchecks.collect",
        lambda feeds, **kwargs: ["https://x.com/a"],
    )
    assert main([str(out)]) == 0
    assert out.read_text() == "https://x.com/a\n"
