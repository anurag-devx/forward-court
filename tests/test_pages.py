import httpx
import pytest

from evidence import pages
from evidence.pages import (
    download_bytes,
    extract_claim_review,
    extract_main_text,
    extract_publish_date,
    fetch_html,
    make_excerpt,
)

ARTICLE_HTML = """<html><head>
<meta property="article:published_time" content="2026-04-10T09:30:00+05:30" />
<script type="application/ld+json">
{"@type": "NewsArticle", "datePublished": "2020-01-01"}
</script>
</head><body><article>
<p>First paragraph of the article with enough words to pass extraction.</p>
<p>Second paragraph continues the story with further factual details here.</p>
<p>Third paragraph adds background and quotes from named official sources.</p>
<p>Fourth paragraph wraps up the report with context and next steps.</p>
</article></body></html>"""


def test_meta_tag_beats_json_ld():
    assert extract_publish_date(ARTICLE_HTML) == "2026-04-10"


def test_json_ld_object_graph_and_list():
    assert (
        extract_publish_date(
            '<script type="application/ld+json">'
            '{"@graph": [{"@type": "Article", "datePublished": "2019-05-06"}]}'
            "</script>"
        )
        == "2019-05-06"
    )
    assert (
        extract_publish_date(
            '<script type="application/ld+json">'
            '[{"datePublished": "2018-02-03T10:00:00Z"}]'
            "</script>"
        )
        == "2018-02-03"
    )


def test_invalid_meta_falls_through_to_json_ld():
    html = (
        '<meta property="article:published_time" content="not a date" />'
        '<script type="application/ld+json">{"datePublished": "2017-07-07"}</script>'
    )
    assert extract_publish_date(html) == "2017-07-07"


def test_no_date_signals_returns_none():
    assert extract_publish_date("<html><body>no dates</body></html>") is None
    assert extract_publish_date("") is None


def test_main_text_and_excerpt():
    text = extract_main_text(ARTICLE_HTML)
    assert "First paragraph" in text
    excerpt = make_excerpt(text)
    assert len(excerpt) <= 500
    assert "  " not in excerpt


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_sends_browser_ua_and_returns_html():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="<html>ok</html>",
                              headers={"content-type": "text/html"})

    assert fetch_html("https://a.com/x", client=_client(handler)) == "<html>ok</html>"
    assert seen["ua"].startswith("Mozilla/5.0")


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(404, text="gone"),
        lambda request: httpx.Response(
            200, content=b"x" * (pages.MAX_BYTES + 1),
            headers={"content-type": "text/html"},
        ),
        lambda request: httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        ),
    ],
)
def test_fetch_failures_return_none(handler):
    assert fetch_html("https://a.com/x", client=_client(handler)) is None


def test_fetch_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("down")

    assert fetch_html("https://a.com/x", client=_client(handler)) is None


def test_fetch_rejects_non_http():
    assert fetch_html("file:///etc/passwd") is None


def test_download_bytes_for_images():
    def handler(request):
        return httpx.Response(
            200, content=b"img-bytes", headers={"content-type": "image/jpeg"}
        )

    data = download_bytes(
        "https://a.com/x.jpg",
        client=_client(handler),
        content_must_contain=("image/",),
    )
    assert data == b"img-bytes"


def test_download_bytes_rejects_wrong_type_and_oversize():
    def html_handler(request):
        return httpx.Response(
            200, text="<html>", headers={"content-type": "text/html"}
        )

    assert (
        download_bytes(
            "https://a.com/x.jpg",
            client=_client(html_handler),
            content_must_contain=("image/",),
        )
        is None
    )

    def big_handler(request):
        return httpx.Response(
            200, content=b"x" * 100, headers={"content-type": "image/png"}
        )

    assert (
        download_bytes("https://a.com/x.png", client=_client(big_handler), max_bytes=10)
        is None
    )


def _claim_review_html(node):
    import json

    return (
        "<html><head><script type=\"application/ld+json\">"
        + json.dumps(node)
        + "</script></head><body><p>text</p></body></html>"
    )


def _review_node(rating, reviewed="Vaccines contain microchips"):
    return {
        "@type": "ClaimReview",
        "claimReviewed": reviewed,
        "reviewRating": rating,
    }


def test_claim_review_alternate_name_mapping():
    html = _claim_review_html(_review_node({"alternateName": "False"}))
    review = extract_claim_review(html)
    assert review.verdict == "False"
    assert review.rating_raw == "False"
    assert review.claim_text == "Vaccines contain microchips"


@pytest.mark.parametrize(
    "raw,verdict",
    [
        ("fake", "False"),
        ("Partly False", "Misleading"),
        ("missing context", "Misleading"),
        ("TRUE", "True"),
        ("correct", "True"),
    ],
)
def test_claim_review_alternate_name_variants(raw, verdict):
    html = _claim_review_html(_review_node({"alternateName": raw}))
    assert extract_claim_review(html).verdict == verdict


def test_claim_review_inside_graph_and_list():
    html = _claim_review_html(
        {"@graph": [_review_node({"alternateName": "Misleading"})]}
    )
    assert extract_claim_review(html).verdict == "Misleading"
    html = _claim_review_html([_review_node({"alternateName": "True"})])
    assert extract_claim_review(html).verdict == "True"


def test_claim_review_numeric_uses_best_worst_scale():
    html = _claim_review_html(
        _review_node(
            {"ratingValue": "1", "bestRating": "5", "worstRating": "1"}
        )
    )
    assert extract_claim_review(html).verdict == "False"
    html = _claim_review_html(
        _review_node(
            {"ratingValue": 5, "bestRating": 5, "worstRating": 1}
        )
    )
    assert extract_claim_review(html).verdict == "True"
    html = _claim_review_html(
        _review_node(
            {"ratingValue": "3", "bestRating": "5", "worstRating": "1"}
        )
    )
    review = extract_claim_review(html)
    assert review.verdict == "Unverified"
    assert review.rating_raw == "3"


def test_claim_review_numeric_defaults_to_schema_scale():
    html = _claim_review_html(_review_node({"ratingValue": "1"}))
    assert extract_claim_review(html).verdict == "False"


def test_claim_review_unmapped_name_falls_back_to_numeric():
    html = _claim_review_html(
        _review_node({"alternateName": "Mixture", "ratingValue": "1"})
    )
    assert extract_claim_review(html).verdict == "False"


def test_claim_review_absent_or_broken_returns_none():
    assert extract_claim_review("<html><body>no json</body></html>") is None
    assert extract_claim_review("") is None
    html = _claim_review_html({"@type": "ClaimReview"})
    assert extract_claim_review(html) is None
    html = _claim_review_html(_review_node({"alternateName": "Mixture"}))
    assert extract_claim_review(html) is None
