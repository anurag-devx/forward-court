import pytest

from evidence.urls import normalize_url, tier_of


def test_strips_tracking_params_and_fragment():
    assert (
        normalize_url("https://Example.com/a?utm_source=x&fbclid=1&id=7#frag")
        == "https://example.com/a?id=7"
    )


def test_matches_trailing_slash_and_case_variants():
    assert normalize_url("HTTPS://Example.com") == normalize_url(
        "https://example.com/"
    )


def test_drops_default_ports_and_sorts_query():
    assert normalize_url("https://example.com:443/a?b=2&a=1") == (
        "https://example.com/a?a=1&b=2"
    )


def test_returns_garbage_unchanged():
    assert normalize_url("not a url") == "not a url"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.altnews.in/check", 1),
        ("https://reuters.com/world/x", 1),
        ("https://m.bbc.co.uk/news/y", 1),
        ("https://x.com/user/status/1", 3),
        ("https://m.facebook.com/story", 3),
        ("https://kbssidhu.substack.com/p/z", 3),
        ("https://www.scribd.com/doc/1", 3),
        ("https://thedailyjagran.com/india/x", 2),
        ("https://indiatoday.in/fact-check/a", 1),
        ("https://english.factcrescendo.com/x", 1),
        ("https://m.factcrescendo.com/y", 1),
        ("not a url", 2),
    ],
)
def test_tier_of(url, expected):
    assert tier_of(url) == expected
