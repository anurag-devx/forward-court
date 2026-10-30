from types import SimpleNamespace

import pytest

from config import UsageTracker
from evidence.factcheck import (
    extract_entities,
    is_trusted_factcheck,
    lookup_factchecks,
    lookup_queries,
    trusted_hosts,
)
from tests.fakes import FakeClient, dead_http_client, search_call


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.altnews.in/some-check", True),
        ("https://boomlive.in/x", True),
        ("https://nepal.newschecker.co/y", False),  # wrong TLD
        ("https://newschecker.in/z", True),
        ("https://indiatoday.in/fact-check/a", True),
        ("https://indiatoday.in/sports/b", False),  # outside scoped path
        ("https://thequint.com/news/webqoof/c", True),
        ("https://thequint.com/entertainment/d", False),
        ("https://english.factcrescendo.com/2026/10/12/x", True),
        ("https://factcrescendo.com/x", True),
        ("https://randomblog.com/fact-check", False),
        ("not a url", False),
    ],
)
def test_trusted_domain_matching(url, expected):
    assert is_trusted_factcheck(url) == expected


def test_lookup_keeps_trusted_only_and_tags():
    client = FakeClient(
        outputs=[
            [
                search_call(
                    [
                        ("https://www.altnews.in/check", "T1", "debunked 2026-01-02"),
                        ("https://evil.com/fake", "T2", "junk"),
                        ("https://boomlive.in/check", "T3", "false"),
                    ]
                )
            ]
        ]
    )
    tracker = UsageTracker()
    items = lookup_factchecks(
        ["claim one"], client=client, tracker=tracker,
        http_client=dead_http_client(),
    )
    assert [i.source_url for i in items] == [
        "https://www.altnews.in/check",
        "https://boomlive.in/check",
    ]
    assert all(i.fact_check for i in items)
    assert items[0].date == "2026-01-02"
    assert "factcheck:claim1" in tracker.breakdown()[0]
    assert "fact check" in client.responses.calls[0][1]["input"].lower()


def test_lookup_caps_matches_per_claim():
    client = FakeClient(
        outputs=[
            [
                search_call(
                    [(f"https://a{i}.altnews.in/x", "T", "s") for i in range(6)]
                )
            ]
        ]
    )
    items = lookup_factchecks(
        ["c"], client=client, http_client=dead_http_client()
    )
    assert len(items) == 3


def test_lookup_restricts_query_to_trusted_sites():
    client = FakeClient(outputs=[[search_call([])]])
    lookup_factchecks(["c"], client=client, http_client=dead_http_client())
    call = client.responses.calls[0][1]
    assert "filters" not in call["tools"][0]
    sent = call["input"]
    assert "site:altnews.in" in sent
    assert "site:boomlive.in" in sent
    assert "site:factcrescendo.com" in sent
    assert "site:english.factcrescendo.com" in sent
    assert " OR site:" in sent


def test_lookup_appends_entities_to_query():
    client = FakeClient(outputs=[[search_call([])]])
    lookup_factchecks(
        ['Mayor "Jane Rao" resigned in Patna'], client=client,
        http_client=dead_http_client(),
    )
    sent = client.responses.calls[0][1]["input"]
    assert "Jane Rao" in sent and "Patna" in sent


def test_non_english_claim_adds_english_entity_query():
    client = FakeClient(outputs=[[search_call([])], [search_call([])]])
    tracker = UsageTracker()
    lookup_factchecks(
        ["कल Patna में बाढ़ आई"], client=client, tracker=tracker,
        http_client=dead_http_client(), language="hi",
    )
    inputs = [kwargs["input"] for _, kwargs in client.responses.calls]
    assert len(inputs) == 2
    assert inputs[1].startswith("fact check: Patna site:")
    assert any("factcheck:claim1:en" in line for line in tracker.breakdown())


def test_non_english_claim_without_entities_runs_once():
    client = FakeClient(outputs=[[search_call([])]])
    lookup_factchecks(
        ["कल बाढ़ आई"], client=client, http_client=dead_http_client(),
        language="hi",
    )
    assert len(client.responses.calls) == 1


def test_extract_entities():
    assert extract_entities('Mayor "Jane Rao" met PM Modi in Patna') == [
        "Jane Rao",
        "Mayor",
        "PM Modi",
        "Patna",
    ]
    assert extract_entities("nothing capitalized here") == []


def test_lookup_queries_shape():
    [(suffix, query)] = lookup_queries("plain claim", "en")
    assert suffix == ""
    assert query.startswith("fact check: plain claim site:")
    assert "site:thequint.com" in query
    assert len(lookup_queries("x", "hi")) == 1


def test_lookup_never_raises():
    class Exploding:
        def create(self, **kwargs):
            raise ConnectionError("down")

    client = SimpleNamespace(responses=Exploding())
    assert lookup_factchecks(["c"], client=client) == []
    no_search = FakeClient(
        outputs=[[SimpleNamespace(type="message", status="completed")]]
    )
    assert lookup_factchecks(["c"], client=no_search) == []
