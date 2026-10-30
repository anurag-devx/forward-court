from types import SimpleNamespace

import httpx
import pytest

from config import UsageTracker
from evidence.grounder import GroundingError, gather_evidence
from tests.fakes import FakeClient, dead_http_client, search_call


def _client(outputs):
    return FakeClient(outputs=outputs)


def _page(marker, day):
    paras = "".join(f"<p>{marker} paragraph {i} with extra words here.</p>" for i in range(6))
    return (
        "<html><head>"
        f'<meta property="article:published_time" content="{day}T08:00:00Z" />'
        f"</head><body><article>{paras}</article></body></html>"
    )


def _http_client(pages, seen):
    def handler(request):
        seen.append(str(request.url))
        if str(request.url) in pages:
            return httpx.Response(
                200, text=pages[str(request.url)],
                headers={"content-type": "text/html"},
            )
        return httpx.Response(404, text="gone")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_dual_queries_and_dedup():
    client = _client(
        [
            [search_call([("https://a.com/x?utm_s=1", "T", "2026-01-05 done")])],
            [
                search_call(
                    [
                        ("https://a.com/x", "T2", "dup of first"),
                        ("https://b.com/y", "T3", "April 2, 2026 news"),
                    ]
                )
            ],
        ]
    )
    items = gather_evidence(
        ["claim one"], client=client, http_client=dead_http_client()
    )
    assert [i.source_url for i in items] == ["https://a.com/x?utm_s=1", "https://b.com/y"]
    assert items[0].date == "2026-01-05"
    assert items[1].date == "2026-04-02"
    # one supporting + one refuting query per claim
    assert len(client.responses.calls) == 2
    assert "REFUTING" in client.responses.calls[1][1]["input"]
    assert client.responses.calls[0][1]["tools"] == [
        {"type": "web_search", "search_context_size": "low"}
    ]
    assert client.responses.calls[0][1]["include"] == ["web_search_call.results"]


def test_handles_live_api_dict_shaped_results():
    # The live API returns web_search_call.results entries as plain dicts.
    dict_call = {
        "type": "web_search_call",
        "status": "completed",
        "results": [
            {
                "type": "text_result",
                "title": "Lemon water fact check, April 3, 2026",
                "url": "https://debunk.com/x",
                "snippet": "",
            }
        ],
    }
    client = _client([[dict_call], [search_call([])]])
    items = gather_evidence(
        ["claim one"], client=client, http_client=dead_http_client()
    )
    assert [i.source_url for i in items] == ["https://debunk.com/x"]
    assert items[0].snippet == "Lemon water fact check, April 3, 2026"
    assert items[0].date == "2026-04-03"


def test_fetches_top3_interleaved_with_meta_dates():
    support = [
        ("https://x.com/s1", "S1 title", "search snippet s1"),
        ("https://x.com/s2", "S2 title", "search snippet s2"),
    ]
    refute = [
        ("https://x.com/r1", "R1 title", "search snippet r1"),
        ("https://x.com/r2", "R2 title", "April 2, 2026 search snippet r2"),
    ]
    client = _client([[search_call(support)], [search_call(refute)]])
    seen: list[str] = []
    http = _http_client(
        {
            "https://x.com/s1": _page("S1MARK", "2021-03-04"),
            "https://x.com/r1": _page("R1MARK", "2022-05-06"),
            "https://x.com/s2": _page("S2MARK", "2023-07-08"),
            "https://x.com/r2": _page("R2MARK", "2024-09-10"),
        },
        seen,
    )
    items = gather_evidence(["claim one"], client=client, http_client=http)
    assert [i.source_url for i in items] == [
        "https://x.com/s1",
        "https://x.com/r1",
        "https://x.com/s2",
        "https://x.com/r2",
    ]
    # round-robin fetch order: s1, r1, s2 -- r2 beyond the top-3 budget
    assert seen == ["https://x.com/s1", "https://x.com/r1", "https://x.com/s2"]
    assert "S1MARK" in items[0].snippet and len(items[0].snippet) <= 500
    assert items[0].date == "2021-03-04"
    assert "R1MARK" in items[1].snippet
    assert items[1].date == "2022-05-06"
    assert "S2MARK" in items[2].snippet
    assert items[2].date == "2023-07-08"
    # unfetched item keeps the search snippet and regex date
    assert items[3].snippet == "April 2, 2026 search snippet r2"
    assert items[3].date == "2026-04-02"


def test_fetch_failure_keeps_fallback_and_continues():
    support = [("https://x.com/gone", "G title", "2020-11-12 gone snippet")]
    refute = [("https://x.com/ok", "O title", "ok snippet")]
    client = _client([[search_call(support)], [search_call(refute)]])
    seen: list[str] = []
    http = _http_client({"https://x.com/ok": _page("OKMARK", "2019-01-02")}, seen)
    items = gather_evidence(["claim one"], client=client, http_client=http)
    assert items[0].snippet == "2020-11-12 gone snippet"
    assert items[0].date == "2020-11-12"
    assert "OKMARK" in items[1].snippet
    assert items[1].date == "2019-01-02"


def test_raises_when_model_does_not_search():
    client = _client([[SimpleNamespace(type="message", status="completed")]])
    with pytest.raises(GroundingError, match="no grounded search"):
        gather_evidence(["claim one"], client=client)


def test_raises_when_claim_has_zero_items():
    client = _client([[search_call([])], [search_call([])]])
    with pytest.raises(GroundingError, match="No grounded evidence"):
        gather_evidence(["claim one"], client=client)


def test_raises_on_api_failure():
    class Exploding:
        def create(self, **kwargs):
            raise ConnectionError("down")

    client = SimpleNamespace(responses=Exploding())
    with pytest.raises(GroundingError, match="Search grounding failed"):
        gather_evidence(["claim one"], client=client)


def test_tracks_usage():
    client = _client(
        [[search_call([("https://a.com", "T", "s")])], [search_call([])]]
    )
    tracker = UsageTracker()
    gather_evidence(
        ["c"], client=client, tracker=tracker, http_client=dead_http_client()
    )
    assert tracker.total_tokens == 300
