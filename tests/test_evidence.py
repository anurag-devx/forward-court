from types import SimpleNamespace

import pytest

from config import UsageTracker
from evidence.grounder import GroundingError, gather_evidence
from tests.fakes import FakeClient, search_call


def _client(outputs):
    return FakeClient(outputs=outputs)


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
    items = gather_evidence(["claim one"], client=client)
    assert [i.source_url for i in items] == ["https://a.com/x?utm_s=1", "https://b.com/y"]
    assert items[0].date == "2026-01-05"
    assert items[1].date == "2026-04-02"
    # one supporting + one refuting query per claim
    assert len(client.responses.calls) == 2
    assert "REFUTING" in client.responses.calls[1][1]["input"]
    assert client.responses.calls[0][1]["tools"] == [
        {"type": "web_search", "search_context_size": "medium"}
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
    items = gather_evidence(["claim one"], client=client)
    assert [i.source_url for i in items] == ["https://debunk.com/x"]
    assert items[0].snippet == "Lemon water fact check, April 3, 2026"
    assert items[0].date == "2026-04-03"


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
    gather_evidence(["c"], client=client, tracker=tracker)
    assert tracker.total_tokens == 300
