from config import UsageTracker
from evidence.models import EvidenceItem
from evidence.relevance import (
    RelevanceScore,
    RelevanceScores,
    filter_evidence,
)
from tests.fakes import FakeClient


def _item(claim, url):
    return EvidenceItem(claim=claim, source_url=url, snippet="s")


def _client(scores):
    return FakeClient(
        parsed_by_model={
            RelevanceScores: RelevanceScores(
                scores=[
                    RelevanceScore(index=i, score=s)
                    for i, s in enumerate(scores)
                ]
            )
        }
    )


def test_drops_zero_scores_and_keeps_rest():
    evidence = [_item("c", f"https://x.com/{i}") for i in range(3)]
    kept, bare = filter_evidence(["c"], evidence, client=_client([2, 0, 1]))
    assert [item.source_url for item in kept] == [
        "https://x.com/0",
        "https://x.com/2",
    ]
    assert bare == []
    assert kept[0].relevance == 2
    assert kept[1].relevance == 1


def test_claim_with_all_zeros_has_no_evidence():
    evidence = [_item("gone", "https://x.com/0"), _item("kept", "https://x.com/1")]
    kept, bare = filter_evidence(
        ["gone", "kept"], evidence, client=_client([0, 2])
    )
    assert [item.claim for item in kept] == ["kept"]
    assert bare == ["gone"]


def test_empty_evidence_marks_all_claims_bare():
    kept, bare = filter_evidence(["a", "b"], [], client=_client([]))
    assert kept == []
    assert bare == ["a", "b"]


def test_dropped_count_and_bare_claims_logged():
    tracker = UsageTracker()
    evidence = [_item("gone", "https://x.com/0"), _item("kept", "https://x.com/1")]
    filter_evidence(["gone", "kept"], evidence, client=_client([0, 2]),
                    tracker=tracker)
    assert "relevance filter dropped 1/2 items." in tracker.notes
    assert "no evidence kept for claim: gone" in tracker.notes
    assert tracker.stages["relevance"] == (100, 50)


def test_missing_scores_default_to_kept():
    evidence = [_item("c", f"https://x.com/{i}") for i in range(3)]
    kept, _ = filter_evidence(["c"], evidence, client=_client([0]))
    assert [item.source_url for item in kept] == [
        "https://x.com/1",
        "https://x.com/2",
    ]
