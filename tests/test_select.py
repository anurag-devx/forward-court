from config import UsageTracker
from evidence.models import ClaimReview, EvidenceItem
from evidence.select import MAX_PER_CLAIM, select_evidence


def _item(claim, url, **flags):
    return EvidenceItem(claim=claim, source_url=url, snippet="s", **flags)


def _review():
    return ClaimReview(
        rating_raw="False", verdict="False", claim_text="the claim here"
    )


def test_ranking_prefers_tier_then_flags_then_order():
    evidence = [
        _item("c", "https://plain.com/g"),
        _item("c", "https://www.altnews.in/tier1"),
        _item("c", "https://old.com/exact", exact_match=True),
        _item("c", "https://boomlive.in/fc", fact_check=True),
    ]
    kept = select_evidence(["c"], evidence)
    assert [i.source_url for i in kept] == [
        "https://boomlive.in/fc",
        "https://www.altnews.in/tier1",
        "https://old.com/exact",
        "https://plain.com/g",
    ]


def test_domain_dedupe_keeps_two_per_host():
    evidence = [
        _item("c", f"https://same.com/{i}") for i in range(5)
    ] + [_item("c", "https://other.com/x")]
    kept = select_evidence(["c"], evidence)
    assert [i.source_url for i in kept] == [
        "https://same.com/0",
        "https://same.com/1",
        "https://other.com/x",
    ]


def test_cap_boundary_per_claim():
    evidence = [
        _item("c", f"https://host{i}.com/x") for i in range(MAX_PER_CLAIM + 5)
    ]
    kept = select_evidence(["c"], evidence)
    assert len(kept) == MAX_PER_CLAIM
    assert kept[-1].source_url == f"https://host{MAX_PER_CLAIM - 1}.com/x"


def test_claim_review_items_exempt_from_cap_and_domain():
    evidence = [
        _item("c", f"https://same.com/{i}", claim_review=_review())
        for i in range(5)
    ] + [
        _item("c", f"https://host{i}.com/x")
        for i in range(MAX_PER_CLAIM + 5)
    ]
    kept = select_evidence(["c"], evidence)
    reviews = [i for i in kept if i.claim_review is not None]
    assert len(reviews) == 5
    assert len(kept) == 5 + MAX_PER_CLAIM
    assert kept[:5] == reviews


def test_per_claim_independence_and_empty_safe():
    assert select_evidence(["c"], []) == []
    evidence = [_item("a", "https://a.com/1"), _item("b", "https://b.com/2")]
    kept = select_evidence(["a", "b"], evidence)
    assert [i.claim for i in kept] == ["a", "b"]
    assert select_evidence(["a"], evidence) == [evidence[0]]


def test_capped_count_logged():
    tracker = UsageTracker()
    evidence = [_item("c", f"https://same.com/{i}") for i in range(4)]
    select_evidence(["c"], evidence, tracker=tracker)
    assert "evidence capped 2/4 items." in tracker.notes
