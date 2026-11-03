from card.corroborate import select_corroboration
from court.models import Verdict
from evidence.models import ClaimReview, EvidenceItem


def _verdict(**overrides):
    base = {
        "verdict": "True",
        "reason": "confirmed",
        "key_source": "https://key.example/a",
        "language": "en",
        "confidence": "high",
    }
    base.update(overrides)
    return Verdict(**base)


def _item(url, snippet, **flags):
    return EvidenceItem(
        claim="India beat South Africa in the final",
        source_url=url,
        date="2024-06-29",
        snippet=snippet,
        **flags,
    )


def _review(verdict, text):
    return ClaimReview(
        rating_raw=verdict, verdict=verdict, claim_text=text
    )


CLAIM = "India beat South Africa in the final"


def test_true_prefers_rated_factcheck_then_tier1_caps_two():
    evidence = [
        _item("https://key.example/a", "India beat South Africa"),
        _item(
            "https://sports.example/r", "India beat South Africa in the final",
        ),
        _item(
            "https://www.thehindu.com/r",
            "India beat South Africa in the final",
        ),
        _item(
            "https://factly.in/r", "India beat South Africa in the final",
            claim_review=_review(
                "True", "India beat South Africa in the final"
            ),
        ),
    ]
    picked = select_corroboration(_verdict(), [CLAIM], evidence)
    assert [item.source_url for item in picked] == [
        "https://factly.in/r",
        "https://www.thehindu.com/r",
    ]


def test_mismatched_factcheck_excluded():
    evidence = [
        _item("https://key.example/a", "India beat South Africa"),
        _item(
            "https://www.vishvasnews.com/x",
            "David Miller clarified that he has not retired. Details here.",
            fact_check=True,
        ),
        _item(
            "https://www.bbc.com/r", "India beat South Africa in the final",
        ),
    ]
    picked = select_corroboration(_verdict(), [CLAIM], evidence)
    assert [item.source_url for item in picked] == [
        "https://www.bbc.com/r"
    ]


def test_key_domain_and_duplicates_excluded():
    evidence = [
        _item("https://key.example/a", "India beat South Africa"),
        _item(
            "https://key.example/b", "India beat South Africa in the final",
        ),
        _item(
            "https://www.bbc.com/one",
            "India beat South Africa in the final",
        ),
        _item(
            "https://www.bbc.com/two",
            "India beat South Africa in the final",
        ),
    ]
    picked = select_corroboration(_verdict(), [CLAIM], evidence)
    assert [item.source_url for item in picked] == [
        "https://www.bbc.com/one"
    ]


def test_false_only_takes_false_rated_factchecks():
    verdict = _verdict(
        verdict="False", key_source="https://factly.in/k",
    )
    evidence = [
        _item(
            "https://factly.in/k", "the claim is false",
            claim_review=_review("False", CLAIM),
        ),
        _item(
            "https://sports.example/r", "India beat South Africa in the final",
        ),
        _item(
            "https://www.altnews.in/x", "the claim is false",
            claim_review=_review("False", CLAIM),
        ),
    ]
    picked = select_corroboration(verdict, [CLAIM], evidence)
    assert [item.source_url for item in picked] == [
        "https://www.altnews.in/x"
    ]


def test_misleading_takes_direct_reports():
    verdict = _verdict(
        verdict="Misleading", key_source="https://old.example/k",
    )
    evidence = [
        _item("https://old.example/k", "India beat South Africa"),
        _item(
            "https://www.bbc.com/r", "India beat South Africa in the final",
        ),
    ]
    picked = select_corroboration(verdict, [CLAIM], evidence)
    assert [item.source_url for item in picked] == [
        "https://www.bbc.com/r"
    ]


def test_unverified_and_dangerous_take_none():
    evidence = [
        _item(
            "https://sports.example/r", "India beat South Africa in the final",
        ),
    ]
    assert (
        select_corroboration(_verdict(verdict="Unverified"), [CLAIM], evidence)
        == []
    )
    assert (
        select_corroboration(_verdict(verdict="DANGEROUS"), [CLAIM], evidence)
        == []
    )


def test_tier2_tier3_visual_shared_denylisted_never_qualify():
    evidence = [
        _item("https://key.example/a", "India beat South Africa"),
        _item(
            "https://sports.example/r",
            "India beat South Africa in the final",
        ),
        _item(
            "https://x.com/p", "India beat South Africa in the final",
        ),
        _item(
            "https://pic.example/p", "India beat South Africa in the final",
            visual_only=True,
        ),
        _item(
            "https://shared.example/s",
            "India beat South Africa in the final",
            shared_article=True,
        ),
        _item(
            "https://instalker.org/x",
            "India beat South Africa in the final",
        ),
    ]
    assert select_corroboration(_verdict(), [CLAIM], evidence) == []
