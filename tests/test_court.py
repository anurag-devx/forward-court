from datetime import date

import pytest

from config import UsageTracker
from court.defender import argue_true
from court.judge import (
    find_claim_review_verdict,
    judge_verdict,
    no_evidence_verdict,
    outlet_host,
)
from court.models import (
    Argument,
    ArgPoint,
    ImageContext,
    Verdict,
    evidence_brief,
)
from court.prosecutor import argue_false
from evidence.models import ClaimReview, EvidenceItem
from tests.fakes import FakeClient


def _evidence():
    return [
        EvidenceItem(
            claim="c",
            source_url="https://a.com/x",
            date="2026-01-05",
            snippet="s1",
        ),
        EvidenceItem(
            claim="c", source_url="https://b.com/y", date=None, snippet="s2"
        ),
    ]


def test_debaters_receive_only_evidence_list():
    fake = FakeClient(
        parsed_by_model={
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://a.com/x")]
            )
        }
    )
    argue_false(["c"], _evidence(), client=fake)
    argue_true(["c"], _evidence(), client=fake)
    for _, kwargs in fake.responses.calls:
        assert "https://a.com/x" in kwargs["input"]
        assert "https://b.com/y" in kwargs["input"]


def _verdict(**overrides):
    base = {
        "verdict": "False",
        "reason": "cited debunk",
        "key_source": "https://a.com/x",
        "language": "en",
        "confidence": "high",
        "claim_timeframe": None,
    }
    base.update(overrides)
    return Verdict(**base)


def _old_evidence(day="2015-04-25"):
    return [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date=day, snippet="old"
        )
    ]


def _defense():
    return Argument(
        points=[ArgPoint(text="it happened", source_url="https://a.com/x")]
    )


def test_judge_drops_uncited_points():
    fake = FakeClient(parsed_by_model={Verdict: _verdict(verdict="False")})
    prosecution = Argument(
        points=[
            ArgPoint(text="real cite", source_url="https://a.com/x?utm_s=1"),
            ArgPoint(text="invented", source_url="https://evil.com/fake"),
        ]
    )
    verdict = judge_verdict(
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", False,
        client=fake,
    )
    brief = fake.responses.calls[0][1]["input"]
    assert "real cite" in brief
    assert "invented" not in brief
    assert verdict.verdict == "False"


def test_judge_forces_unverified_when_all_dropped():
    fake = FakeClient(parsed_by_model={Verdict: _verdict(verdict="True")})
    prosecution = Argument(
        points=[ArgPoint(text="invented", source_url="https://evil.com")]
    )
    verdict = judge_verdict(
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", False,
        client=fake,
    )
    assert verdict.verdict == "Unverified"
    assert verdict.confidence == "low"


def test_judge_falls_back_to_most_cited_key_source():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(key_source="https://evil.com")}
    )
    prosecution = Argument(
        points=[
            ArgPoint(text="p1", source_url="https://b.com/y"),
            ArgPoint(text="p2", source_url="https://b.com/y"),
            ArgPoint(text="p3", source_url="https://a.com/x"),
        ]
    )
    verdict = judge_verdict(
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", False,
        client=fake,
    )
    assert verdict.key_source == "https://b.com/y"


def test_judge_flags_recycled_old_news():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(
                verdict="True", confidence="high", claim_timeframe="2026-10-27"
            )
        }
    )
    verdict = judge_verdict(
        ["c"], _old_evidence(), Argument(points=[]), _defense(), "en", True,
        client=fake,
    )
    assert verdict.verdict == "Misleading"
    assert "recycled" in verdict.reason


def test_judge_skips_override_when_not_presented_as_recent():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(verdict="True", claim_timeframe="2026-10-27")
        }
    )
    verdict = judge_verdict(
        ["c"], _old_evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake,
    )
    assert verdict.verdict == "True"


def test_judge_defaults_implied_date_to_today():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict="True", claim_timeframe=None)}
    )
    verdict = judge_verdict(
        ["c"], _old_evidence(), Argument(points=[]), _defense(), "en", True,
        client=fake, today=date(2026, 10, 27),
    )
    assert verdict.verdict == "Misleading"


@pytest.mark.parametrize(
    "source_day, expected",
    [("2026-09-27", "True"), ("2026-09-26", "Misleading")],
)
def test_override_threshold_is_30_days(source_day, expected):
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict="True", claim_timeframe=None)}
    )
    verdict = judge_verdict(
        ["c"], _old_evidence(source_day), Argument(points=[]), _defense(),
        "en", True, client=fake, today=date(2026, 10, 27),
    )
    assert verdict.verdict == expected


def test_override_cites_earliest_source():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(
                verdict="True", key_source="https://new.com/x",
                claim_timeframe="2026-10-27",
            )
        }
    )
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://new.com/x", date="2026-10-01",
            snippet="new",
        ),
        EvidenceItem(
            claim="c", source_url="https://old.com/y", date="2015-04-25",
            snippet="old",
        ),
    ]
    defense = Argument(
        points=[
            ArgPoint(text="new", source_url="https://new.com/x"),
            ArgPoint(text="old", source_url="https://old.com/y"),
        ]
    )
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", True,
        client=fake,
    )
    assert verdict.verdict == "Misleading"
    assert verdict.key_source == "https://old.com/y"


def test_visual_dates_never_trigger_override():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict="True", claim_timeframe=None)}
    )
    visual = [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date="2015-04-25",
            snippet="similar", visual_only=True,
        )
    ]
    verdict = judge_verdict(
        ["c"], visual, Argument(points=[]), _defense(), "en", True,
        client=fake, today=date(2026, 10, 27),
    )
    assert verdict.verdict == "True"


@pytest.mark.parametrize(
    "model_verdict, origin_only, reverse, expected, confidence",
    [
        ("True", True, False, "Unverified", "low"),
        ("False", True, False, "Unverified", "low"),
        ("False", False, False, "False", "high"),
        ("True", True, True, "True", "high"),
        ("Misleading", True, False, "Misleading", "high"),
    ],
)
def test_origin_cap_without_reverse_evidence(
    model_verdict, origin_only, reverse, expected, confidence
):
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict=model_verdict)}
    )
    verdict = judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake,
        image=ImageContext(origin_only=origin_only, reverse_evidence=reverse),
    )
    assert verdict.verdict == expected
    assert verdict.confidence == confidence


def test_unverified_confidence_capped_at_medium():
    fake = FakeClient(
        parsed_by_model={
            Verdict: _verdict(verdict="Unverified", confidence="high")
        }
    )
    verdict = judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake,
    )
    assert verdict.verdict == "Unverified"
    assert verdict.confidence == "medium"


def test_brief_labels_fact_check_items():
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/x", date=None,
            snippet="debunked", fact_check=True,
        )
    ]
    assert "existing fact-check" in evidence_brief(evidence)


def test_brief_labels_tiers():
    import json

    evidence = [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/x", date=None,
            snippet="t1",
        ),
        EvidenceItem(
            claim="c", source_url="https://x.com/post", date=None,
            snippet="t3",
        ),
    ]
    entries = json.loads(evidence_brief(evidence))
    assert [entry["tier"] for entry in entries] == [1, 3]


@pytest.mark.parametrize(
    "key_source, cited_url, expected",
    [
        ("https://www.altnews.in/x", "https://a.com/x", "yes"),
        ("https://a.com/x", "https://www.altnews.in/x", "yes"),
        ("https://a.com/x", "https://a.com/x", "no"),
    ],
)
def test_judge_notes_fact_check_usage(key_source, cited_url, expected):
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(key_source=key_source)}
    )
    evidence = _evidence() + [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/x", date=None,
            snippet="debunked", fact_check=True,
        )
    ]
    defense = Argument(
        points=[ArgPoint(text="p", source_url=cited_url)]
    )
    tracker = UsageTracker()
    judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", False,
        client=fake, tracker=tracker,
    )
    assert tracker.notes[-1] == f"existing fact-check cited: {expected}."
    assert tracker.notes[0].startswith("judge agreement: 3/3")


def test_judge_silent_without_fact_check_items():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    tracker = UsageTracker()
    judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake, tracker=tracker,
    )
    assert tracker.notes == ["judge agreement: 3/3 (False, False, False)."]


def test_majority_vote_sets_verdict_and_medium():
    fake = FakeClient(
        parse_queue=[
            _verdict(verdict="False", reason="first"),
            _verdict(verdict="Misleading", reason="second"),
            _verdict(verdict="False", reason="third"),
        ]
    )
    tracker = UsageTracker()
    verdict = judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake, tracker=tracker,
    )
    assert verdict.verdict == "False"
    assert verdict.confidence == "medium"
    assert verdict.reason == "first"
    assert tracker.notes == ["judge agreement: 2/3 (False, Misleading, False)."]


def test_tie_vote_becomes_unverified_low():
    fake = FakeClient(
        parse_queue=[
            _verdict(verdict="True"),
            _verdict(verdict="False"),
            _verdict(verdict="Misleading"),
        ]
    )
    verdict = judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake,
    )
    assert verdict.verdict == "Unverified"
    assert verdict.confidence == "low"


def test_tier3_key_source_rejected():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(key_source="https://x.com/post")}
    )
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/x", date=None,
            snippet="t1",
        ),
        EvidenceItem(
            claim="c", source_url="https://x.com/post", date=None,
            snippet="t3",
        ),
    ]
    defense = Argument(
        points=[
            ArgPoint(text="a", source_url="https://x.com/post"),
            ArgPoint(text="b", source_url="https://x.com/post"),
            ArgPoint(text="c", source_url="https://www.altnews.in/x"),
        ]
    )
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", False,
        client=fake,
    )
    assert verdict.key_source == "https://www.altnews.in/x"


def test_all_tier3_keeps_most_cited():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(key_source="https://evil.com")}
    )
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://x.com/a", date=None, snippet="s1"
        ),
        EvidenceItem(
            claim="c", source_url="https://x.com/b", date=None, snippet="s2"
        ),
    ]
    defense = Argument(
        points=[
            ArgPoint(text="a", source_url="https://x.com/b"),
            ArgPoint(text="b", source_url="https://x.com/b"),
            ArgPoint(text="c", source_url="https://x.com/a"),
        ]
    )
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", False,
        client=fake,
    )
    assert verdict.key_source == "https://x.com/b"


def test_judge_keeps_verdict_without_source_dates():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict="True", claim_timeframe=None)}
    )
    dateless = [
        EvidenceItem(claim="c", source_url="https://a.com/x", date=None, snippet="s")
    ]
    verdict = judge_verdict(
        ["c"], dateless, Argument(points=[]), _defense(), "en", True,
        client=fake, today=date(2026, 10, 27),
    )
    assert verdict.verdict == "True"


def _review_item(url, claim, verdict, reviewed, **flags):
    return EvidenceItem(
        claim=claim,
        source_url=url,
        date=None,
        snippet="s",
        claim_review=ClaimReview(
            rating_raw="False", verdict=verdict, claim_text=reviewed
        ),
        **flags,
    )


def test_claim_review_hit_sets_verdict_without_model_calls():
    item = _review_item(
        "https://www.altnews.in/check", "Vaccines contain microchips",
        "False", "Vaccines contain microchips, the viral claim says",
    )
    verdict = find_claim_review_verdict(
        ["Vaccines contain microchips"], [item], "en"
    )
    assert verdict.verdict == "False"
    assert verdict.confidence == "high"
    assert verdict.key_source == "https://www.altnews.in/check"
    assert verdict.fact_checked_by == "altnews.in"
    assert "altnews.in" in verdict.reason


def test_claim_review_unverified_caps_confidence_at_medium():
    item = _review_item(
        "https://www.altnews.in/check", "Vaccines contain microchips",
        "Unverified", "Vaccines contain microchips, the viral claim says",
    )
    verdict = find_claim_review_verdict(
        ["Vaccines contain microchips"], [item], "en"
    )
    assert verdict.verdict == "Unverified"
    assert verdict.confidence == "medium"


def test_claim_review_matches_on_shared_tokens():
    item = _review_item(
        "https://boomlive.in/x", "Prime minister resigned yesterday",
        "Misleading", "Did the prime minister quit yesterday morning?",
    )
    verdict = find_claim_review_verdict(
        ["Prime minister resigned yesterday"], [item], "en"
    )
    assert verdict is not None
    assert verdict.verdict == "Misleading"


def test_claim_review_matches_lookup_item_with_score_two():
    item = _review_item(
        "https://factly.in/y", "कल पटना में बाढ़ आई", "False",
        "Completely different English words here",
        fact_check=True, relevance=2,
    )
    verdict = find_claim_review_verdict(["कल पटना में बाढ़ आई"], [item], "hi")
    assert verdict is not None
    assert verdict.verdict == "False"
    assert verdict.language == "hi"


def test_claim_review_lookup_match_needs_score_two():
    item = _review_item(
        "https://factly.in/y", "कल पटना में बाढ़ आई", "False",
        "Completely different English words here",
        fact_check=True, relevance=1,
    )
    assert (
        find_claim_review_verdict(["कल पटना में बाढ़ आई"], [item], "hi")
        is None
    )


def test_claim_review_contradiction_falls_through():
    items = [
        _review_item(
            "https://www.altnews.in/a", "Vaccines contain microchips",
            "False", "Vaccines contain microchips in the blood",
        ),
        _review_item(
            "https://boomlive.in/b", "Vaccines contain microchips",
            "True", "Do vaccines contain microchips inside blood?",
        ),
    ]
    assert find_claim_review_verdict(["Vaccines contain microchips"], items, "en") is None


def test_claim_review_ignores_tier2_and_unmatched():
    tier2 = _review_item(
        "https://randomblog.com/z", "Vaccines contain microchips",
        "False", "Vaccines contain microchips", relevance=2,
        fact_check=False,
    )
    assert find_claim_review_verdict(["Vaccines contain microchips"], [tier2], "en") is None
    unmatched = _review_item(
        "https://www.altnews.in/a", "Vaccines contain microchips",
        "False", "Cricket match abandoned due to rain", relevance=2,
        fact_check=False,
    )
    assert (
        find_claim_review_verdict(["Vaccines contain microchips"], [unmatched], "en")
        is None
    )
    plain = EvidenceItem(
        claim="c", source_url="https://www.altnews.in/a", snippet="s"
    )
    assert find_claim_review_verdict(["c"], [plain], "en") is None


def test_outlet_host_strips_www():
    assert outlet_host("https://www.altnews.in/a") == "altnews.in"
    assert outlet_host("not a url") == "not a url"


def test_no_evidence_verdict_is_unverified_low():
    verdict = no_evidence_verdict("en")
    assert verdict.verdict == "Unverified"
    assert verdict.confidence == "low"
    assert verdict.key_source == ""
    assert no_evidence_verdict("hi").reason != verdict.reason


def test_judge_brief_marks_claims_without_evidence():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    evidence = [
        EvidenceItem(claim="kept", source_url="https://a.com/x", snippet="s")
    ]
    judge_verdict(
        ["kept", "bare"], evidence, Argument(points=[]), _defense(), "en",
        False, client=fake, no_evidence_claims=["bare"],
    )
    sent = fake.responses.calls[0][1]["input"]
    assert "bare" in sent
    assert "no kept evidence" in sent


def _media_image():
    return ImageContext(origin_only=False, reverse_evidence=True)


def _true_recent_vote():
    return FakeClient(
        parsed_by_model={
            Verdict: _verdict(
                verdict="True", confidence="high",
                claim_timeframe="2026-10-27",
            )
        }
    )


def test_media_override_ignores_old_topic_article():
    verdict = judge_verdict(
        ["c"], _old_evidence(), Argument(points=[]), _defense(), "en", True,
        client=_true_recent_vote(), image=_media_image(),
    )
    assert verdict.verdict == "True"


def test_media_override_fires_on_old_exact_match():
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://a.com/x", date="2015-04-25",
            snippet="old", exact_match=True,
        )
    ]
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), _defense(), "en", True,
        client=_true_recent_vote(), image=_media_image(),
        exact_urls={"https://a.com/x"},
    )
    assert verdict.verdict == "Misleading"
    assert verdict.key_source == "https://a.com/x"


def test_media_override_fires_on_factcheck_in_exact_urls():
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/old", date="2015-04-25",
            snippet="old", fact_check=True,
        )
    ]
    defense = Argument(
        points=[ArgPoint(text="old", source_url="https://www.altnews.in/old")]
    )
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", True,
        client=_true_recent_vote(), image=_media_image(),
        exact_urls={"https://www.altnews.in/old"},
    )
    assert verdict.verdict == "Misleading"


def test_media_override_fires_on_review_matching_media_claim():
    item = _review_item(
        "https://www.altnews.in/old", "photo shows Patna floods",
        "False", "photo shows Patna floods in the viral post",
    )
    item = item.model_copy(update={"date": "2015-04-25"})
    defense = Argument(
        points=[ArgPoint(text="old", source_url="https://www.altnews.in/old")]
    )
    verdict = judge_verdict(
        ["photo shows Patna floods"], [item], Argument(points=[]), defense,
        "en", True, client=_true_recent_vote(), image=_media_image(),
        media_claims=["photo shows Patna floods"],
    )
    assert verdict.verdict == "Misleading"


def test_media_override_skips_untied_factcheck():
    evidence = [
        EvidenceItem(
            claim="c", source_url="https://www.altnews.in/other",
            date="2015-04-25", snippet="old", fact_check=True,
        )
    ]
    defense = Argument(
        points=[ArgPoint(text="old", source_url="https://www.altnews.in/other")]
    )
    verdict = judge_verdict(
        ["c"], evidence, Argument(points=[]), defense, "en", True,
        client=_true_recent_vote(), image=_media_image(),
        exact_urls=set(), media_claims=["c"],
    )
    assert verdict.verdict == "True"


def test_judge_brief_sections_media_first_and_mismatch_line():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    judge_verdict(
        ["floods hit Patna", "video shows it"], _evidence(),
        Argument(points=[]), _defense(), "en", False, client=fake,
        media_claims=["video shows it"], media_mismatch="shows a test pattern",
    )
    sent = fake.responses.calls[0][1]["input"]
    assert "Media claims:\n['video shows it']" in sent
    assert "Event claims:\n['floods hit Patna']" in sent
    assert sent.index("Media claims") < sent.index("Event claims")
    assert "media does not show the claimed event" in sent
    assert "test pattern" in sent


def test_judge_brief_omits_mismatch_line_when_none():
    fake = FakeClient(parsed_by_model={Verdict: _verdict()})
    judge_verdict(
        ["c"], _evidence(), Argument(points=[]), _defense(), "en", False,
        client=fake,
    )
    assert "Media mismatch" not in fake.responses.calls[0][1]["input"]
