from datetime import date

import pytest

from court.defender import argue_true
from court.judge import judge_verdict
from court.models import Argument, ArgPoint, ImageContext, Verdict
from court.prosecutor import argue_false
from evidence.models import EvidenceItem
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
