from court.defender import argue_true
from court.judge import judge_verdict
from court.models import Argument, ArgPoint, Verdict
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


def test_judge_drops_uncited_points():
    fake = FakeClient(parsed_by_model={Verdict: _verdict(verdict="False")})
    prosecution = Argument(
        points=[
            ArgPoint(text="real cite", source_url="https://a.com/x?utm_s=1"),
            ArgPoint(text="invented", source_url="https://evil.com/fake"),
        ]
    )
    verdict = judge_verdict(
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", client=fake
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
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", client=fake
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
        ["c"], _evidence(), prosecution, Argument(points=[]), "en", client=fake
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
    old = [
        EvidenceItem(
            claim="c",
            source_url="https://a.com/x",
            date="2015-04-25",
            snippet="old quake",
        )
    ]
    prosecution = Argument(points=[])
    defense = Argument(
        points=[ArgPoint(text="it happened", source_url="https://a.com/x")]
    )
    verdict = judge_verdict(["c"], old, prosecution, defense, "en", client=fake)
    assert verdict.verdict == "Misleading"
    assert "recycled" in verdict.reason


def test_judge_keeps_verdict_without_timeframe():
    fake = FakeClient(
        parsed_by_model={Verdict: _verdict(verdict="True", claim_timeframe=None)}
    )
    old = [
        EvidenceItem(
            claim="c",
            source_url="https://a.com/x",
            date="2015-04-25",
            snippet="old quake",
        )
    ]
    defense = Argument(
        points=[ArgPoint(text="it happened", source_url="https://a.com/x")]
    )
    verdict = judge_verdict(
        ["c"], old, Argument(points=[]), defense, "en", client=fake
    )
    assert verdict.verdict == "True"
