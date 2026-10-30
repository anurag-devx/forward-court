from pathlib import Path

import main
from court.models import Argument, ArgPoint, Verdict
from evidence.grounder import GroundingError
from evidence.relevance import RelevanceScore, RelevanceScores
from intake.models import IntakeResult
from tests.fakes import FakeClient, dead_http_client, search_call

SAMPLES = Path(__file__).parent / "samples"


def _all_relevant():
    return RelevanceScores(
        scores=[RelevanceScore(index=i, score=2) for i in range(12)]
    )


def _pipeline_client(**overrides):
    parsed = {
        IntakeResult: IntakeResult(
            language="en",
            claims=["hot lemon water cures cancer"],
            presented_as_recent=False,
        ),
        RelevanceScores: _all_relevant(),
        Argument: Argument(
            points=[ArgPoint(text="no evidence", source_url="https://debunk.com/x")]
        ),
        Verdict: Verdict(
            verdict="False",
            reason="No credible source supports this.",
            key_source="https://debunk.com/x",
            language="en",
            confidence="high",
            claim_timeframe=None,
        ),
    }
    parsed.update(overrides.get("parsed", {}))
    outputs = overrides.get(
        "outputs",
        [
            [search_call([("https://debunk.com/x", "T", "April 3, 2026 review")])],
            [search_call([("https://debunk.com/x", "T", "dup")])],
            [search_call([("https://www.altnews.in/y", "T", "checked false")])],
        ],
    )
    return FakeClient(parsed_by_model=parsed, outputs=outputs)


def test_full_pipeline_offline():
    card, tracker = main.run_forward(
        (SAMPLES / "english.txt").read_text(),
        client=_pipeline_client(),
        http_client=dead_http_client(),
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "🔗 Source: https://debunk.com/x" in card
    # intake + 2 searches + lookup + relevance + prosecutor + defender + 3 votes
    assert tracker.total_tokens == 10 * 150


def test_pipeline_records_per_stage_breakdown():
    _, tracker = main.run_forward(
        "some claim", client=_pipeline_client(), http_client=dead_http_client()
    )
    stages = [line.split(":")[0].strip() for line in tracker.breakdown()]
    assert stages == [
        "intake",
        "grounding",
        "grounding",
        "factcheck",
        "relevance",
        "prosecutor",
        "defender",
        "judge",
    ]
    assert any("claim1:support" in line for line in tracker.breakdown())
    assert any("claim1:refute" in line for line in tracker.breakdown())


def test_factcheck_disabled_skips_lookup():
    client = _pipeline_client()
    card, tracker = main.run_forward(
        "something",
        client=client,
        http_client=dead_http_client(),
        factcheck=False,
    )
    creates = [c for c in client.responses.calls if c[0] == "create"]
    assert len(creates) == 2  # grounding only, no lookup
    assert "fact-check lookup disabled." in tracker.notes
    assert "FALSE" in card.splitlines()[0]


def test_no_factcheck_flag_plumbs_through(monkeypatch, capsys):
    from config import UsageTracker

    seen = {}

    def fake_run(text, **kwargs):
        seen.update(kwargs)
        return ("card", UsageTracker())

    monkeypatch.setattr(main, "run_forward", fake_run)
    assert main.main(["hello", "--no-factcheck"]) == 0
    assert seen == {"factcheck": False}
    assert "lookup disabled" not in capsys.readouterr().out


def test_claim_review_path_skips_debate_and_votes(monkeypatch):
    import evidence.factcheck as factcheck_module
    from evidence.models import ClaimReview

    review = ClaimReview(
        rating_raw="False",
        verdict="False",
        claim_text="hot lemon water cures cancer",
    )
    real_enrich = factcheck_module.enrich_from_page

    def patched(url, title, fallback, **kwargs):
        snippet, date, _ = real_enrich(url, title, fallback, **kwargs)
        if "altnews" in url:
            return snippet, date, review
        return snippet, date, None

    monkeypatch.setattr(factcheck_module, "enrich_from_page", patched)
    client = _pipeline_client()
    card, tracker = main.run_forward(
        "some claim", client=client, http_client=dead_http_client()
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "Already fact-checked by altnews.in" in card
    assert "🔗 Source: https://www.altnews.in/y" in card
    parses = [c for c in client.responses.calls if c[0] == "parse"]
    assert len(parses) == 2  # intake + relevance; no debate, no judge votes
    assert any("ClaimReview verdict" in note for note in tracker.notes)


def test_all_filtered_evidence_yields_unverified_without_debate():
    client = _pipeline_client(
        parsed={
            RelevanceScores: RelevanceScores(
                scores=[
                    RelevanceScore(index=0, score=0),
                    RelevanceScore(index=1, score=0),
                ]
            )
        }
    )
    card, tracker = main.run_forward(
        "some claim", client=client, http_client=dead_http_client()
    )
    assert "UNVERIFIED" in card.splitlines()[0]
    assert "🔗" not in card
    assert "no relevant evidence: verdict Unverified." in tracker.notes
    parses = [c for c in client.responses.calls if c[0] == "parse"]
    assert len(parses) == 2  # intake + relevance only


def test_deadline_returns_timeout_card_naming_stalled_stage():
    import threading

    class HangingResponses:
        def parse(self, **kwargs):
            threading.Event().wait(30)

        def create(self, **kwargs):
            threading.Event().wait(30)

    class HangingClient:
        responses = HangingResponses()

    card, tracker = main.run_forward(
        "something", client=HangingClient(), deadline_s=0.2
    )
    assert "UNVERIFIED" in card.splitlines()[0]
    assert "Couldn't verify right now" in card
    assert "run timed out during stage: intake." in tracker.notes


def test_deadline_card_uses_detected_language():
    import threading
    from types import SimpleNamespace

    from tests.fakes import FakeUsage

    intake = IntakeResult(
        language="hi", claims=["c"], presented_as_recent=False
    )

    class HangingGrounding:
        def parse(self, **kwargs):
            return SimpleNamespace(output_parsed=intake, usage=FakeUsage())

        def create(self, **kwargs):
            threading.Event().wait(30)

    class HangingClient:
        responses = HangingGrounding()

    card, tracker = main.run_forward(
        "something", client=HangingClient(), deadline_s=0.5
    )
    assert "अभी सत्यापित नहीं हो सका" in card
    assert "run timed out during stage: grounding." in tracker.notes


def test_no_claims_short_circuits():
    client = _pipeline_client(
        parsed={
            IntakeResult: IntakeResult(
                language="en", claims=[], presented_as_recent=False
            )
        }
    )
    card, _ = main.run_forward("good morning!", client=client)
    assert card == "No checkable claims found in this forward."
    assert len(client.responses.calls) == 1  # intake only


def test_grounding_error_propagates():
    client = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=["c"], presented_as_recent=False
            ),
        },
        outputs=[[search_call([])], [search_call([])]],
    )
    try:
        main.run_forward("something", client=client)
    except GroundingError:
        pass
    else:
        raise AssertionError("expected GroundingError")


def test_cli_exits_on_empty_text(capsys):
    assert main.main([""]) == 2
    assert "provide forward text" in capsys.readouterr().err.lower()


def test_samples_exist_and_differ():
    texts = {
        name: (SAMPLES / name).read_text()
        for name in ("english.txt", "hindi.txt", "recycled.txt")
    }
    assert all(t.strip() for t in texts.values())
    assert len(set(texts.values())) == 3
