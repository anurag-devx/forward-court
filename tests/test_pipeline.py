from pathlib import Path

import main
from court.models import Argument, ArgPoint, Verdict
from evidence.grounder import GroundingError
from evidence.relevance import RelevanceScore, RelevanceScores
from intake.models import IntakeResult
from links.scamtext import ScamReading
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
        ScamReading: ScamReading(scam_type="", signals=[]),
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
        ],
    )
    lookup_outputs = overrides.get(
        "lookup_outputs",
        [[search_call([("https://www.altnews.in/y", "T", "checked false")])]],
    )
    return FakeClient(
        parsed_by_model=parsed, outputs=outputs,
        lookup_outputs=lookup_outputs,
    )


def test_full_pipeline_offline():
    card, tracker = main.run_forward(
        (SAMPLES / "english.txt").read_text(),
        client=_pipeline_client(),
        http_client=dead_http_client(),
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "🔗 Source: https://debunk.com/x" in card
    # intake + 2 searches + lookup + relevance + prosecutor + defender + 3 votes
    assert tracker.total_tokens == 11 * 150


def test_pipeline_records_per_stage_breakdown():
    from collections import Counter

    _, tracker = main.run_forward(
        "some claim", client=_pipeline_client(), http_client=dead_http_client()
    )
    stages = [line.split(":")[0].strip() for line in tracker.breakdown()]
    # concurrent grounding/lookup interleave; compare order-free
    assert Counter(stages) == Counter(
        [
            "scamtext",
            "intake",
            "grounding",
            "grounding",
            "factcheck",
            "relevance",
            "prosecutor",
            "defender",
            "judge",
        ]
    )
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
    assert seen == {"factcheck": False, "claimreview": True, "cache": True}
    assert "lookup disabled" not in capsys.readouterr().out


def test_no_cache_flag_plumbs_through(monkeypatch):
    from config import UsageTracker

    seen = {}

    def fake_run(text, **kwargs):
        seen.update(kwargs)
        return ("card", UsageTracker())

    monkeypatch.setattr(main, "run_forward", fake_run)
    assert main.main(["hello", "--no-cache"]) == 0
    assert seen["cache"] is False


def test_card_out_writes_card_file(monkeypatch, tmp_path):
    from config import UsageTracker

    def fake_run(text, **kwargs):
        return ("CARD TEXT", UsageTracker())

    monkeypatch.setattr(main, "run_forward", fake_run)
    out = tmp_path / "card.txt"
    assert main.main(["hello", "--card-out", str(out)]) == 0
    assert out.read_text() == "CARD TEXT\n"


def test_claim_review_path_skips_debate_and_votes(monkeypatch):
    import evidence.factcheck as factcheck_module
    from evidence.models import ClaimReview

    review = ClaimReview(
        rating_raw="False",
        verdict="False",
        claim_text="hot lemon water cures cancer",
    )
    real_fetch = factcheck_module.fetch_page_unit

    def patched(args):
        url = args[0]
        snippet, date, _ = real_fetch(args)
        if "altnews" in url:
            return snippet, date, review
        return snippet, date, None

    monkeypatch.setattr(factcheck_module, "fetch_page_unit", patched)
    client = _pipeline_client()
    card, tracker = main.run_forward(
        "some claim", client=client, http_client=dead_http_client()
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "Already fact-checked by altnews.in" in card
    assert "🔗 Source: https://www.altnews.in/y" in card
    parses = [c for c in client.responses.calls if c[0] == "parse"]
    assert len(parses) == 2  # scamtext + intake only; no debate, no votes
    assert any("ClaimReview verdict" in note for note in tracker.notes)
    assert "factcheck" in tracker.times  # lookup awaited; its tokens count
    assert "grounding" not in tracker.times  # raced, abandoned, discarded
    assert not [
        stage for stage in tracker.stages if stage.startswith("grounding:")
    ]


def test_review_exit_does_not_wait_for_grounding(monkeypatch):
    import threading

    import evidence.factcheck as factcheck_module
    from evidence.models import ClaimReview

    review = ClaimReview(
        rating_raw="False",
        verdict="False",
        claim_text="hot lemon water cures cancer",
    )
    real_fetch = factcheck_module.fetch_page_unit

    def patched(args):
        url = args[0]
        snippet, date, _ = real_fetch(args)
        if "altnews" in url:
            return snippet, date, review
        return snippet, date, None

    monkeypatch.setattr(factcheck_module, "fetch_page_unit", patched)

    started = threading.Event()
    release = threading.Event()

    def slow_gather(claims, **kwargs):
        started.set()
        assert release.wait(timeout=60), "abandoned grounding never released"
        return []

    real_lookup = factcheck_module.lookup_factchecks

    def gated_lookup(claims, **kwargs):
        assert started.wait(timeout=60), "grounding never launched"
        return real_lookup(claims, **kwargs)

    monkeypatch.setattr(main, "gather_evidence", slow_gather)
    monkeypatch.setattr(main, "lookup_factchecks", gated_lookup)
    waited = {}
    try:
        card, _ = main.run_forward(
            "some claim", client=_pipeline_client(),
            http_client=dead_http_client(),
        )
        waited["grounding"] = release.is_set()
    finally:
        release.set()
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert waited["grounding"] is False


def test_contradicting_reviews_continue_to_debate(monkeypatch):
    import evidence.factcheck as factcheck_module
    from evidence.models import ClaimReview

    reviews = {
        "altnews": ClaimReview(
            rating_raw="False", verdict="False",
            claim_text="hot lemon water cures cancer",
        ),
        "boomlive": ClaimReview(
            rating_raw="True", verdict="True",
            claim_text="hot lemon water cures cancer",
        ),
    }
    real_fetch = factcheck_module.fetch_page_unit

    def patched(args):
        url = args[0]
        snippet, date, _ = real_fetch(args)
        for host, review in reviews.items():
            if host in url:
                return snippet, date, review
        return snippet, date, None

    monkeypatch.setattr(factcheck_module, "fetch_page_unit", patched)
    client = _pipeline_client(
        lookup_outputs=[[
            search_call([
                ("https://www.altnews.in/y", "T", "checked false"),
                ("https://www.boomlive.in/z", "T", "checked true"),
            ])
        ]]
    )
    card, tracker = main.run_forward(
        "some claim", client=client, http_client=dead_http_client()
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    parses = [c for c in client.responses.calls if c[0] == "parse"]
    assert len(parses) > 2  # debate + votes ran
    creates = [c for c in client.responses.calls if c[0] == "create"]
    assert any(
        not call[1].get("input", "").startswith("fact check:")
        for call in creates
    ), "grounding ran after the early check found no consensus"
    assert "grounding" in tracker.times  # raced grounding merged back
    assert any(
        stage.startswith("grounding:") for stage in tracker.stages
    )


def test_no_claimreview_flag_forces_debate_path(monkeypatch):
    import evidence.factcheck as factcheck_module
    from evidence.models import ClaimReview

    review = ClaimReview(
        rating_raw="False",
        verdict="False",
        claim_text="hot lemon water cures cancer",
    )
    real_fetch = factcheck_module.fetch_page_unit

    def patched(args):
        url = args[0]
        snippet, date, _ = real_fetch(args)
        if "altnews" in url:
            return snippet, date, review
        return snippet, date, None

    monkeypatch.setattr(factcheck_module, "fetch_page_unit", patched)
    client = _pipeline_client()
    card, tracker = main.run_forward(
        "some claim",
        client=client,
        http_client=dead_http_client(),
        claimreview=False,
    )
    assert "Already fact-checked" not in card
    parses = [c for c in client.responses.calls if c[0] == "parse"]
    # intake + relevance + prosecutor + defender + 3 judge votes
    assert len(parses) == 8  # scamtext + intake + relevance + 2 agents + 3 votes
    assert "ClaimReview short-circuit disabled." in tracker.notes


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
    assert len(parses) == 3  # scamtext + intake + relevance only


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
            from links.scamtext import ScamReading

            if kwargs.get("text_format") is ScamReading:
                parsed = ScamReading(scam_type="", signals=[])
            else:
                parsed = intake
            return SimpleNamespace(output_parsed=parsed, usage=FakeUsage())

        def create(self, **kwargs):
            threading.Event().wait(30)

    class HangingClient:
        responses = HangingGrounding()

    card, tracker = main.run_forward(
        "something", client=HangingClient(), deadline_s=0.5
    )
    assert "अभी सत्यापित नहीं हो सका" in card
    # grounding and lookup race; either may hold the last mark
    assert any("run timed out during stage" in n for n in tracker.notes)
    assert tracker.notes[-1] in (
        "run timed out during stage: grounding.",
        "run timed out during stage: lookup.",
    )


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
    assert len(client.responses.calls) == 2  # scamtext + intake only


def _no_claims_client():
    return _pipeline_client(
        parsed={
            IntakeResult: IntakeResult(
                language="en", claims=[], presented_as_recent=False
            )
        }
    )


def test_bare_link_without_claims_gets_link_reply():
    card, _ = main.run_forward(
        "https://example.com/some-article",
        client=_no_claims_client(), http_client=dead_http_client(),
    )
    assert card == main.BARE_LINK_NO_CLAIMS
    assert "safe" not in card.lower()


def test_bare_link_reply_lists_signals():
    card, _ = main.run_forward(
        "http://example.com/x",
        client=_no_claims_client(), http_client=dead_http_client(),
    )
    assert card.startswith(main.BARE_LINK_NO_CLAIMS)
    assert "plain_http" in card
    assert "safe" not in card.lower()


def test_grounding_error_propagates():
    client = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=["c"], presented_as_recent=False
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
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
