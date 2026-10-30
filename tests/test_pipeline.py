from pathlib import Path

import main
from court.models import Argument, ArgPoint, Verdict
from evidence.grounder import GroundingError
from intake.models import IntakeResult
from tests.fakes import FakeClient, search_call

SAMPLES = Path(__file__).parent / "samples"


def _pipeline_client(**overrides):
    parsed = {
        IntakeResult: IntakeResult(language="en", claims=["hot lemon water cures cancer"]),
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
    return FakeClient(parsed_by_model=parsed, outputs=outputs)


def test_full_pipeline_offline():
    card, tracker = main.run_forward(
        (SAMPLES / "english.txt").read_text(), client=_pipeline_client()
    )
    assert "❌ *Forward Court Verdict: FALSE*" in card
    assert "🔗 Source: https://debunk.com/x" in card
    # intake + 2 searches + prosecutor + defender + judge
    assert tracker.total_tokens == 6 * 150


def test_no_claims_short_circuits():
    client = _pipeline_client(
        parsed={IntakeResult: IntakeResult(language="en", claims=[])}
    )
    card, _ = main.run_forward("good morning!", client=client)
    assert card == "No checkable claims found in this forward."
    assert len(client.responses.calls) == 1  # intake only


def test_grounding_error_propagates():
    client = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(language="en", claims=["c"]),
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
