from config import UsageTracker, reasoning_kwargs
from court.defender import argue_true
from court.models import Argument
from intake.extractor import extract_intake
from intake.models import IntakeResult
from tests.fakes import FakeClient


def test_table_values_and_unknown_stage():
    assert reasoning_kwargs("intake") == {"reasoning": {"effort": "low"}}
    assert reasoning_kwargs("relevance") == {"reasoning": {"effort": "low"}}
    assert reasoning_kwargs("mismatch") == {"reasoning": {"effort": "low"}}
    assert reasoning_kwargs("vision") == {"reasoning": {"effort": "low"}}
    assert reasoning_kwargs("video") == {"reasoning": {"effort": "medium"}}
    assert reasoning_kwargs("prosecutor") == {"reasoning": {"effort": "medium"}}
    assert reasoning_kwargs("defender") == {"reasoning": {"effort": "medium"}}
    assert reasoning_kwargs("judge") == {"reasoning": {"effort": "medium"}}
    assert reasoning_kwargs("transcribe") == {}


def test_low_stage_sends_low_effort():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en",
                claims=["c"],
                presented_as_recent=False,
            )
        }
    )
    extract_intake("text", client=fake, tracker=UsageTracker())
    assert fake.responses.calls[0][1]["reasoning"] == {"effort": "low"}


def test_medium_stage_sends_medium_effort():
    fake = FakeClient(parsed_by_model={Argument: Argument(points=[])})
    argue_true(["c"], [], client=fake, tracker=UsageTracker())
    assert fake.responses.calls[0][1]["reasoning"] == {"effort": "medium"}
