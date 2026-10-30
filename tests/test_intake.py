from config import MAX_CLAIMS, UsageTracker
from intake.extractor import extract_intake
from intake.models import IntakeResult
from tests.fakes import FakeClient


def test_extract_intake_returns_language_and_claims():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="hi", claims=["WhatsApp will start charging users"]
            )
        }
    )
    result = extract_intake("कल से WhatsApp पैसे लेगा", client=fake)
    assert result.language == "hi"
    assert result.claims == ["WhatsApp will start charging users"]


def test_extract_intake_truncates_to_max_claims():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=[f"claim {i}" for i in range(10)]
            )
        }
    )
    result = extract_intake("many claims", client=fake)
    assert len(result.claims) == MAX_CLAIMS


def test_extract_intake_tracks_usage():
    fake = FakeClient(
        parsed_by_model={IntakeResult: IntakeResult(language="en", claims=[])}
    )
    tracker = UsageTracker()
    extract_intake("hello", client=fake, tracker=tracker)
    assert tracker.total_tokens == 150
