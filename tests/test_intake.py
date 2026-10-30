from config import MAX_CLAIMS, UsageTracker
from intake.extractor import extract_intake
from intake.models import IntakeResult
from tests.fakes import FakeClient


def test_extract_intake_returns_language_and_claims():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="hi",
                claims=["WhatsApp will start charging users"],
                presented_as_recent=True,
            )
        }
    )
    result = extract_intake("कल से WhatsApp पैसे लेगा", client=fake)
    assert result.language == "hi"
    assert result.claims == ["WhatsApp will start charging users"]
    assert result.presented_as_recent is True


def test_extract_intake_truncates_to_max_claims():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en",
                claims=[f"claim {i}" for i in range(10)],
                presented_as_recent=False,
            )
        }
    )
    result = extract_intake("many claims", client=fake)
    assert len(result.claims) == MAX_CLAIMS


def test_extract_intake_tracks_usage():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=[], presented_as_recent=False
            )
        }
    )
    tracker = UsageTracker()
    extract_intake("hello", client=fake, tracker=tracker)
    assert tracker.total_tokens == 150


def test_extract_intake_splits_event_and_media_claims():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en",
                claims=["floods hit Patna", "video shows it", "bridge fell"],
                event_claims=["floods hit Patna", "bridge fell"],
                media_claims=["video shows it"],
                presented_as_recent=True,
            )
        }
    )
    result = extract_intake("floods video", client=fake)
    assert result.event_claims == ["floods hit Patna", "bridge fell"]
    assert result.media_claims == ["video shows it"]


def test_extract_intake_partitions_follow_truncation():
    fake = FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en",
                claims=[f"claim {i}" for i in range(10)],
                event_claims=[f"claim {i}" for i in range(10)],
                media_claims=["claim 9"],
                presented_as_recent=False,
            )
        }
    )
    result = extract_intake("many claims", client=fake)
    assert result.event_claims == [f"claim {i}" for i in range(MAX_CLAIMS)]
    assert result.media_claims == []
