import base64

import pytest

from config import UsageTracker
from media.models import VisionResult
from media.vision import describe_image, read_image
from tests.fakes import FakeClient


def _vision():
    return VisionResult(
        scene_description="A flooded street.",
        visible_text="",
        people_places_objects=["flooded street", "rescue boat"],
        claims_from_image=["The photo shows a flooded street."],
        language="en",
        image_role="viral_claim",
    )


def test_describe_image_sends_data_url_and_parses():
    fake = FakeClient(parsed_by_model={VisionResult: _vision()})
    result = describe_image(b"fake-bytes", client=fake)
    assert result.scene_description == "A flooded street."
    sent = fake.responses.calls[0][1]["input"][0]["content"]
    kinds = {block["type"] for block in sent}
    assert {"input_text", "input_image"} <= kinds
    image_block = next(b for b in sent if b["type"] == "input_image")
    assert image_block["image_url"] == (
        "data:image/jpeg;base64," + base64.b64encode(b"fake-bytes").decode()
    )


def test_describe_image_tracks_vision_stage():
    fake = FakeClient(parsed_by_model={VisionResult: _vision()})
    tracker = UsageTracker()
    describe_image(b"x", client=fake, tracker=tracker)
    assert [line.split(":")[0].strip() for line in tracker.breakdown()] == ["vision"]


def test_read_image_validates_file(tmp_path):
    target = tmp_path / "a.jpg"
    target.write_bytes(b"123")
    assert read_image(target) == (b"123", "image/jpeg")
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")


def test_read_image_rejects_oversize(tmp_path, monkeypatch):
    import media.vision as vision_module

    monkeypatch.setattr(vision_module, "MAX_IMAGE_BYTES", 2)
    target = tmp_path / "big.png"
    target.write_bytes(b"123")
    with pytest.raises(ValueError, match="exceeds"):
        read_image(target)


def test_describe_image_includes_claim_context():
    fake = FakeClient(parsed_by_model={VisionResult: _vision()})
    describe_image(b"x", client=fake, context="floods hit Patna today")
    sent = fake.responses.calls[0][1]["input"][0]["content"]
    text = next(b for b in sent if b["type"] == "input_text")["text"]
    assert "Claim context: floods hit Patna today" in text


def test_describe_image_defaults_to_unclear_without_context():
    fake = FakeClient(parsed_by_model={VisionResult: _vision()})
    describe_image(b"x", client=fake)
    sent = fake.responses.calls[0][1]["input"][0]["content"]
    text = next(b for b in sent if b["type"] == "input_text")["text"]
    assert "Claim context" not in text
    assert _vision().shows_claimed_event == "unclear"


def _vote(showing, reason="r"):
    return _vision().model_copy(
        update={"shows_claimed_event": showing, "mismatch_reason": reason}
    )


def test_describe_image_votes_showing_with_context():
    from config import UsageTracker

    fake = FakeClient(
        parse_queue=[
            _vote("no", "first reason"),
            _vote("yes"),
            _vote("no", "second reason"),
        ]
    )
    tracker = UsageTracker()
    result = describe_image(b"x", client=fake, context="floods", tracker=tracker)
    assert result.shows_claimed_event == "no"
    assert result.mismatch_reason == "first reason"
    assert result.consistency_agreement == 2
    assert "consistency votes: 2/3 no." in tracker.notes
    assert len(fake.responses.calls) == 3


def test_describe_image_tie_is_unclear():
    fake = FakeClient(
        parse_queue=[_vote("no", "r"), _vote("yes"), _vote("unclear")]
    )
    result = describe_image(b"x", client=fake, context="floods")
    assert result.shows_claimed_event == "unclear"
    assert result.mismatch_reason == ""
    assert result.consistency_agreement == 1


def test_describe_image_single_call_without_context():
    fake = FakeClient(parsed_by_model={VisionResult: _vision()})
    result = describe_image(b"x", client=fake)
    assert len(fake.responses.calls) == 1
    assert result.consistency_agreement == 0
