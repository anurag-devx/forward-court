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
