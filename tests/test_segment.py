import io
from types import SimpleNamespace

from PIL import Image

from config import UsageTracker
from media.segment import segment_subjects
from tests.fakes import FakeClient, FakeUsage


def _fixture(width=200, height=100):
    img = Image.new("RGB", (width, height), "blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def _message_output(text):
    return [
        SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=text)],
        )
    ]


def _sam_text(records):
    return "<0f>" + "".join(
        f"{i}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=200;h=100;c={c}|>"
        for i, (x1, y1, x2, y2, c) in enumerate(records, start=1)
    )


def test_crops_top2_by_confidence():
    fake = FakeClient(
        outputs=[
            _message_output(_sam_text([(10, 10, 60, 60, 0.9), (70, 10, 120, 60, 0.5)])),
            _message_output(_sam_text([(130, 10, 190, 60, 0.99)])),
        ]
    )
    crops = segment_subjects(_fixture(), ["boat", "bridge"], client=fake)
    assert [c.subject for c in crops] == ["bridge", "boat"]
    assert [c.confidence for c in crops] == [0.99, 0.9]
    sizes = [Image.open(io.BytesIO(c.data)).size for c in crops]
    assert sizes == [(60, 50), (50, 50)]


def test_sam_failure_yields_no_crops():
    class Exploding:
        def create(self, **kwargs):
            raise ConnectionError("down")

    client = SimpleNamespace(responses=Exploding())
    assert segment_subjects(_fixture(), ["boat"], client=client) == []


def test_unparsable_output_yields_no_crops():
    fake = FakeClient(outputs=[_message_output("no boxes here")])
    assert segment_subjects(_fixture(), ["boat"], client=fake) == []


def test_out_of_range_boxes_are_clamped():
    fake = FakeClient(
        outputs=[_message_output(_sam_text([(180, 80, 500, 400, 0.9)]))]
    )
    crops = segment_subjects(_fixture(), ["boat"], client=fake)
    assert Image.open(io.BytesIO(crops[0].data)).size == (20, 20)


def test_tracks_segment_stage():
    fake = FakeClient(outputs=[_message_output(_sam_text([(1, 1, 9, 9, 0.5)]))])
    tracker = UsageTracker()
    segment_subjects(_fixture(), ["boat"], client=fake, tracker=tracker)
    assert tracker.total_tokens == FakeUsage.total_tokens
    assert len(tracker.breakdown()) == 1
