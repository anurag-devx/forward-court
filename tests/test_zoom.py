import io
from types import SimpleNamespace

import pytest
from PIL import Image

import main
from config import SEGMENTATION_MODEL, UsageTracker
from court.models import Argument, ArgPoint, Verdict
from evidence.relevance import RelevanceScore, RelevanceScores
from intake.models import IntakeResult
from links.scamtext import ScamReading
from media.models import VideoResult, VisionResult, ZoomReading
from media.segment import (
    TEXT_PROMPTS,
    rank_regions,
    tight_crops,
    zoom_crops,
)
from media.video import merge_video_zoom
from media.vision import merge_zoom, merge_zoom_fields, zoom_views
from tests.fakes import FakeClient, dead_http_client, search_call


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


def _zoom_reading(**overrides):
    base = {
        "visible_text": "zoom line",
        "visible_domain": None,
        "domain_from_text_crop": False,
        "people_places_objects": ["zoom subject"],
    }
    base.update(overrides)
    return ZoomReading(**base)


# --- ranking ---


def test_rank_regions_tags_extra_prompts():
    fake = FakeClient(
        outputs=[
            _message_output(_sam_text([(10, 10, 60, 60, 0.9)])),
            _message_output(_sam_text([(70, 10, 120, 60, 0.99)])),
        ]
    )
    ranked = rank_regions(
        _fixture(), ["boat"], ["url bar"], client=fake
    )
    assert [(r.prompt, r.extra) for r in ranked] == [
        ("url bar", True),
        ("boat", False),
    ]
    assert [r.confidence for r in ranked] == [0.99, 0.9]
    sent = [
        kwargs["input"][0]["content"][0]["text"]
        for verb, kwargs in fake.responses.calls
        if verb == "create"
    ]
    assert sent == ["boat", "url bar"]
    assert set(TEXT_PROMPTS) == {"text", "logo", "url bar"}


def test_rank_regions_failure_yields_empty():
    class Exploding:
        def create(self, **kwargs):
            raise RuntimeError("sam down")

    class Client:
        responses = Exploding()

    assert rank_regions(_fixture(), ["boat"], client=Client()) == []


# --- crops ---


def test_tight_crops_exclude_extra_regions():
    fake = FakeClient(
        outputs=[
            _message_output(_sam_text([(10, 10, 60, 60, 0.5)])),
            _message_output(_sam_text([(70, 10, 120, 60, 0.99)])),
        ]
    )
    ranked = rank_regions(
        _fixture(), ["boat"], ["url bar"], client=fake
    )
    crops = tight_crops(_fixture(), ranked)
    assert [c.subject for c in crops] == ["boat"]
    assert Image.open(io.BytesIO(crops[0].data)).size == (50, 50)


def test_zoom_crops_pad_and_upscale_small():
    fake = FakeClient(
        outputs=[
            _message_output(_sam_text([(10, 10, 60, 60, 0.9)])),
        ]
    )
    ranked = rank_regions(_fixture(), ["boat"], client=fake)
    crops = zoom_crops(_fixture(), ranked)
    assert len(crops) == 1
    # 50x50 box + 15% pad = 64x64, upscaled: min dim 800
    assert Image.open(io.BytesIO(crops[0].data)).size == (800, 800)


def test_zoom_crops_keep_large_regions():
    fake = FakeClient(
        outputs=[
            _message_output(_sam_text([(0, 0, 200, 100, 0.9)])),
        ]
    )
    ranked = rank_regions(_fixture(900, 620), ["page"], client=fake)
    crops = zoom_crops(_fixture(900, 620), ranked)
    assert Image.open(io.BytesIO(crops[0].data)).size == (900, 620)


def test_crops_empty_on_bad_bytes_or_empty_ranking():
    assert zoom_crops(b"nope", []) == []
    assert tight_crops(b"nope", []) == []


# --- zoom call ---


def test_zoom_views_sends_full_plus_crops_once():
    fake = FakeClient(parsed_by_model={ZoomReading: _zoom_reading()})
    reading = zoom_views(
        _fixture(), [_fixture(50, 50), _fixture(60, 60)], client=fake
    )
    assert reading is not None and reading.visible_text == "zoom line"
    parses = [c for c in fake.responses.calls if c[0] == "parse"]
    assert len(parses) == 1
    content = parses[0][1]["input"][0]["content"]
    images = [part for part in content if part["type"] == "input_image"]
    assert len(images) == 3  # full + 2 crops
    assert parses[0][1]["text_format"] is ZoomReading


def test_zoom_views_none_without_crops_or_on_error():
    fake = FakeClient(parsed_by_model={ZoomReading: _zoom_reading()})
    assert zoom_views(_fixture(), [], client=fake) is None

    class Exploding:
        def parse(self, **kwargs):
            raise RuntimeError("vision down")

    class Client:
        responses = Exploding()

    assert (
        zoom_views(_fixture(), [_fixture(10, 10)], client=Client()) is None
    )


# --- merge ---


def _vision(**overrides):
    base = {
        "scene_description": "s",
        "visible_text": "base line",
        "people_places_objects": ["base subject"],
        "claims_from_image": ["c"],
        "language": "en",
        "image_role": "viral_claim",
    }
    base.update(overrides)
    return VisionResult(**base)


def test_merge_unions_text_and_subjects():
    merged, conflicted = merge_zoom(
        _vision(), _zoom_reading(visible_text="zoom line")
    )
    assert merged.visible_text == "base line\nzoom line"
    assert merged.people_places_objects == ["base subject", "zoom subject"]
    assert conflicted is False
    assert merged.claims_from_image == ["c"]  # claims untouched


def test_merge_dedupes_repeated_lines():
    merged, _ = merge_zoom(
        _vision(visible_text="same"), _zoom_reading(visible_text="same")
    )
    assert merged.visible_text == "same"


def test_merge_domain_zoom_only_fills():
    merged, conflicted = merge_zoom(
        _vision(visible_domain=None),
        _zoom_reading(visible_domain="Example.COM"),
    )
    assert merged.visible_domain == "example.com"
    assert conflicted is False


def test_merge_domain_agreement_and_missing():
    merged, conflicted = merge_zoom(
        _vision(visible_domain="a.example"),
        _zoom_reading(visible_domain="a.example"),
    )
    assert (merged.visible_domain, conflicted) == ("a.example", False)
    merged, conflicted = merge_zoom(
        _vision(visible_domain="a.example"), _zoom_reading()
    )
    assert (merged.visible_domain, conflicted) == ("a.example", False)
    merged, conflicted = merge_zoom(_vision(), None)
    assert (merged.visible_domain, conflicted) == (None, False)


def test_merge_domain_text_crop_wins():
    merged, conflicted = merge_zoom(
        _vision(visible_domain="pass.example"),
        _zoom_reading(
            visible_domain="zoom.example", domain_from_text_crop=True
        ),
    )
    assert (merged.visible_domain, conflicted) == ("zoom.example", False)


def test_merge_domain_conflict_keeps_pass1_and_flags():
    merged, conflicted = merge_zoom(
        _vision(visible_domain="pass.example"),
        _zoom_reading(visible_domain="zoom.example"),
    )
    assert (merged.visible_domain, conflicted) == ("pass.example", True)


def test_merge_video_zoom_same_rules():
    video = VideoResult(
        scene_summary="s",
        visible_text="base",
        spoken_claims_hint="",
        people_places_objects=[],
        claims_from_video=["c"],
        language="en",
        visible_domain="pass.example",
    )
    merged, conflicted = merge_video_zoom(
        video, _zoom_reading(visible_domain="zoom.example")
    )
    assert merged.visible_domain == "pass.example"
    assert conflicted is True
    assert "zoom line" in merged.visible_text


# --- pipeline ---


def _image_file(tmp_path):
    target = tmp_path / "fwd.jpg"
    Image.new("RGB", (200, 100), "blue").save(target, format="JPEG")
    return str(target)


def _sam_box_output():
    return [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="<0f>1<|box;x1=10;y1=10;x2=60;y2=60;"
                    "w=200;h=100;c=0.9|>",
                )
            ],
        )
    ]


def _zoom_client(zoom, sam_calls=4):
    return FakeClient(
        parsed_by_model={
            VisionResult: VisionResult(
                scene_description="A flooded street.",
                visible_text="",
                people_places_objects=["flooded street"],
                claims_from_image=["vision claim one"],
                language="en",
                image_role="viral_claim",
            ),
            IntakeResult: IntakeResult(
                language="en",
                claims=["caption claim one"],
                presented_as_recent=False,
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
            ZoomReading: zoom,
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2) for i in range(12)]
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/0")]
            ),
            Verdict: Verdict(
                verdict="True", reason="ok",
                key_source="https://g.com/0", language="en",
                confidence="high", claim_timeframe=None,
            ),
        },
        outputs=[_sam_box_output()] * sam_calls
        + [
            [search_call([(f"https://g.com/{i}", "T", "s")])]
            if i % 2 == 0
            else [search_call([])]
            for i in range(6)
        ],
        lookup_outputs=[[search_call([])]] * 3,
    )


def test_image_zoom_merges_and_sends_text_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOM", "on")
    client = _zoom_client(
        _zoom_reading(
            visible_text="url bar text",
            visible_domain="shot.example",
            domain_from_text_crop=True,
        )
    )
    card, tracker = main.run_image_forward(
        _image_file(tmp_path), "some caption",
        client=client, http_client=dead_http_client(),
    )
    assert "zoom merged 3 crop(s)." in tracker.notes
    prompts = [
        kwargs["input"][0]["content"][0]["text"]
        for verb, kwargs in client.responses.calls
        if verb == "create" and kwargs.get("model") == SEGMENTATION_MODEL
    ]
    assert prompts == ["flooded street", "text", "logo", "url bar"]
    assert "TRUE" in card.splitlines()[0]


def test_image_zoom_conflict_skips_impersonation(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOM", "on")
    client = _zoom_client(
        _zoom_reading(visible_domain="other.example")
    )
    client.responses.parsed_by_model[VisionResult] = VisionResult(
        scene_description="s",
        visible_text="",
        people_places_objects=["flooded street"],
        claims_from_image=["vision claim one"],
        visible_domain="abcnews.com.co",
        language="en",
        image_role="viral_claim",
    )
    _, tracker = main.run_image_forward(
        _image_file(tmp_path), "some caption",
        client=client, http_client=dead_http_client(),
    )
    assert "visible-domain conflict: impersonation check skipped." in (
        tracker.notes
    )
    assert not [
        note for note in tracker.notes if note.startswith("impersonated domain")
    ]


def test_image_zoom_off_skips_sam_text_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOM", "off")
    client = _zoom_client(_zoom_reading(), sam_calls=0)
    _, tracker = main.run_image_forward(
        _image_file(tmp_path), "some caption",
        client=client, http_client=dead_http_client(),
    )
    assert "zoom disabled (ZOOM=off)." in tracker.notes
    sam_prompts = [
        kwargs["input"][0]["content"][0]["text"]
        for verb, kwargs in client.responses.calls
        if verb == "create" and kwargs.get("model") == SEGMENTATION_MODEL
    ]
    assert sam_prompts == []


def test_video_zoom_runs_on_first_frame_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOOM", "on")
    frame = _fixture()
    monkeypatch.setattr(main, "extract_key_frames", lambda path: [frame, frame])
    monkeypatch.setattr(main, "extract_audio", lambda path: None)
    fake = FakeClient(
        parsed_by_model={
            VideoResult: VideoResult(
                scene_summary="A flooded street.",
                visible_text="",
                spoken_claims_hint="",
                people_places_objects=["flooded street"],
                claims_from_video=["video claim one"],
                language="en",
            ),
            ScamReading: ScamReading(scam_type="", signals=[]),
            ZoomReading: _zoom_reading(visible_text="frame text"),
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2) for i in range(12)]
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/0")]
            ),
            Verdict: Verdict(
                verdict="True", reason="ok",
                key_source="https://g.com/0", language="en",
                confidence="high", claim_timeframe=None,
            ),
        },
        outputs=[_sam_box_output()] * 2  # 1 subject x 2 frames, no captions
        + [
            [search_call([("https://g.com/0", "T", "s")])],
            [search_call([])],
        ],
        lookup_outputs=[[search_call([])]],
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-mp4")
    _, tracker = main.run_video_forward(
        str(video_path), client=fake, http_client=dead_http_client()
    )
    assert "zoom merged 1 crop(s)." in tracker.notes
    zoom_parses = [
        kwargs
        for verb, kwargs in fake.responses.calls
        if verb == "parse" and kwargs.get("text_format") is ZoomReading
    ]
    assert len(zoom_parses) == 1
