import io
from types import SimpleNamespace

from PIL import Image

import main
from court.models import Argument, ArgPoint, Verdict
from evidence.relevance import RelevanceScore, RelevanceScores
from evidence.reverse_image import ReverseMatch
from intake.models import IntakeResult
from media.models import VideoResult
from tests.fakes import FakeClient, dead_http_client, search_call


def _frame_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def _sam_output():
    return [
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(
                    type="output_text",
                    text="<0f>1<|box;x1=10;y1=10;x2=60;y2=60;w=200;h=100;c=0.9|>",
                )
            ],
        )
    ]


def _video():
    return VideoResult(
        scene_summary="A flooded street.",
        visible_text="breaking",
        spoken_claims_hint="floods",
        people_places_objects=["flooded street"],
        claims_from_video=["video claim one", "video claim two"],
        language="en",
    )


def _client(frames=2):
    return FakeClient(
        parsed_by_model={
            VideoResult: _video(),
            IntakeResult: IntakeResult(
                language="en",
                claims=["spoken claim", "caption claim"],
                presented_as_recent=True,
            ),
            Argument: Argument(
                points=[ArgPoint(text="p", source_url="https://g.com/0")]
            ),
            RelevanceScores: RelevanceScores(
                scores=[RelevanceScore(index=i, score=2) for i in range(12)]
            ),
            Verdict: Verdict(
                verdict="False",
                reason="Debunked.",
                key_source="https://g.com/0",
                language="en",
                confidence="high",
                claim_timeframe=None,
            ),
        },
        outputs=[_sam_output() for _ in range(frames)]
        + [
            [search_call([(f"https://g.com/{i}", "T", "snippet")])]
            for i in range(6)
        ]
        + [[search_call([])]] * 3,
    )


class _FakeProvider:
    def __init__(self, matches):
        self.matches = matches

    def search(self, image):
        return list(self.matches)


def _clip(tmp_path):
    clip = tmp_path / "fwd.mp4"
    clip.write_bytes(b"fake-mp4-bytes")
    return str(clip)


def test_full_video_pipeline_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main, "extract_key_frames", lambda path: [_frame_bytes(), _frame_bytes()]
    )
    monkeypatch.setattr(main, "extract_audio", lambda path: b"fake-wav")
    monkeypatch.setattr(
        main, "transcribe_audio", lambda audio, **kwargs: "the river flooded"
    )
    provider = _FakeProvider(
        [ReverseMatch(source_url="https://old.com/v", page_title="old", exact=True)]
    )
    card, tracker = main.run_video_forward(
        _clip(tmp_path),
        "floods today",
        client=_client(),
        http_client=dead_http_client(),
        provider=provider,
    )
    assert "FALSE" in card.splitlines()[0]
    assert "key frames extracted: 2." in tracker.notes
    assert "SerpAPI image searches: 4." in tracker.notes
    stages = [line.split(":")[0].strip() for line in tracker.breakdown()]
    assert stages[0] == "video"
    assert "segment" in stages


def test_claims_merge_video_first_truncated_to_3(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "extract_key_frames", lambda path: [])
    monkeypatch.setattr(main, "extract_audio", lambda path: b"fake-wav")
    monkeypatch.setattr(
        main, "transcribe_audio", lambda audio, **kwargs: "the river flooded"
    )
    client = _client(frames=0)
    main.run_video_forward(
        _clip(tmp_path),
        "floods today",
        client=client,
        http_client=dead_http_client(),
        provider=_FakeProvider([]),
    )
    grounding_inputs = [
        kwargs["input"]
        for _, kwargs in client.responses.calls
        if "tools" in kwargs and not kwargs["input"].startswith("fact check:")
    ]
    assert len(grounding_inputs) == 6  # 3 merged claims x support/refute
    assert "video claim one" in grounding_inputs[0]
    assert "caption claim" not in "".join(grounding_inputs)


def test_no_audio_skips_transcript(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "extract_key_frames", lambda path: [])
    monkeypatch.setattr(main, "extract_audio", lambda path: None)
    _, tracker = main.run_video_forward(
        _clip(tmp_path),
        None,
        client=_client(frames=0),
        http_client=dead_http_client(),
        provider=_FakeProvider([]),
    )
    assert "no usable audio track: transcript skipped." in tracker.notes
    assert "transcribe" not in "".join(tracker.breakdown())


def test_transcribe_error_continues_without_transcript(tmp_path, monkeypatch):
    from media.audio import TranscribeError

    def _boom(audio, **kwargs):
        raise TranscribeError("endpoint down")

    monkeypatch.setattr(main, "extract_key_frames", lambda path: [])
    monkeypatch.setattr(main, "extract_audio", lambda path: b"fake-wav")
    monkeypatch.setattr(main, "transcribe_audio", _boom)
    card, tracker = main.run_video_forward(
        _clip(tmp_path),
        None,
        client=_client(frames=0),
        http_client=dead_http_client(),
        provider=_FakeProvider([]),
    )
    assert "UNVERIFIED" in card.splitlines()[0]
    assert any("transcription unavailable" in note for note in tracker.notes)


def test_cli_rejects_video_combos(capsys):
    assert main.main(["hello", "--video", "x.mp4"]) == 2
    assert "only one" in capsys.readouterr().err
    assert main.main(["--image", "a.jpg", "--video", "x.mp4"]) == 2
    assert "only one" in capsys.readouterr().err


def test_cli_rejects_oversize_video(tmp_path, capsys):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    assert main.main(["--video", str(big)]) == 2
    assert "16MB" in capsys.readouterr().err
