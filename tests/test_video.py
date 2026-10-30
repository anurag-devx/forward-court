import shutil
import subprocess

import pytest

from media.models import VideoResult
from media.video import describe_video, extract_key_frames, read_video
from tests.fakes import FakeClient

ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _video_result():
    return VideoResult(
        scene_summary="A street.",
        visible_text="",
        spoken_claims_hint="",
        people_places_objects=["street"],
        claims_from_video=["video claim"],
        language="en",
    )


def test_read_video_rejects_missing_wrong_suffix_and_oversize(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video(tmp_path / "nope.mp4")
    avi = tmp_path / "clip.avi"
    avi.write_bytes(b"data")
    with pytest.raises(ValueError, match="Only .mp4"):
        read_video(avi)
    big = tmp_path / "big.mp4"
    big.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="16MB"):
        read_video(big)


def test_read_video_ok(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-mp4-bytes")
    data, resolved = read_video(clip)
    assert data == b"fake-mp4-bytes"
    assert resolved == str(clip)


def test_describe_video_uploads_references_and_deletes():
    client = FakeClient(parsed_by_model={VideoResult: _video_result()})
    result = describe_video(b"bytes", client=client)
    assert result.claims_from_video == ["video claim"]
    assert client.files.calls[0][0] == "create"
    assert client.files.calls[0][1]["purpose"] == "user_data"
    sent = client.responses.calls[0][1]
    blocks = sent["input"][0]["content"]
    assert {"type": "input_file", "file_id": "file-test-123"} in blocks
    assert client.files.deleted == ["file-test-123"]


def test_describe_video_upload_failure_raises():
    client = FakeClient(parsed_by_model={VideoResult: _video_result()})
    client.files.file_id = ""
    with pytest.raises(RuntimeError, match="no file id"):
        describe_video(b"bytes", client=client)


def test_describe_video_includes_claim_context():
    client = FakeClient(parsed_by_model={VideoResult: _video_result()})
    describe_video(b"bytes", client=client, context="bridge collapsed today")
    sent = client.responses.calls[0][1]["input"][0]["content"]
    text = next(b for b in sent if b["type"] == "input_text")["text"]
    assert "Claim context: bridge collapsed today" in text


@ffmpeg
def test_key_frames_from_changing_clip_capped(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=red:s=160x120:d=0.5",
            "-f", "lavfi", "-i", "color=blue:s=160x120:d=0.5",
            "-f", "lavfi", "-i", "color=green:s=160x120:d=0.5",
            "-filter_complex", "[0][1][2]concat=n=3",
            str(clip),
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    frames = extract_key_frames(clip)
    assert 1 <= len(frames) <= 3
    assert all(frame[:2] == b"\xff\xd8" for frame in frames)


@ffmpeg
def test_key_frames_missing_file_returns_empty(tmp_path):
    assert extract_key_frames(tmp_path / "nope.mp4") == []
