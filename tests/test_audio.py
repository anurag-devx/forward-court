import shutil
import subprocess

import httpx
import pytest

from config import TRANSCRIBE_MODEL, UsageTracker
from media.audio import TranscribeError, extract_audio, transcribe_audio

ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@ffmpeg
def test_extract_audio_from_clip_with_tone(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=s=160x120:d=0.5",
            "-f", "lavfi", "-i", "sine=frequency=440:d=0.5",
            "-shortest",
            str(clip),
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    data = extract_audio(clip)
    assert data and data[:4] == b"RIFF"  # WAV container
    assert len(data) > 100


@ffmpeg
def test_extract_audio_missing_file_returns_none(tmp_path):
    assert extract_audio(tmp_path / "nope.mp4") is None


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_transcribe_posts_multipart_and_returns_transcript(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["accept"] = request.headers.get("accept")
        seen["body"] = request.read().decode("latin1")
        return httpx.Response(
            200,
            json={
                "transcript": "markets closed today",
                "audioDurationMs": 500,
                "sessionId": "s1",
                "turns": [],
            },
        )

    tracker = UsageTracker()
    transcript = transcribe_audio(
        b"fake-wav", http_client=_client(handler), tracker=tracker
    )
    assert transcript == "markets closed today"
    assert seen["auth"] == "Bearer test-key"
    assert seen["accept"] == "application/json"
    assert TRANSCRIBE_MODEL in seen["body"]
    assert "PUSH_TO_TALK" in seen["body"]
    assert "transcribed 500ms of audio." in tracker.notes


def test_transcribe_missing_key_raises_without_calling(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    def handler(request):
        raise AssertionError("must not call the API without a key")

    with pytest.raises(TranscribeError, match="MODEL_API_KEY is not set"):
        transcribe_audio(b"fake-wav", http_client=_client(handler))


def test_transcribe_http_error_raises(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(500, json={"message": "boom"})

    with pytest.raises(TranscribeError, match="transcription request failed"):
        transcribe_audio(b"fake-wav", http_client=_client(handler))
