"""Audio extraction + speech transcription for video forwards.

ffmpeg extracts mono 16kHz WAV (the endpoint requires WAV, max 10 minutes);
muse-voice-transcribe-1.0 transcribes it via POST /asr/transcribe
(multipart `request` JSON + `audio` clip, per Meta's voice API docs).
The endpoint returns no language; the pipeline detects it via intake on
the transcript. Transcription failures raise TranscribeError so the
pipeline can continue on video understanding alone.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from config import API_KEY_ENV, BASE_URL, TRANSCRIBE_MODEL

FFMPEG_TIMEOUT = 120
TRANSCRIBE_TIMEOUT = 120.0
MAX_AUDIO_SECONDS = 600
TRANSCRIBE_URL = f"{BASE_URL}/asr/transcribe"


class TranscribeError(RuntimeError):
    """Speech transcription failed. The pipeline continues without it."""


def extract_audio(path: str | Path) -> bytes | None:
    """Extract mono 16kHz WAV (capped at 10 min). None when unavailable."""
    if shutil.which("ffmpeg") is None:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "audio.wav")
            completed = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(path),
                    "-vn", "-ac", "1", "-ar", "16000",
                    "-t", str(MAX_AUDIO_SECONDS),
                    "-c:a", "pcm_s16le",
                    target,
                ],
                timeout=FFMPEG_TIMEOUT,
                capture_output=True,
            )
            if completed.returncode != 0:
                return None
            data = Path(target).read_bytes()
            return data or None
    except (subprocess.SubprocessError, OSError):
        return None


def transcribe_audio(
    audio_bytes: bytes, *, http_client=None, tracker=None
) -> str:
    """Transcribe WAV bytes via /asr/transcribe. Returns the transcript."""
    if tracker is not None:
        tracker.mark("transcribe")
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise TranscribeError(
            f"{API_KEY_ENV} is not set. Put it in .env and retry."
        )
    request = {
        "model": TRANSCRIBE_MODEL,
        "audioEncoding": "WAV",
        "mode": "PUSH_TO_TALK",
    }
    files = {
        "request": (None, json.dumps(request), "application/json"),
        "audio": ("audio.wav", audio_bytes, "audio/wav"),
    }
    try:
        if http_client is None:
            with httpx.Client() as owned:
                response = owned.post(
                    TRANSCRIBE_URL,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    },
                    files=files,
                    timeout=TRANSCRIBE_TIMEOUT,
                )
        else:
            response = http_client.post(
                TRANSCRIBE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
                files=files,
                timeout=TRANSCRIBE_TIMEOUT,
            )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise TranscribeError(f"transcription request failed: {exc}") from exc
    transcript = payload.get("transcript", "") if isinstance(payload, dict) else ""
    if tracker is not None:
        duration = payload.get("audioDurationMs", "?") if isinstance(payload, dict) else "?"
        tracker.note(f"transcribed {duration}ms of audio.")
    return transcript or ""
