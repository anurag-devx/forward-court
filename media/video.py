"""Video understanding for video forwards (Files API + structured output).

Local videos are uploaded once via the Files API (purpose=user_data) and
referenced by file_id in an input_file block, per Meta's file-handling docs.
The upload is deleted after understanding so team storage doesn't grow.
Key frames come from local ffmpeg scene-change detection (max MAX_FRAMES).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from config import agent_model

from .models import VideoResult
from .vision import CONSISTENCY_VOTES, majority_showing

MAX_VIDEO_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 3
SCENE_THRESHOLD = 0.4
UPLOAD_PURPOSE = "user_data"
FFMPEG_TIMEOUT = 120

INSTRUCTIONS = (
    "You read a possibly viral video forward. Summarize the scene factually, "
    "transcribe any visible text verbatim, and briefly note what the "
    "soundtrack seems to claim. Name the main depicted people, places, and "
    "objects as short noun phrases. List the checkable claims the video "
    "itself makes about what, where, or when it shows. "
    "Report the ISO 639-1 language of visible text, or 'en' when there is none. "
    "When claim context is given, judge whether the video shows the claimed "
    "event: 'yes', 'no' (with a short mismatch_reason), or 'unclear'. "
    "With no claim context, shows_claimed_event is 'unclear'. "
    "Never guess the video's origin, date, or filmer."
)


def read_video(path: str | Path) -> tuple[bytes, str]:
    """Read a local MP4 file. Returns (bytes, path string)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if file_path.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 videos are supported: {path}")
    data = file_path.read_bytes()
    if len(data) > MAX_VIDEO_BYTES:
        raise ValueError(
            f"Video exceeds WhatsApp's {MAX_VIDEO_BYTES // (1024 * 1024)}MB "
            f"forward limit: {path}"
        )
    return data, str(file_path)


def _upload(client, video_bytes: bytes) -> str:
    """Upload video bytes; returns the file_id. Raises on failure."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as handle:
            uploaded = client.files.create(file=handle, purpose=UPLOAD_PURPOSE)
    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise RuntimeError("Files API upload returned no file id.")
    return file_id


def _parse_video(client, tracker, prompt: str, file_id: str) -> VideoResult:
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {"type": "input_file", "file_id": file_id},
                ],
            }
        ],
        text_format=VideoResult,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="video")
    return response.output_parsed


def describe_video(
    video_bytes: bytes, *, client, tracker=None, context: str | None = None
) -> VideoResult:
    """Run video understanding on uploaded bytes. Deletes the upload after.

    Uploads once; with claim context, the showing judgment is voted 3x
    (tie: unclear) and other fields come from the first vote.
    """
    if tracker is not None:
        tracker.mark("video")
    file_id = _upload(client, video_bytes)
    prompt = "Describe this forwarded video."
    if context and context.strip():
        prompt += f" Claim context: {context.strip()}"
    try:
        first = _parse_video(client, tracker, prompt, file_id)
        if not (context and context.strip()):
            return first
        votes = [first] + [
            _parse_video(client, tracker, prompt, file_id)
            for _ in range(CONSISTENCY_VOTES - 1)
        ]
    finally:
        try:
            client.files.delete(file_id)
        except Exception:
            if tracker is not None:
                tracker.note(f"uploaded video file not deleted: {file_id}.")
    value, agreement, reason = majority_showing(votes)
    if tracker is not None:
        tracker.note(f"consistency votes: {agreement}/3 {value}.")
    return first.model_copy(
        update={
            "shows_claimed_event": value,
            "mismatch_reason": reason,
            "consistency_agreement": agreement,
        }
    )


def extract_key_frames(path: str | Path, max_frames: int = MAX_FRAMES) -> list[bytes]:
    """Scene-change key frames as JPEG bytes. [] when ffmpeg is missing/fails."""
    if shutil.which("ffmpeg") is None:
        return []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "frame-%02d.jpg")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(path),
                    "-vf", f"select='gt(scene,{SCENE_THRESHOLD})'",
                    "-frames:v", str(max_frames),
                    "-q:v", "3",
                    pattern,
                ],
                check=True,
                timeout=FFMPEG_TIMEOUT,
                capture_output=True,
            )
            frames = []
            for frame_path in sorted(Path(tmp).glob("frame-*.jpg")):
                frames.append(frame_path.read_bytes())
            return frames
    except (subprocess.SubprocessError, OSError):
        return []
