"""Image understanding for image forwards (structured output)."""
from __future__ import annotations

import base64
from pathlib import Path

from config import agent_model

from .models import VisionResult

MAX_IMAGE_BYTES = 10 * 1024 * 1024

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

INSTRUCTIONS = (
    "You read a possibly viral image forward. Describe the scene factually, "
    "transcribe any visible text verbatim, and name the main depicted people, "
    "places, and objects as short noun phrases. List the checkable claims the "
    "image itself makes about what, where, or when it shows. "
    "Report the ISO 639-1 language of visible text, or 'en' when there is none. "
    "Never guess the image's origin, date, or photographer."
)


def read_image(path: str | Path) -> tuple[bytes, str]:
    """Read a local image file. Returns (bytes, mime type)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data = file_path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)}MB: {path}"
        )
    mime = _MIME_BY_SUFFIX.get(file_path.suffix.lower(), "image/jpeg")
    return data, mime


def describe_image(
    image_bytes: bytes, *, mime: str = "image/jpeg", client, tracker=None
) -> VisionResult:
    """Run image understanding. Local bytes are sent as a base64 data URL."""
    data_url = (
        f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    )
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Describe this forwarded image.",
                    },
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text_format=VisionResult,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="vision")
    return response.output_parsed
