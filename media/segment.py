"""Subject segmentation with SAM 3.1 for reverse-search crops.

One Responses API call per subject noun phrase; records are ranked by
confidence and the top TOP_CROPS box regions are cropped with Pillow.
Crops use each mask's box (SAM boxes tightly locate the object), which keeps
this dependency-free. Any failure yields fewer crops -- never an exception.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

from PIL import Image, ImageOps

from config import SEGMENTATION_MODEL

MAX_SUBJECTS = 3
TOP_CROPS = 2

_BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([\d.]+))?\|>"
)


@dataclass
class Crop:
    data: bytes
    subject: str
    confidence: float


def _output_text(output) -> str:
    parts: list[str] = []
    for item in output or []:
        get = item.get if isinstance(item, dict) else lambda k: getattr(item, k, None)
        if get("type") != "message":
            continue
        for block in get("content") or []:
            bget = (
                block.get
                if isinstance(block, dict)
                else lambda k: getattr(block, k, None)
            )
            if bget("type") == "output_text":
                parts.append(bget("text") or "")
    return "".join(parts)


def _parse_boxes(text: str) -> list[tuple[float, tuple[int, int, int, int, int, int]]]:
    boxes = []
    for match in _BOX_RE.finditer(text or ""):
        x1, y1, x2, y2, w, h = (int(g) for g in match.groups()[:6])
        conf = float(match.group(7)) if match.group(7) is not None else 0.0
        if x2 > x1 and y2 > y1 and w > 0 and h > 0:
            boxes.append((conf, (x1, y1, x2, y2, w, h)))
    return boxes


def _crop_box(img: Image.Image, box: tuple[int, ...]) -> bytes | None:
    x1, y1, x2, y2, w, h = box
    sx, sy = img.width / w, img.height / h
    region = img.crop(
        (
            max(0, int(x1 * sx)),
            max(0, int(y1 * sy)),
            min(img.width, int(x2 * sx)),
            min(img.height, int(y2 * sy)),
        )
    )
    if region.width <= 0 or region.height <= 0:
        return None
    buffer = io.BytesIO()
    region.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def segment_subjects(
    image_bytes: bytes,
    subjects: list[str],
    *,
    mime: str = "image/jpeg",
    client,
    tracker=None,
) -> list[Crop]:
    """Crop the top-2 confident subject regions. Returns [] on any failure."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    except Exception:
        return []
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    ranked: list[tuple[float, tuple[int, ...], str]] = []
    for subject in subjects[:MAX_SUBJECTS]:
        try:
            response = client.responses.create(
                model=SEGMENTATION_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": subject},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
            )
        except Exception:
            continue
        if tracker is not None:
            tracker.add(response.usage, stage="segment")
        for conf, box in _parse_boxes(_output_text(response.output)):
            ranked.append((conf, box, subject))
    ranked.sort(key=lambda item: item[0], reverse=True)
    crops: list[Crop] = []
    for conf, box, subject in ranked[:TOP_CROPS]:
        data = _crop_box(img, box)
        if data is not None:
            crops.append(Crop(data=data, subject=subject, confidence=conf))
    return crops
