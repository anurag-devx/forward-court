"""Subject segmentation with SAM 3.1 for reverse-search and zoom crops.

One Responses API call per prompt phrase; records are ranked by confidence.
Reverse search takes the top TOP_CROPS tight box regions; the zoom pass takes
the top TOP_ZOOM_CROPS with padding and upscaling so small text stays legible.
Any failure yields fewer crops -- never an exception.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

from PIL import Image, ImageOps

from config import SEGMENTATION_MODEL
from evidence.responses import output_text

MAX_SUBJECTS = 3
TOP_CROPS = 2
TOP_ZOOM_CROPS = 3
TEXT_PROMPTS = ("text", "logo", "url bar")
ZOOM_PAD_FRACTION = 0.15
ZOOM_MIN_DIM = 800
ZOOM_UPSCALE_BELOW = 400

_BOX_RE = re.compile(
    r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([\d.]+))?\|>"
)


@dataclass
class Crop:
    data: bytes
    subject: str
    confidence: float


@dataclass
class RankedRegion:
    confidence: float
    box: tuple[int, ...]
    prompt: str
    extra: bool = False


def _parse_boxes(text: str) -> list[tuple[float, tuple[int, int, int, int, int, int]]]:
    boxes = []
    for match in _BOX_RE.finditer(text or ""):
        x1, y1, x2, y2, w, h = (int(g) for g in match.groups()[:6])
        conf = float(match.group(7)) if match.group(7) is not None else 0.0
        if x2 > x1 and y2 > y1 and w > 0 and h > 0:
            boxes.append((conf, (x1, y1, x2, y2, w, h)))
    return boxes


def _pixels(img: Image.Image, box: tuple[int, ...]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2, w, h = box
    sx, sy = img.width / w, img.height / h
    return (
        max(0, int(x1 * sx)),
        max(0, int(y1 * sy)),
        min(img.width, int(x2 * sx)),
        min(img.height, int(y2 * sy)),
    )


def _encode(region: Image.Image) -> bytes | None:
    if region.width <= 0 or region.height <= 0:
        return None
    buffer = io.BytesIO()
    region.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _crop_box(img: Image.Image, box: tuple[int, ...]) -> bytes | None:
    return _encode(img.crop(_pixels(img, box)))


def _zoom_box(img: Image.Image, box: tuple[int, ...]) -> bytes | None:
    """Padded crop, upscaled when small so text stays legible."""
    x1, y1, x2, y2 = _pixels(img, box)
    pad = int(max(x2 - x1, y2 - y1) * ZOOM_PAD_FRACTION)
    region = img.crop(
        (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(img.width, x2 + pad),
            min(img.height, y2 + pad),
        )
    )
    if region.width <= 0 or region.height <= 0:
        return None
    small = min(region.width, region.height)
    if small < ZOOM_UPSCALE_BELOW:
        scale = ZOOM_MIN_DIM / small
        region = region.resize(
            (int(region.width * scale), int(region.height * scale)),
            Image.LANCZOS,
        )
    return _encode(region)


def rank_regions(
    image_bytes: bytes,
    subjects: list[str],
    extra_prompts: list[str] | tuple[str, ...] = (),
    *,
    mime: str = "image/jpeg",
    client,
    tracker=None,
) -> list[RankedRegion]:
    """Rank SAM boxes over subject + extra prompts. [] on any failure."""
    if tracker is not None:
        tracker.mark("segment")
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    ranked: list[RankedRegion] = []
    prompts = [(subject, False) for subject in subjects[:MAX_SUBJECTS]]
    prompts += [(prompt, True) for prompt in extra_prompts]
    for prompt, is_extra in prompts:
        try:
            response = client.responses.create(
                model=SEGMENTATION_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
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
        for conf, box in _parse_boxes(output_text(response.output)):
            ranked.append(
                RankedRegion(
                    confidence=conf, box=box, prompt=prompt, extra=is_extra
                )
            )
    ranked.sort(key=lambda region: region.confidence, reverse=True)
    return ranked


def _open(image_bytes: bytes) -> Image.Image | None:
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    except Exception:
        return None


def tight_crops(
    image_bytes: bytes,
    ranked: list[RankedRegion],
    top_n: int = TOP_CROPS,
) -> list[Crop]:
    """Top tight subject crops from a ranking (reverse-search inputs).

    Extra-prompt (text/logo) regions are excluded: reverse search sees
    exactly what it saw before zoom existed.
    """
    img = _open(image_bytes)
    if img is None:
        return []
    crops: list[Crop] = []
    for region in [item for item in ranked if not item.extra][:top_n]:
        data = _crop_box(img, region.box)
        if data is not None:
            crops.append(
                Crop(
                    data=data, subject=region.prompt,
                    confidence=region.confidence,
                )
            )
    return crops


def zoom_crops(
    image_bytes: bytes,
    ranked: list[RankedRegion],
    top_n: int = TOP_ZOOM_CROPS,
) -> list[Crop]:
    """Top padded/upscaled crops from a ranking (zoom-view inputs)."""
    img = _open(image_bytes)
    if img is None:
        return []
    crops: list[Crop] = []
    for region in ranked[:top_n]:
        data = _zoom_box(img, region.box)
        if data is not None:
            crops.append(
                Crop(
                    data=data, subject=region.prompt,
                    confidence=region.confidence,
                )
            )
    return crops


def segment_subjects(
    image_bytes: bytes,
    subjects: list[str],
    extra_prompts: list[str] | tuple[str, ...] = (),
    *,
    mime: str = "image/jpeg",
    client,
    tracker=None,
) -> list[Crop]:
    """Crop the top-2 confident subject regions. Returns [] on any failure."""
    ranked = rank_regions(
        image_bytes, subjects, extra_prompts, mime=mime, client=client,
        tracker=tracker,
    )
    return tight_crops(image_bytes, ranked)
