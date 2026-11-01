"""Image understanding for image forwards (structured output)."""
from __future__ import annotations

import base64
from collections import Counter
from pathlib import Path

from config import agent_model, reasoning_kwargs

from .models import VisionResult, ZoomReading

MAX_IMAGE_BYTES = 10 * 1024 * 1024
CONSISTENCY_VOTES = 3

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
    "image itself makes about what, where, or when it shows, and tag each in "
    "claim_kinds: 'origin_claim' for where/when/who the photo shows, "
    "'content_claim' for statements, headlines, or text visible in a "
    "screenshot, poster, article, or chat. "
    "Report the ISO 639-1 language of visible text, or 'en' when there is none. "
    "Report visible_domain when a domain name is shown (screenshots, posters, "
    "articles), else null. "
    "Classify image_role: 'viral_claim' for the image as shared in a viral "
    "forward (plain photo, screenshot, or meme making a claim); "
    "'original_source' for an undoctored original photo a claim misuses; "
    "'fact_check_marked' when it carries fact-check markings (FAKE/FALSE/ "
    "MISLEADING stamps or labels, side-by-side comparisons, outlet logos, "
    "watermarks, or tickers); 'other' otherwise. "
    "When claim context is given, judge whether the image shows the claimed "
    "event: 'yes', 'no' (with a short mismatch_reason), or 'unclear'. "
    "'unclear' is only for media that could plausibly show the event but "
    "cannot be confirmed; abstract, decorative, or clearly unrelated "
    "visuals (test patterns, solid colors, wrong place or subject) are 'no'. "
    "With no claim context, shows_claimed_event is 'unclear'. "
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


ZOOM_INSTRUCTIONS = (
    "You get one full view plus zoomed views of its main regions. Read small "
    "text the full view may miss (URL bars, dates, watermarks, captions, "
    "tickers) and transcribe it verbatim. Name the main depicted people, "
    "places, and objects. Report visible_domain when a domain name is shown "
    "in any view, else null; set domain_from_text_crop true only when the "
    "domain comes from a zoomed text, logo, or url-bar view rather than the "
    "full view. Never guess text you cannot read."
)


def zoom_views(
    image_bytes: bytes, crops: list[bytes], *, mime: str = "image/jpeg",
    client, tracker=None,
) -> ZoomReading | None:
    """One vision call over the full view plus zoomed crops.

    Returns None when there are no crops or the call fails: zoom must
    never block the pipeline.
    """
    if not crops:
        return None
    if tracker is not None:
        tracker.mark("zoom")
    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                "Full view first, then zoomed views of its main regions. "
                "Read the small text."
            ),
        },
        {
            "type": "input_image",
            "image_url": (
                f"data:{mime};base64,"
                f"{base64.b64encode(image_bytes).decode()}"
            ),
        },
    ]
    for index, crop in enumerate(crops, start=1):
        content.extend([
            {"type": "input_text", "text": f"Zoomed view {index}:"},
            {
                "type": "input_image",
                "image_url": (
                    f"data:image/jpeg;base64,{base64.b64encode(crop).decode()}"
                ),
            },
        ])
    try:
        response = client.responses.parse(
            model=agent_model(),
            instructions=ZOOM_INSTRUCTIONS,
            input=[{"role": "user", "content": content}],
            text_format=ZoomReading,
            **reasoning_kwargs("vision"),
        )
    except Exception:
        return None
    if tracker is not None:
        tracker.add(response.usage, stage="zoom")
    return response.output_parsed


def merge_zoom_fields(
    base_text: str, base_domain: str | None, base_subjects: list[str],
    zoom: ZoomReading | None,
) -> tuple[str, str | None, list[str], bool]:
    """Merge a zoom reading into pass-1 fields.

    Returns (text, domain, subjects, domain_conflicted). Text and subjects
    are unions; the domain prefers pass 1 unless only zoom saw one, except
    a zoom read from a text crop wins outright. A conflict without a
    text-crop read keeps pass 1 and flags conflict (skip impersonation).
    """
    if zoom is None:
        return base_text, base_domain, base_subjects, False
    lines = [line for line in base_text.splitlines() if line.strip()]
    for line in zoom.visible_text.splitlines():
        if line.strip() and line not in lines:
            lines.append(line)
    subjects = list(base_subjects)
    for subject in zoom.people_places_objects:
        if subject not in subjects:
            subjects.append(subject)
    base = (base_domain or "").strip().lower() or None
    zdomain = (zoom.visible_domain or "").strip().lower() or None
    if base is None:
        return "\n".join(lines), zdomain, subjects, False
    if zdomain is None or zdomain == base:
        return "\n".join(lines), base, subjects, False
    if zoom.domain_from_text_crop:
        return "\n".join(lines), zdomain, subjects, False
    return "\n".join(lines), base, subjects, True


def merge_zoom(
    vision: VisionResult, zoom: ZoomReading | None
) -> tuple[VisionResult, bool]:
    """Merge a zoom reading into a VisionResult. Returns (merged, conflict)."""
    text, domain, subjects, conflicted = merge_zoom_fields(
        vision.visible_text, vision.visible_domain,
        vision.people_places_objects, zoom,
    )
    return (
        vision.model_copy(
            update={
                "visible_text": text,
                "visible_domain": domain,
                "people_places_objects": subjects,
            }
        ),
        conflicted,
    )


def majority_showing(
    votes: list[VisionResult],
) -> tuple[str, int, str]:
    """Majority shows_claimed_event; tie (1-1-1) is unclear.

    Returns (value, agreement, reason from the first agreeing vote).
    """
    counts = Counter(vote.shows_claimed_event for vote in votes)
    top, agreement = counts.most_common(1)[0]
    if agreement == 1:
        return "unclear", 1, ""
    reason = next(
        vote.mismatch_reason
        for vote in votes
        if vote.shows_claimed_event == top
    )
    return top, agreement, reason


def _parse_image(client, tracker, prompt: str, data_url: str) -> VisionResult:
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
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text_format=VisionResult,
        **reasoning_kwargs("vision"),
    )
    if tracker is not None:
        tracker.add(response.usage, stage="vision")
    return response.output_parsed


def describe_image(
    image_bytes: bytes, *, mime: str = "image/jpeg", client, tracker=None,
    context: str | None = None,
) -> VisionResult:
    """Run image understanding. Local bytes are sent as a base64 data URL.

    With claim context, the showing judgment is voted 3x (tie: unclear);
    other fields come from the first vote. Without context, one call.
    """
    if tracker is not None:
        tracker.mark("vision")
    data_url = (
        f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    )
    prompt = "Describe this forwarded image."
    if context and context.strip():
        prompt += f" Claim context: {context.strip()}"
    first = _parse_image(client, tracker, prompt, data_url)
    if not (context and context.strip()):
        return first
    votes = [first] + [
        _parse_image(client, tracker, prompt, data_url)
        for _ in range(CONSISTENCY_VOTES - 1)
    ]
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
