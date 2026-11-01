"""Generate the demo screenshot: fictional masthead, real false claim.

The outlet is invented ("The Daily Clarion"); the headline is a known,
already fact-checked false claim so the run reaches FALSE via lookup.
No real outlet branding, no copyrighted images.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

MASTHEAD = "The Daily Clarion"
HEADLINE = "Hot lemon water cures cancer, doctors confirm"
STANDFIRST = (
    "Health experts say a glass of hot lemon water every morning "
    "destroys cancer cells, reports our wellness desk."
)
FOOTER = "Forwarded many times  -  Clarion Wellness Desk"

SAMPLES = ("text", "screenshot", "link", "scam")

WIDTH, HEIGHT = 900, 620


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow: fixed bitmap font
        return ImageFont.load_default()


def _wrapped(draw: ImageDraw.ImageDraw, text: str, font, max_width: int):
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_screenshot(path: str) -> str:
    """Render the fictional news screenshot to path. Returns path."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    margin = 48
    draw.rectangle([0, 0, WIDTH, 110], fill=(20, 40, 90))
    draw.text(
        (margin, 28), MASTHEAD, font=_font(44), fill="white"
    )
    draw.text(
        (margin, 132), "WELLNESS  -  MORNING EDITION",
        font=_font(20), fill=(120, 120, 120),
    )
    y = 175
    for line in _wrapped(draw, HEADLINE, _font(40), WIDTH - 2 * margin):
        draw.text((margin, y), line, font=_font(40), fill="black")
        y += 52
    y += 10
    for line in _wrapped(draw, STANDFIRST, _font(24), WIDTH - 2 * margin):
        draw.text((margin, y), line, font=_font(24), fill=(60, 60, 60))
        y += 34
    draw.rectangle([margin, HEIGHT - 90, WIDTH - margin, HEIGHT - 88],
                   fill=(200, 200, 200))
    draw.text(
        (margin, HEIGHT - 72), FOOTER, font=_font(20), fill=(120, 120, 120)
    )
    image.save(path, "PNG")
    return path
