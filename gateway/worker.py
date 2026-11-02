"""Webhook job: ack fast, run the pipeline in a subprocess, reply."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from gateway.senders import Sender, SenderError
from links.decide import decide

ROOT = Path(__file__).resolve().parent.parent
RUN_TIMEOUT_S = 240
MAX_MEDIA_BYTES = 16 * 1024 * 1024
MEDIA_TIMEOUT_S = 60.0

ACK = {
    "en": "Checking this forward…",
    "hi": "यह फ़ॉरवर्ड जाँचा जा रहा है…",
}
COULD_NOT_VERIFY = {
    "en": "Couldn't verify this right now. Please try again later.",
    "hi": "अभी इसकी जाँच नहीं हो सकी। कृपया बाद में पुनः प्रयास करें।",
}
RISKY_FILE_TYPE = {
    "en": "Dangerous file attachment",
    "hi": "खतरनाक फाइल अटैचमेंट",
}


def ack_language(text: str) -> str:
    """'hi' when the text contains Devanagari, else 'en'."""
    return "hi" if any("\u0900" <= char <= "\u097f" for char in text) else "en"


def download_media(
    url: str,
    auth: tuple[str, str] | None = None,
    *,
    client: httpx.Client | None = None,
) -> bytes | None:
    """GET provider-hosted media (basic auth when given). None on failure."""
    if client is None:
        with httpx.Client() as owned:
            return _download(owned, url, auth)
    return _download(client, url, auth)


def _download(
    client: httpx.Client, url: str, auth: tuple[str, str]
) -> bytes | None:
    try:
        with client.stream(
            "GET", url, auth=auth, timeout=MEDIA_TIMEOUT_S,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            size = 0
            chunks: list[bytes] = []
            for chunk in response.iter_bytes(65536):
                size += len(chunk)
                if size > MAX_MEDIA_BYTES:
                    return None
                chunks.append(chunk)
    except httpx.HTTPError:
        return None
    return b"".join(chunks)


def _suffix(content_type: str) -> str:
    if content_type.startswith("image/"):
        return ".jpg"
    if content_type.startswith("video/"):
        return ".mp4"
    return ".bin"


def run_pipeline(args: list[str], card_path: Path) -> str | None:
    """Run main.py with --card-out; return the card text or None on failure."""
    try:
        subprocess.run(
            [sys.executable, "main.py", *args, "--card-out", str(card_path)],
            cwd=ROOT,
            timeout=RUN_TIMEOUT_S,
            capture_output=True,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if not card_path.exists():
        return None
    return card_path.read_text()


_URL_RE = re.compile(r"https?://\S+")
_PHONE_RE = re.compile(r"\d{8,}")


def _mask_run(match: re.Match) -> str:
    run = match.group(0)
    return "*" * (len(run) - 4) + run[-4:]


def mask_phones(text: str) -> str:
    """Mask phone-like digit runs outside URLs, keeping the last 4 digits.

    URLs pass through byte-identical: article IDs and tracking numbers in
    links are not phone numbers.
    """
    parts = []
    last = 0
    for found in _URL_RE.finditer(text):
        parts.append(_PHONE_RE.sub(_mask_run, text[last:found.start()]))
        parts.append(found.group(0))
        last = found.end()
    parts.append(_PHONE_RE.sub(_mask_run, text[last:]))
    return "".join(parts)


def _send(sender: Sender, to: str, body: str) -> None:
    try:
        sender.send(to, body)
    except SenderError as exc:
        print(f"warning: {mask_phones(str(exc))}", file=sys.stderr)


def handle_risky_file(
    sender: Sender, *, to: str, filename: str, caption: str = ""
) -> None:
    """Reply DANGEROUS for a risky attachment without downloading it."""
    from card.formatter import format_card

    language = ack_language(caption)
    _send(sender, to, ACK[language])
    verdict = decide(
        {"risky_download": filename}, [], RISKY_FILE_TYPE[language],
        language,
    )
    assert verdict is not None
    _send(sender, to, format_card(verdict))


def handle_message(
    sender: Sender,
    *,
    to: str,
    body: str,
    media_url: str | None,
    media_type: str,
    media_auth: tuple[str, str] | None = None,
) -> None:
    """Ack, verify the forward, and reply with the verdict card."""
    language = ack_language(body)
    _send(sender, to, ACK[language])
    card: str | None = None
    if media_url:
        kind = "image" if media_type.startswith("image/") else (
            "video" if media_type.startswith("video/") else None
        )
        if kind is not None:
            data = download_media(media_url, media_auth)
            if data is not None:
                with tempfile.TemporaryDirectory(
                    prefix="forward-court-"
                ) as tmp:
                    media_path = Path(tmp) / f"media{_suffix(media_type)}"
                    media_path.write_bytes(data)
                    args = [f"--{kind}", str(media_path)]
                    if body.strip():
                        args += ["--caption", body]
                    card = run_pipeline(
                        args, Path(tmp) / "card.txt")
    elif body.strip():
        with tempfile.TemporaryDirectory(prefix="forward-court-") as tmp:
            card = run_pipeline([body], Path(tmp) / "card.txt")
    _send(sender, to, card or COULD_NOT_VERIFY[language])
