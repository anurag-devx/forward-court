"""Full-trace recorder for --record: build, scrub, save, validate."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import (
    SEGMENTATION_MODEL,
    TRANSCRIBE_MODEL,
    agent_model,
)
from gateway.worker import mask_phones

RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

REQUIRED_KEYS = (
    "id",
    "kind",
    "label",
    "expected",
    "input",
    "card",
    "claims",
    "evidence",
    "verdict",
    "models",
    "tokens",
    "times",
    "recorded_at",
    "git_commit",
)

VERDICTS = (
    "True", "False", "Misleading", "Unverified", "DANGEROUS", "SUSPICIOUS"
)

SENSITIVE_ENV_KEYS = (
    "MODEL_API_KEY",
    "SERPAPI_KEY",
    "GSB_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "PUBLIC_BASE_URL",
    "VONAGE_API_KEY",
    "VONAGE_API_SECRET",
    "VONAGE_WHATSAPP_NUMBER",
    "VONAGE_SIGNATURE_SECRET",
    "VONAGE_WEBHOOK_TOKEN",
)


def git_commit() -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent),
             "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if commit else "unknown"


def build_recording(
    *, sample_id: str, kind: str, label: str, expected: str,
    input_desc: dict, card: str, tracker, elapsed_s: float,
) -> dict:
    """Assemble a full trace from a finished run. Not yet scrubbed."""
    trace = tracker.trace
    return {
        "id": sample_id,
        "kind": kind,
        "label": label,
        "expected": expected,
        "input": input_desc,
        "card": card,
        "claims": trace.get("claims", []),
        "link_safety": trace.get("link_safety"),
        "article": trace.get("article"),
        "evidence": trace.get("evidence", []),
        "relevance_drops": trace.get("relevance_drops", []),
        "claimreview_hits": trace.get("claimreview_hits", []),
        "debate": trace.get("debate"),
        "votes": trace.get("votes", []),
        "verdict": trace.get("verdict"),
        "models": {
            "agent": agent_model(),
            "segmentation": SEGMENTATION_MODEL,
            "transcribe": TRANSCRIBE_MODEL,
        },
        "tokens": {
            "total": tracker.total_tokens,
            "stages": {
                stage: {"in": pair[0], "out": pair[1]}
                for stage, pair in tracker.stages.items()
            },
        },
        "times": {
            stage: round(seconds, 1)
            for stage, seconds in tracker.times.items()
        },
        "notes": list(tracker.notes),
        "elapsed_s": round(elapsed_s, 1),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }


def _scrub_string(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "***")
    return mask_phones(text)


def scrub(recording: dict) -> dict:
    """Strip keys, tunnel URLs, and phone numbers from a recording."""
    secrets = [
        value
        for key in SENSITIVE_ENV_KEYS
        if (value := os.environ.get(key, ""))
    ]

    def walk(node):
        if isinstance(node, str):
            return _scrub_string(node, secrets)
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        return node

    return walk(recording)


def save_recording(recording: dict, path: str | Path | None = None) -> Path:
    """Scrub and write a recording. Returns the path written."""
    target = Path(path) if path else RECORDINGS_DIR / f"{recording['id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(scrub(recording), ensure_ascii=False, indent=2) + "\n"
    )
    return target


def validate_recording(recording: dict) -> list[str]:
    """Schema errors; [] when the recording is valid."""
    errors = []
    for key in REQUIRED_KEYS:
        if key not in recording:
            errors.append(f"missing key: {key}")
    verdict = recording.get("verdict") or {}
    if verdict.get("verdict") not in VERDICTS:
        errors.append(f"bad verdict: {verdict.get('verdict')!r}")
    for vote in recording.get("votes", []):
        if vote.get("verdict") not in VERDICTS:
            errors.append(f"bad vote: {vote.get('verdict')!r}")
    for item in recording.get("evidence", []):
        for key in ("source_url", "outlet", "tier", "date"):
            if key not in item:
                errors.append(f"evidence item missing: {key}")
                break
    return errors
