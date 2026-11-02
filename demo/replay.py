"""Offline replay of recorded demo traces: no key, no network, no models."""
from __future__ import annotations

import json
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
SAMPLES_PATH = DEMO_DIR / "samples.json"
RECORDINGS_DIR = DEMO_DIR / "recordings"

PAIRS = (
    ("text", "text_real", "text_fake"),
    ("screenshot", "shot_real", "shot_fake"),
    ("link", "link_real", "link_fake"),
    ("malicious", None, "malicious"),
)


def load_samples() -> list[dict]:
    return json.loads(SAMPLES_PATH.read_text())


def load_recordings() -> dict[str, dict]:
    """Recordings by sample id. Missing ids are absent, never an error."""
    found = {}
    for path in sorted(RECORDINGS_DIR.glob("*.json")):
        recording = json.loads(path.read_text())
        found[recording["id"]] = recording
    return found


def _claim_summary(recording: dict) -> str:
    claims = recording.get("claims", [])
    if not claims:
        return "no claims"
    kinds = sorted({claim.get("kind", "?") for claim in claims})
    return f"{len(claims)} claim(s) ({', '.join(kinds)})"


def _key_evidence(recording: dict) -> str:
    verdict = recording.get("verdict") or {}
    key = verdict.get("key_source") or ""
    if not key:
        scam = verdict.get("scam_type")
        return f"scam rule ({scam})" if scam else "no key source"
    tier = next(
        (
            item.get("tier")
            for item in recording.get("evidence", [])
            if item.get("source_url") == key
        ),
        None,
    )
    host = key.split("/")[2] if "/" in key else key
    return f"{host} (tier {tier})" if tier else host


def _path_summary(recording: dict) -> str:
    if recording.get("votes"):
        votes = ", ".join(vote["verdict"] for vote in recording["votes"])
        return f"votes {votes}"
    hits = recording.get("claimreview_hits", [])
    if hits:
        outlets = sorted({hit["source_url"].split("/")[2] for hit in hits})
        return f"ClaimReview {', '.join(outlets)}"
    link = recording.get("link_safety") or {}
    signals = sorted(link.get("signals", {})) + sorted(
        link.get("scam_signals", [])
    )
    if signals:
        return f"link rules ({', '.join(signals[:3])})"
    return "short-circuit"


def provenance_line(recording: dict) -> str:
    """Short claims → evidence → path → verdict line for one recording."""
    verdict = (recording.get("verdict") or {}).get("verdict", "?")
    return (
        f"{_claim_summary(recording)} → {_key_evidence(recording)} → "
        f"{_path_summary(recording)} → {verdict}"
    )


def render_comparison(recordings: dict[str, dict]) -> str:
    """Real-vs-fake verdict table with the fake side's key source.

    Missing recordings show as 'not recorded', never crash.
    """
    rows = [("| type | real | fake | key source |"),
            ("|---|---|---|---|")]
    for label, real_id, fake_id in PAIRS:
        real = recordings.get(real_id) if real_id else None
        fake = recordings.get(fake_id)
        if real_id is None:
            real_verdict = "—"
        elif real is None:
            real_verdict = "not recorded"
        else:
            real_verdict = (real.get("verdict") or {}).get("verdict", "?")
        if fake is None:
            fake_verdict, key = "not recorded", "—"
        else:
            fake_verdict = (fake.get("verdict") or {}).get("verdict", "?")
            key = _key_evidence(fake)
        rows.append(f"| {label} | {real_verdict} | {fake_verdict} | {key} |")
    return "\n".join(rows)


def run_replay() -> int:
    """Print banner + table + cards from recordings. No models called."""
    recordings = load_recordings()
    samples = load_samples()
    if recordings:
        first = next(iter(recordings.values()))
        recorded_on = str(first.get("recorded_at", "?"))[:10]
    else:
        recorded_on = "?"
    print(f"REPLAY — recorded live on {recorded_on}, no models called.\n")
    print(render_comparison(recordings))
    for sample in samples:
        recording = recordings.get(sample["id"])
        print(f"\n=== {sample['id']} ({sample['label']}) ===")
        print(f"expected: {sample['expected']}")
        if recording is None:
            print("not recorded yet. Run: python main.py --demo --record")
            continue
        print(recording.get("card", "(no card)"))
        print(f"provenance: {provenance_line(recording)}")
    return 0
