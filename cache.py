"""Repeat cache: local SQLite for final cards. Never fails a run."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import unicodedata
from pathlib import Path

CACHE_ENV = "CACHE"
CACHE_DIR_ENV = "FC_CACHE_DIR"
CACHE_DIRNAME = ".cache"
DB_NAME = "forwardcourt.sqlite"
OFF_VALUES = frozenset({"0", "false", "no", "off"})

TTL_DEFAULT_S = 24 * 3600
TTL_STICKY_S = 7 * 24 * 3600
TTL_UNVERIFIED_S = 3600


def enabled() -> bool:
    """Cache is default-on; CACHE=0/false/no/off disables it."""
    return os.environ.get(CACHE_ENV, "on").strip().lower() not in OFF_VALUES


def normalize_key_text(text: str) -> str:
    """NFKC, trimmed, whitespace collapsed. Case preserved."""
    cleaned = unicodedata.normalize("NFKC", text or "").strip()
    return " ".join(cleaned.split())


def _digest(*parts: str) -> str:
    blob = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def text_key(text: str, flags: str) -> str:
    """Cache key for a text forward."""
    return "text:" + _digest(normalize_key_text(text), flags)


def media_key_for_file(path: str, caption: str | None, flags: str) -> str | None:
    """Cache key for a media forward; None when the file is unreadable."""
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None
    return "media:" + _digest(digest, normalize_key_text(caption or ""), flags)


def ttl_for_verdict(verdict: dict) -> int:
    """Sticky for DANGEROUS and ClaimReview FALSE, short for Unverified."""
    name = verdict.get("verdict")
    if name == "DANGEROUS":
        return TTL_STICKY_S
    if name == "False" and verdict.get("fact_checked_by"):
        return TTL_STICKY_S
    if name == "Unverified":
        return TTL_UNVERIFIED_S
    return TTL_DEFAULT_S


def _db_path() -> Path:
    root = Path(os.environ.get(CACHE_DIR_ENV, CACHE_DIRNAME))
    return root / DB_NAME


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, card TEXT NOT NULL, "
        "verdict TEXT NOT NULL, created REAL NOT NULL, ttl_s REAL NOT NULL)"
    )
    return conn


def lookup(key: str | None) -> dict | None:
    """Stored {"card", "verdict"} for key, or None on miss/expiry/error."""
    if key is None:
        return None
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT card, verdict, created, ttl_s FROM cache WHERE key=?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None
    if row is None:
        return None
    card, verdict_json, created, ttl_s = row
    if time.time() - created > ttl_s:
        try:
            conn = _connect()
            try:
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                conn.commit()
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            pass
        return None
    try:
        verdict = json.loads(verdict_json)
    except ValueError:
        return None
    return {"card": card, "verdict": verdict}


def maybe_store(key: str | None, card: str, tracker) -> None:
    """Store a completed run. Skips verdict-less runs and all errors.

    Timeouts and plain-text replies record no verdict snapshot, so they
    are never cached; exceptions propagate before this is reached.
    """
    if key is None:
        return
    trace = getattr(tracker, "trace", {}) or {}
    verdict = trace.get("verdict")
    if not isinstance(verdict, dict):
        return
    try:
        conn = _connect()
        try:
            now = time.time()
            conn.execute(
                "DELETE FROM cache WHERE created + ttl_s < ?", (now,)
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(key, card, verdict, created, ttl_s) VALUES (?,?,?,?,?)",
                (
                    key, card, json.dumps(verdict), now,
                    ttl_for_verdict(verdict),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        pass
