"""Deterministic recency detection for image forwards.

A forward is presented-as-recent when its visible text or caption carries
urgency/recency cues (English + Hindi) or an explicit date within
RECENT_DAYS of today. Pure function of its inputs: offline and testable.
"""
from __future__ import annotations

import re
from datetime import date

from dates import extract_all_dates

RECENT_DAYS = 7

CUES = (
    # English urgency / recency markers.
    "breaking",
    "just in",
    "just now",
    "today",
    "tonight",
    "this morning",
    "this evening",
    "yesterday",
    "alert",
    "urgent",
    "viral",
    "must share",
    # Hindi equivalents.
    "ब्रेकिंग",
    "ताजा",
    "ताज़ा",
    "आज",
    "अभी",
    "कल",
    "सावधान",
    "वायरल",
    "तुरंत",
)

_CUE_RES = [re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE) for cue in CUES]


def is_presented_as_recent(
    texts: list[str], today: date | None = None
) -> bool:
    """True when any text presents events as breaking or current."""
    combined = "\n".join(text for text in texts if text)
    if not combined.strip():
        return False
    if any(pattern.search(combined) for pattern in _CUE_RES):
        return True
    today = today or date.today()
    for iso in extract_all_dates(combined):
        delta = (today - date.fromisoformat(iso)).days
        if 0 <= delta <= RECENT_DAYS:
            return True
    return False
