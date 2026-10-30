"""Deterministic publication-date extraction from search-result text.

Regex-only on purpose: dates annotate grounded results, and this keeps the
extractor offline and predictable. Returns ISO YYYY-MM-DD or None.
"""
from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

_PATTERNS = [
    # 2026-04-10, 2026/04/10
    re.compile(r"\b((?:19|20)\d{2})[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])\b"),
    # April 10, 2026 / Apr 10 2026
    re.compile(
        rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b",
        re.IGNORECASE,
    ),
    # 10 April 2026 / 10 Apr 2026
    re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE}),?\s+((?:19|20)\d{{2}})\b",
        re.IGNORECASE,
    ),
]


def _valid(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_date(text: str) -> str | None:
    """Return the first valid calendar date in text as ISO, else None."""
    if not text:
        return None
    candidates: list[tuple[int, str | None]] = []
    for match in _PATTERNS[0].finditer(text):
        year, month, day = (int(g) for g in match.groups())
        candidates.append((match.start(), _valid(year, month, day)))
    for match in _PATTERNS[1].finditer(text):
        month_name, day, year = match.groups()
        candidates.append(
            (match.start(), _valid(int(year), _MONTHS[month_name.lower()], int(day)))
        )
    for match in _PATTERNS[2].finditer(text):
        day, month_name, year = match.groups()
        candidates.append(
            (match.start(), _valid(int(year), _MONTHS[month_name.lower()], int(day)))
        )
    for _, iso in sorted(candidates):
        if iso is not None:
            return iso
    return None
