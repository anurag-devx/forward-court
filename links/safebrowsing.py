"""Google Safe Browsing v4 Lookup API reputation check."""
from __future__ import annotations

import httpx

ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
CLIENT_ID = "forwardcourt"
CLIENT_VERSION = "1.0"
THREAT_TYPES = [
    "MALWARE",
    "SOCIAL_ENGINEERING",
    "UNWANTED_SOFTWARE",
    "POTENTIALLY_HARMFUL_APPLICATION",
]
TIMEOUT_S = 10.0


def check_urls(
    urls: list[str], api_key: str, client: httpx.Client
) -> dict[str, list[str]] | None:
    """Map flagged URLs to matched threat types.

    {} when clean; None when the API call itself failed (unknown).
    """
    if not urls or not api_key:
        return {}
    try:
        response = client.post(
            ENDPOINT,
            params={"key": api_key},
            json={
                "client": {
                    "clientId": CLIENT_ID,
                    "clientVersion": CLIENT_VERSION,
                },
                "threatInfo": {
                    "threatTypes": THREAT_TYPES,
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{"url": url} for url in urls],
                },
            },
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    hits: dict[str, list[str]] = {}
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    for match in matches:
        threat = match.get("threat", {}) if isinstance(match, dict) else {}
        url = threat.get("url")
        kind = match.get("threatType", "UNKNOWN") if isinstance(match, dict) else "UNKNOWN"
        if url:
            hits.setdefault(url, []).append(kind)
    return hits
