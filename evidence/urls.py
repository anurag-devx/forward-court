"""URL normalization for citation matching and evidence dedup."""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from config import TIER1_OUTLETS, TIER3_UGC

_TRACKING_PREFIXES = ("utm_", "pk_", "piwik_", "matomo_")
_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "gclsrc",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "twclid",
        "srsltid",
        "vero_id",
        "yclid",
        "_openstat",
    }
)


def _is_tracking(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def normalize_url(url: str) -> str:
    """Canonicalize a URL so the same source matches across modules.

    Lowercases scheme/host, drops fragments and tracking parameters, sorts the
    remaining query, and strips trailing slashes. Returns the stripped input
    unchanged when it does not parse as an absolute URL.
    """
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.hostname:
        return url
    scheme = parts.scheme.lower() or "https"
    host = parts.hostname.lower()
    port = parts.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/") or "/"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking(k)
    )
    return urlunsplit((scheme, host, path, urlencode(query), ""))


def split_host_path(url: str) -> tuple[str | None, str]:
    """Return (lowercased host without www, path) or (None, '') if invalid."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None, ""
    if not parts.hostname:
        return None, ""
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    return host, parts.path or "/"


def matches_entries(host: str, path: str, entries) -> bool:
    """True when host (+path) matches hostname or host/path-prefix entries."""
    for entry in entries:
        if "/" in entry:
            entry_host, entry_path = entry.split("/", 1)
            if (host == entry_host or host.endswith("." + entry_host)) and (
                path.startswith("/" + entry_path)
            ):
                return True
        elif host == entry or host.endswith("." + entry):
            return True
    return False


def tier_of(url: str) -> int:
    """Source tier: 1 for top outlets, 3 for UGC, 2 by default."""
    host, path = split_host_path(url)
    if host is None:
        return 2
    if matches_entries(host, path, TIER3_UGC):
        return 3
    if matches_entries(host, path, TIER1_OUTLETS):
        return 1
    return 2
