"""URL normalization for citation matching and evidence dedup."""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from config import (
    DENYLISTED_HOSTS,
    SPOOFED_OUTLETS,
    TIER1_OUTLETS,
    TIER3_UGC,
    TRUSTED_FACTCHECK_DOMAINS,
)

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


def youtube_video_id(url: str) -> str | None:
    """YouTube video ID from watch/embed/shorts/live/youtu.be URLs."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host.endswith("youtu.be"):
        ident = parts.path.strip("/").split("/")[0]
        return ident or None
    if not host.endswith("youtube.com"):
        return None
    path = parts.path
    if path == "/watch":
        params = dict(parse_qsl(parts.query, keep_blank_values=True))
        return params.get("v") or None
    for prefix in ("/embed/", "/shorts/", "/live/", "/v/"):
        if path.startswith(prefix):
            ident = path[len(prefix):].split("/")[0]
            return ident or None
    return None


def same_source(first: str, second: str) -> bool:
    """True when two URLs identify the same source page or video.

    Normalized-URL equality, or the same YouTube video ID across
    watch/embed/shorts/youtu.be forms.
    """
    if normalize_url(first) == normalize_url(second):
        return True
    first_id = youtube_video_id(first)
    return first_id is not None and first_id == youtube_video_id(second)


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


def _outlet_hosts() -> set[str]:
    hosts = set()
    for entry in SPOOFED_OUTLETS + TIER1_OUTLETS:
        host = entry.split("/")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        hosts.add(host)
    return hosts


def impersonates_outlet(domain: str | None) -> bool:
    """True when a visible domain spoofs a known outlet's name.

    Genuine outlet hosts (exact or subdomain) return False; a domain that
    merely carries an outlet name as a label (abcnews.com.co) returns True.
    """
    if not domain:
        return False
    text = domain.strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    host = text.split("/")[0].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if not host or "." not in host:
        return False
    known = _outlet_hosts()
    if host in known or any(host.endswith("." + outlet) for outlet in known):
        return False
    names = set()
    for outlet in known:
        labels = outlet.split(".")
        name = labels[0] if labels[0] not in ("english", "www") else labels[1]
        names.add(name)
    return any(label in names for label in host.split("."))


def is_factcheck_url(url: str) -> bool:
    """True when the URL belongs to a trusted fact-check outlet."""
    host, path = split_host_path(url)
    if host is None:
        return False
    return matches_entries(host, path, TRUSTED_FACTCHECK_DOMAINS)


def _matches_denylist(host: str) -> bool:
    """True when the host is a listed mirror/scraper.

    Dotted entries match the host and its subdomains; bare names match a
    full DNS label on any TLD.
    """
    labels = host.split(".")
    for entry in DENYLISTED_HOSTS:
        if "." in entry:
            if host == entry or host.endswith("." + entry):
                return True
        elif entry in labels:
            return True
    return False


def tier_of(url: str) -> int:
    """Source tier: 1 for top outlets, 3 for UGC, 2 by default."""
    host, path = split_host_path(url)
    if host is None:
        return 2
    if matches_entries(host, path, TIER3_UGC) or _matches_denylist(host):
        return 3
    if matches_entries(host, path, TIER1_OUTLETS):
        return 1
    return 2
