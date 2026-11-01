"""Deterministic link-risk heuristics; each hit is a named signal."""
from __future__ import annotations

import ipaddress
from datetime import date
from urllib.parse import urlsplit

import httpx

from config import BRAND_TARGETS
from evidence.urls import impersonates_outlet

RISKY_EXTENSIONS = (".apk", ".exe", ".msi", ".scr", ".bat", ".js", ".zip", ".rar")
RISKY_CONTENT_TYPES = (
    "application/vnd.android.package-archive",
    "application/x-msdownload",
    "application/x-msdos-program",
    "application/x-sh",
    "application/zip",
    "application/x-rar",
    "application/x-javascript",
    "application/octet-stream",
)
MAX_SUBDOMAINS = 3
NEW_DOMAIN_DAYS = 30
RDAP_BOOTSTRAP = "https://data.iana.org/rdap/dns.json"
HEAD_TIMEOUT_S = 8.0
RDAP_TIMEOUT_S = 10.0


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _brand_spoofed(host: str) -> str | None:
    """Brand name carried as a label on a non-brand host, if any."""
    bare = host[4:] if host.startswith("www.") else host
    for brand in BRAND_TARGETS:
        if brand in bare.split(".") and not bare.startswith(brand + "."):
            return brand
    return None


def url_signals(url: str) -> dict[str, str]:
    """Pure (no-contact) signals for one URL: name -> detail."""
    signals: dict[str, str] = {}
    try:
        parts = urlsplit(url)
    except ValueError:
        return signals
    host = (parts.hostname or "").lower()
    if not host:
        return signals
    if host.startswith("xn--") or ".xn--" in host:
        signals["punycode_host"] = host
    try:
        ipaddress.ip_address(host)
        signals["raw_ip_host"] = host
    except ValueError:
        pass
    bare = host[4:] if host.startswith("www.") else host
    if len(bare.split(".")) - 1 > MAX_SUBDOMAINS:
        signals["many_subdomains"] = host
    if "@" in parts.netloc:
        signals["at_in_url"] = url
    if parts.scheme == "http":
        signals["plain_http"] = url
    if impersonates_outlet(host):
        signals["lookalike_domain"] = f"outlet spoof: {host}"
    else:
        brand = _brand_spoofed(host)
        if brand:
            signals["lookalike_domain"] = f"brand spoof ({brand}): {host}"
    if parts.path.lower().endswith(RISKY_EXTENSIONS):
        signals["risky_download"] = f"risky extension: {parts.path}"
    return signals


def head_signals(url: str, client: httpx.Client) -> dict[str, str]:
    """HEAD-based signals (final host; call only after a clean GSB check)."""
    try:
        response = client.head(
            url, follow_redirects=True, timeout=HEAD_TIMEOUT_S
        )
    except httpx.HTTPError:
        return {}
    content_type = response.headers.get("content-type", "").split(";")[0]
    content_type = content_type.strip().lower()
    if content_type in RISKY_CONTENT_TYPES:
        return {"risky_download": f"risky content-type: {content_type}"}
    return {}


def _rdap_server(tld: str, client: httpx.Client) -> str | None:
    try:
        response = client.get(RDAP_BOOTSTRAP, timeout=RDAP_TIMEOUT_S)
        response.raise_for_status()
        services = response.json().get("services", [])
    except (httpx.HTTPError, ValueError):
        return None
    for entry in services:
        if (
            isinstance(entry, list)
            and len(entry) == 2
            and tld in entry[0]
            and entry[1]
        ):
            return entry[1][0].rstrip("/")
    return None


def domain_age_days(host: str, client: httpx.Client) -> int | None:
    """Days since RDAP registration; None when unknown/unreachable."""
    bare = host[4:] if host.startswith("www.") else host
    if "." not in bare:
        return None
    tld = bare.rsplit(".", 1)[1]
    server = _rdap_server(tld, client)
    if server is None:
        return None
    try:
        response = client.get(
            f"{server}/domain/{bare}", timeout=RDAP_TIMEOUT_S,
            headers={"Accept": "application/rdap+json"},
        )
        response.raise_for_status()
        events = response.json().get("events", [])
    except (httpx.HTTPError, ValueError):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("eventAction") == "registration" and event.get(
            "eventDate"
        ):
            try:
                registered = date.fromisoformat(
                    event["eventDate"][:10]
                )
            except ValueError:
                return None
            return (date.today() - registered).days
    return None


def age_signals(url: str, client: httpx.Client) -> dict[str, str]:
    """New-domain signal via RDAP; {} when age is unknown or old."""
    host = _host(url)
    if not host:
        return {}
    age = domain_age_days(host, client)
    if age is not None and age < NEW_DOMAIN_DAYS:
        return {"new_domain": f"registered {age}d ago: {host}"}
    return {}
