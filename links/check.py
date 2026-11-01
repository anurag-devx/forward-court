"""Link-safety orchestration: GSB before any contact with the host."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx

from court.models import Verdict
from links import decide as decide_module
from links import scamtext as scamtext_module
from links.extract import expand_shorteners, find_urls
from links.heuristics import age_signals, head_signals, url_signals
from links.safebrowsing import check_urls


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def check_links(
    text: str,
    language: str,
    *,
    client,
    http_client: httpx.Client | None = None,
    tracker=None,
) -> tuple[Verdict | None, dict]:
    """Run link safety. Returns (DANGEROUS verdict|None, info dict).

    Order: extract URLs -> expand shortener hops (HEAD only) -> GSB lookup
    on raw + expanded URLs -> heuristics/HEAD/RDAP for hosts GSB cleared.
    A GSB-hit host is never contacted again.
    """
    info: dict = {"urls": [], "expanded": {}, "flagged_hosts": set(),
                  "safe_urls": [], "signals": {}}
    if http_client is None:
        with httpx.Client() as owned:
            return check_links(
                text, language, client=client, http_client=owned,
                tracker=tracker,
            )
    urls = find_urls(text)
    info["urls"] = urls
    expanded = expand_shorteners(urls, http_client)
    info["expanded"] = expanded
    candidates = list(dict.fromkeys(urls + list(expanded.values())))
    hits = check_urls(
        candidates, os.environ.get("GSB_API_KEY", ""), http_client
    )
    flagged = {_host(url) for url in hits} if hits else set()
    info["flagged_hosts"] = flagged
    if tracker is not None:
        tracker.mark("link_safety")
        if hits:
            tracker.note(f"GSB hits: {len(hits)} URL(s).")
        elif hits is None:
            tracker.note("GSB unavailable (error); heuristics only.")
        elif candidates and os.environ.get("GSB_API_KEY", ""):
            tracker.note("GSB clear.")
        else:
            tracker.note("GSB skipped (no key).")
    signals: dict[str, str] = {}
    safe_urls: list[str] = []
    for url in candidates:
        if _host(url) in flagged:
            signals["gsb_hit"] = ", ".join(sorted(hits.get(url, [])))
            continue
        safe_urls.append(url)
        signals.update(url_signals(url))
        signals.update(head_signals(url, http_client))
        signals.update(age_signals(url, http_client))
    info["safe_urls"] = list(dict.fromkeys(safe_urls))
    info["signals"] = signals
    scam = scamtext_module.read_scam_text(
        text, language, client=client, tracker=tracker
    )
    info["scam_signals"] = scam.signals
    verdict = decide_module.decide(
        signals, scam.signals, scam.scam_type, language
    )
    if verdict is not None and tracker is not None:
        tracker.note(f"DANGEROUS: {verdict.reason}")
    return verdict, info
