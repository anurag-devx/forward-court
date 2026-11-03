"""Deterministic phishing scoring over message text + URLs. No model calls."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from config import TIER1_OUTLETS
from evidence.urls import matches_entries, split_host_path
from links.extract import find_urls

LEXICON_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "phishing_lexicon.yaml"
)

SUSPICIOUS_MIN = 3
DANGEROUS_MIN = 6
TOP_REASONS = 3
PHRASE_GAP = 2
NEGATION_BEFORE = 3
NEGATION_AFTER = 2

# A credentials match near one of these is legit advice, not an ask.
NEGATIONS = frozenset({
    "never", "dont", "do", "not", "nahi", "nahin", "mat",
    "मत", "नहीं", "नही",
})


@dataclass(frozen=True)
class Brand:
    aliases: tuple[str, ...]
    official: tuple[str, ...]
    display: str


@dataclass
class BrandHit:
    display: str
    official: bool
    digit_swap: bool


@dataclass
class UrlFinding:
    url: str
    brand: BrandHit | None
    punycode: bool
    risky_tld: str | None
    free_host: str | None
    shortener: bool
    actions: list[str]
    download_ext: str | None
    points: int


@dataclass
class LexiconResult:
    score: int
    level: str  # "ok" | "suspicious" | "dangerous"
    reasons: list[str]
    download: bool
    matched: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Lowercase, NFKC, strip punctuation (spaces kept)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(
        char for char in text
        if not unicodedata.category(char).startswith("P")
    )


def _strip_all(text: str) -> str:
    """Lowercase alphanumeric soup for URL substring matching."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(char for char in text if char.isalnum())


def _display(alias: str) -> str:
    return alias.upper() if len(alias) <= 4 else alias.capitalize()


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    with open(LEXICON_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def load_brands() -> tuple[Brand, ...]:
    brands = []
    for key, officials in load_lexicon()["brands"].items():
        aliases = tuple(
            part.strip().lower() for part in str(key).split("/")
        )
        brands.append(Brand(
            aliases=aliases,
            official=tuple(official.strip().lower() for official in officials),
            display=_display(aliases[0]),
        ))
    return tuple(brands)


def _deleeted(label: str) -> set[str]:
    """Label plus digit-swap/typo variants (1→l and 1→i both tried)."""
    forms = {label}
    base = label.replace("rn", "m")
    variants = {base}
    if "1" in base:
        variants = {form.replace("1", "l") for form in variants} | {
            form.replace("1", "i") for form in variants
        }
    for form in variants:
        forms.add(
            form.replace("0", "o").replace("3", "e").replace("5", "s")
        )
    return forms


def _alias_matches(alias: str, labels: list[str], joined: set[str]) -> bool:
    if len(alias) <= 2:
        return alias in labels
    if len(alias) == 3:
        return any(
            label == alias or label.startswith(alias)
            or label.endswith(alias)
            for label in labels
        )
    return any(alias in text for text in joined)


def _is_official(host: str, brand: Brand) -> bool:
    for official in brand.official:
        if host == official or host.endswith("." + official):
            return True
    if any(official.endswith(".gov.in") for official in brand.official):
        if host in ("gov.in", "nic.in") or host.endswith(".gov.in") \
                or host.endswith(".nic.in"):
            return True
    return False


def _find_brand(host: str, brands: tuple[Brand, ...]) -> BrandHit | None:
    labels = [part for part in host.split(".") if part]
    joined = "".join(labels)
    de_labels: list[str] = []
    for label in labels:
        de_labels.extend(sorted(_deleeted(label)))
    de_joined = _deleeted(joined)
    for brand in brands:
        for alias in brand.aliases:
            if _alias_matches(alias, labels, {joined}):
                return BrandHit(brand.display, _is_official(host, brand), False)
            if _alias_matches(alias, de_labels, de_joined):
                return BrandHit(brand.display, _is_official(host, brand), True)
    return None


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _url_actions_exempt(url: str, brands: tuple[Brand, ...]) -> bool:
    """True only for explicitly listed tier-1 outlets or brand officials.

    Unknown hosts (tier-2 by default) are NOT exempt: they score normally.
    """
    host = _host_of(url)
    if not host:
        return False
    split_host, path = split_host_path(url)
    if split_host is not None and matches_entries(
        split_host, path, TIER1_OUTLETS
    ):
        return True
    return any(_is_official(host, brand) for brand in brands)


def _score_url(url: str, lex: dict, brands: tuple[Brand, ...]) -> UrlFinding:
    host = _host_of(url)
    stripped = _strip_all(url)
    hit = _find_brand(host, brands) if host else None
    punycode = any(label.startswith("xn--") for label in host.split("."))
    digit_swap = hit is not None and hit.digit_swap and not hit.official
    lookalike = digit_swap or punycode
    tld = host.rsplit(".", 1)[-1] if host else ""
    risky_tld = tld if tld in lex["risky_tlds"] else None
    free_host = next(
        (entry for entry in lex["free_hosts"]
         if host == entry or host.endswith("." + entry)),
        None,
    )
    shortener = any(
        host == entry or host.endswith("." + entry)
        for entry in lex["shorteners"]
    )
    matched = [
        word for word in lex["url_action_words"]
        if _strip_all(word) in stripped
    ]
    actions: list[str] = []
    if not _url_actions_exempt(url, brands):
        generic = set(lex["generic_words"])
        specific = [word for word in matched if word not in generic]
        extra = [word for word in matched if word in generic]
        base = len(specific)
        if hit is not None and not hit.official:
            base += 3
        if lookalike:
            base += 3
        if risky_tld is not None or free_host is not None:
            base += 2
        pool = specific + (extra if base > 0 else [])
        actions = pool[:2]
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        path = ""
    filename = path.rsplit("/", 1)[-1]
    download_ext = next(
        (ext for ext in lex["download_exts"]
         if filename.endswith("." + ext)),
        None,
    )
    points = len(actions)
    if hit is not None and not hit.official:
        points += 3
    if lookalike:
        points += 3
    if risky_tld is not None or free_host is not None:
        points += 2
    return UrlFinding(
        url=url, brand=hit, punycode=punycode, risky_tld=risky_tld,
        free_host=free_host, shortener=shortener, actions=actions,
        download_ext=download_ext, points=points,
    )


def _word_eq(pattern_word: str, token: str) -> bool:
    """Token matches a pattern word, tolerating inflections on long words."""
    if token == pattern_word:
        return True
    return len(pattern_word) >= 4 and token.startswith(pattern_word)


def _phrase_spans(words: list[str], tokens: list[str]) -> list[tuple[int, int]]:
    """Token spans where words match in order within a small window."""
    if len(words) == 1:
        return [
            (i, i) for i, token in enumerate(tokens)
            if _word_eq(words[0], token)
        ]
    spans = []
    for i, token in enumerate(tokens):
        if not _word_eq(words[0], token):
            continue
        pos = i
        for word in words[1:]:
            found = None
            stop = i + len(words) + PHRASE_GAP
            for k in range(pos + 1, min(len(tokens), stop)):
                if _word_eq(word, tokens[k]):
                    found = k
                    break
            if found is None:
                break
            pos = found
        else:
            spans.append((i, pos))
    return spans


def _negated(span: tuple[int, int], tokens: list[str]) -> bool:
    start, end = span
    before = tokens[max(0, start - NEGATION_BEFORE):start]
    after = tokens[end + 1:end + 1 + NEGATION_AFTER]
    return any(token in NEGATIONS for token in before + after)


def _pattern_hits(patterns: list[str], tokens: list[str]) -> dict[str, list]:
    """Pattern text -> spans for every matching pattern."""
    hits = {}
    for pattern in patterns:
        words = normalize_text(str(pattern)).split()
        if not words:
            continue
        spans = _phrase_spans(words, tokens)
        if spans:
            hits[str(pattern)] = spans
    return hits


def _has_pay_word(tokens: list[str], text: str, lex: dict) -> bool:
    for word in lex["pay_words"]:
        cleaned = normalize_text(str(word))
        if not cleaned:
            continue
        if " " in cleaned.strip():
            if _phrase_spans(cleaned.split(), tokens):
                return True
        elif cleaned.strip() == "₹":
            if "₹" in text:
                return True
        elif any(_word_eq(cleaned, token) for token in tokens):
            return True
    return False


def _has_claim_link(urls: list[str], lex: dict) -> bool:
    words = [_strip_all(str(word)) for word in lex["claim_link_words"]]
    for url in urls:
        stripped = _strip_all(url)
        if any(word and word in stripped for word in words):
            return True
    return False


def _score_text(tokens: list[str], urls: list[str], lex: dict):
    """Category -> (points, matched detail). Gated categories last."""
    patterns = lex["text_patterns"]
    creds = {
        pattern: spans
        for pattern, spans in _pattern_hits(patterns["credentials"], tokens).items()
        if [span for span in spans if not _negated(span, tokens)]
    }
    threats = _pattern_hits(patterns["threats"], tokens)
    money = _pattern_hits(patterns["money_bait"], tokens)
    lottery_spans = money.pop("lottery", [])
    if lottery_spans and (creds or _has_claim_link(urls, lex)):
        money["lottery"] = lottery_spans
    job = _pattern_hits(patterns["job_bait"], tokens)
    task_spans = job.pop("task", [])
    if task_spans and _has_pay_word(tokens, " ".join(tokens), lex):
        job["task"] = task_spans
    remote = _pattern_hits(patterns["remote_access"], tokens)
    urgency_hits = _pattern_hits(patterns["urgency"], tokens)
    phrase_hit = any(
        len(normalize_text(pattern).split()) > 1 for pattern in urgency_hits
    )
    urgency = urgency_hits if (
        phrase_hit or (urgency_hits and (creds or threats or money))
    ) else {}
    return {
        "credentials": (3 if creds else 0, sorted(creds)),
        "threats": (2 if threats else 0, sorted(threats)),
        "money_bait": (2 if money else 0, sorted(money)),
        "job_bait": (2 if job else 0, sorted(job)),
        "remote_access": (3 if remote else 0, sorted(remote)),
        "urgency": (2 if urgency else 0, sorted(urgency)),
    }


def _credential_reason(matched: list[str], language: str) -> str:
    text = " ".join(matched).lower()
    if language == "hi":
        if "cvv" in text or "card" in text:
            return "कार्ड की जानकारी मांगता है"
        if "password" in text:
            return "पासवर्ड मांगता है"
        if "pin" in text:
            return "PIN मांगता है"
        return "OTP मांगता है"
    if "cvv" in text or "card" in text:
        return "Asks for card details"
    if "password" in text:
        return "Asks for your password"
    if "pin" in text:
        return "Asks for your PIN"
    return "Asks for your OTP"


_THREAT_REASONS = (
    (
        ("bijli", "electricity", "बिजली", "कनेक्शन"),
        "Threatens to cut your electricity",
        "बिजली काटने की धमकी",
    ),
    (
        ("challan", "चालान"),
        "Claims a challan is pending",
        "चालान बकाया होने का दावा",
    ),
    (("legal",), "Threatens legal action", "कानूनी कार्रवाई की धमकी"),
    (("pan",), "Claims your PAN is not linked", "PAN लिंक न होने का दावा"),
    (("kyc",), "Claims your KYC expired", "KYC समाप्त होने का दावा"),
    (
        ("froze", "frozen"),
        "Claims your account is frozen",
        "खाता फ्रीज होने का दावा",
    ),
)


def _threat_reason(matched: list[str], language: str) -> str:
    text = " ".join(matched).lower()
    for keywords, english, hindi in _THREAT_REASONS:
        if any(keyword in text for keyword in keywords):
            return hindi if language == "hi" else english
    if language == "hi":
        return "खाता बंद करने की धमकी"
    return "Threatens to block your account"


def _reasons(
    text_hits: dict, best: UrlFinding | None, language: str,
) -> list[str]:
    hi = language == "hi"
    out: list[str] = []

    def add(reason: str | None):
        if reason and reason not in out:
            out.append(reason)

    if best is not None and best.download_ext:
        add(
            f"फ़ाइल डाउनलोड की ओर ले जाता है (.{best.download_ext})"
            if hi else
            f"Leads to a file download (.{best.download_ext})"
        )
    if best is not None and best.brand is not None \
            and not best.brand.official:
        display = best.brand.display
        add(
            f"{display} होने का ढोंग, लेकिन साइट {display} की नहीं है"
            if hi else
            f"Pretends to be {display} but the site isn't {display}'s"
        )
    if best is not None and (
        best.punycode or (
            best.brand is not None and best.brand.digit_swap
            and not best.brand.official
        )
    ):
        digit_swap = best.brand is not None and best.brand.digit_swap \
            and not best.brand.official
        if digit_swap:
            add(
                "अक्षर बदले हुए मिलते-जुलते पते का उपयोग"
                if hi else
                "Uses a lookalike address with swapped letters"
            )
        else:
            add(
                "भ्रामक विशेष-अक्षर पते का उपयोग"
                if hi else
                "Uses a deceptive special-character address"
            )
    if text_hits["credentials"][0]:
        add(_credential_reason(text_hits["credentials"][1], language))
    if text_hits["remote_access"][0]:
        add(
            "स्क्रीन-शेयरिंग ऐप इंस्टॉल करने को कहता है"
            if hi else
            "Asks you to install a screen-sharing app"
        )
    if best is not None and best.risky_tld is not None:
        add(
            f"संदिग्ध लिंक अंत (.{best.risky_tld})"
            if hi else
            f"Suspicious link ending (.{best.risky_tld})"
        )
    elif best is not None and best.free_host is not None:
        add(
            f"मुफ्त साइट पर होस्टेड ({best.free_host})"
            if hi else
            f"Hosted on a free site ({best.free_host})"
        )
    if text_hits["threats"][0]:
        add(_threat_reason(text_hits["threats"][1], language))
    elif text_hits["urgency"][0]:
        add(
            "तुरंत कार्रवाई का दबाव बनाता है"
            if hi else
            "Pressures you to act right now"
        )
    if text_hits["money_bait"][0]:
        add("लॉटरी या कैशबैक का लालच" if hi else "Promises prize money or cashback")
    if text_hits["job_bait"][0]:
        add("घर बैठे नौकरी का फर्जी लालच" if hi else "Fake work-from-home job bait")
    if best is not None and best.actions:
        add(
            f"लिंक में लालच: {', '.join(best.actions)}"
            if hi else
            f"Link baits with: {', '.join(best.actions)}"
        )
    if best is not None and best.shortener:
        add(
            "छोटे लिंक के पीछे असली पता छिपा है"
            if hi else
            "Hides the real address behind a short link"
        )
    return out[:TOP_REASONS]


def score_message(
    text: str, urls: list[str] | None = None, language: str = "en",
) -> LexiconResult:
    """Score text + URLs. Returns score, level, and top plain-word reasons."""
    lex = load_lexicon()
    brands = load_brands()
    links = list(dict.fromkeys(urls or []))
    scan = text or ""
    for found in dict.fromkeys(find_urls(scan) + links):
        scan = scan.replace(found, " ")
    tokens = normalize_text(scan).split()
    text_hits = _score_text(tokens, links, lex)
    findings = [_score_url(url, lex, brands) for url in links]
    best = max(findings, key=lambda item: item.points) if findings else None
    download = any(item.download_ext is not None for item in findings)
    score = sum(points for points, _ in text_hits.values())
    score += best.points if best is not None else 0
    if download or score >= DANGEROUS_MIN:
        level = "dangerous"
    elif score >= SUSPICIOUS_MIN:
        level = "suspicious"
    else:
        level = "ok"
    matched = {
        name: sorted(detail) if isinstance(detail, list) else detail
        for name, (points, detail) in text_hits.items()
        if points
    }
    if best is not None:
        if best.brand is not None and not best.brand.official:
            matched["brand"] = [best.brand.display]
        if best.punycode or (
            best.brand is not None and best.brand.digit_swap
            and not best.brand.official
        ):
            matched["lookalike"] = [best.url]
        if best.risky_tld is not None:
            matched["risky_tld"] = [best.risky_tld]
        if best.free_host is not None:
            matched["free_host"] = [best.free_host]
        if best.actions:
            matched["url_actions"] = list(best.actions)
        if best.shortener:
            matched["shortener"] = [best.url]
        if best.download_ext is not None:
            matched["download"] = [best.download_ext]
    return LexiconResult(
        score=score,
        level=level,
        reasons=_reasons(text_hits, best, language) if level != "ok" else [],
        download=download,
        matched=matched,
    )
