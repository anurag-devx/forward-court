"""DANGEROUS decision rule over link + message signals."""
from __future__ import annotations

from court.models import Verdict

STRONG_SIGNALS = (
    "gsb_hit",
    "risky_download",
    "lookalike_domain",
    "punycode_host",
    "raw_ip_host",
    "many_subdomains",
    "at_in_url",
    "new_domain",
)

REASONS = {
    "en": {
        "gsb_hit": "Flagged by Google Safe Browsing ({detail}).",
        "risky_download": "Leads to a dangerous download ({detail}).",
        "lookalike_domain": "Uses a lookalike domain ({detail}).",
        "punycode_host": "Uses a deceptive internationalized address.",
        "raw_ip_host": "Points at a raw IP address, not a named site.",
        "many_subdomains": "Hides behind excessive subdomains.",
        "at_in_url": "Contains a misleading @ in the address.",
        "new_domain": "Domain is brand new ({detail}).",
        "scam_message": "The message itself matches a known scam pattern.",
    },
    "hi": {
        "gsb_hit": "Google Safe Browsing ने इसे खतरनाक बताया ({detail}).",
        "risky_download": "खतरनाक डाउनलोड की ओर ले जाता है ({detail}).",
        "lookalike_domain": "मिलते-जुलते नकली डोमेन का उपयोग ({detail}).",
        "punycode_host": "भ्रामक अंतरराष्ट्रीय पते का उपयोग।",
        "raw_ip_host": "किसी नामित साइट के बजाय सीधे IP पते की ओर।",
        "many_subdomains": "अत्यधिक सबडोमेन के पीछे छिपा है।",
        "at_in_url": "पते में भ्रामक @ है।",
        "new_domain": "डोमेन बिल्कुल नया है ({detail}).",
        "scam_message": "संदेश स्वयं ज्ञात ठगी पैटर्न से मेल खाता है।",
    },
}


def _reason(language: str, name: str, detail: str) -> str:
    template = REASONS.get(language, REASONS["en"]).get(
        name, REASONS["en"][name]
    )
    return template.format(detail=detail)


def decide(
    signals: dict[str, str],
    scam_signals: list[str],
    scam_type: str,
    language: str,
) -> Verdict | None:
    """DANGEROUS verdict when the rule fires, else None (continue)."""
    strong = [name for name in signals if name in STRONG_SIGNALS]
    trigger: str | None = None
    ordered: list[str] = []
    if "gsb_hit" in signals:
        trigger, ordered = "gsb_hit", ["gsb_hit"]
    elif "risky_download" in signals:
        trigger, ordered = "risky_download", ["risky_download"]
    elif "lookalike_domain" in signals and scam_signals:
        trigger = "lookalike_domain"
        ordered = ["lookalike_domain", "scam_message"]
    elif len(strong) >= 3:
        trigger = strong[0]
        ordered = strong[:2]
    elif "otp_pin" in scam_signals and "urgency" in scam_signals:
        trigger, ordered = "scam_message", ["scam_message"]
    elif "install_app" in scam_signals and "payment" in scam_signals:
        trigger, ordered = "scam_message", ["scam_message"]
    if trigger is None:
        return None
    reasons = [
        _reason(language, name, signals.get(name, ""))
        for name in ordered[:2]
    ]
    return Verdict(
        verdict="DANGEROUS",
        reason=" ".join(reasons),
        key_source="",
        language=language,
        confidence="high",
        scam_type=scam_type or None,
    )
