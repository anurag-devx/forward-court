"""WhatsApp-style verdict card formatting (pure, no LLM calls)."""
from __future__ import annotations

LABELS = {
    "en": {
        "title": "Forward Court Verdict",
        "confidence": "Confidence",
        "source": "Source",
        "fact_checked": "Already fact-checked by {outlet}",
        "dangerous_title": "DANGEROUS LINK — do not open",
        "safety": "Never share OTP/PIN. Report: cybercrime.gov.in / 1930",
        "article_from": "Article from {date}",
    },
    "hi": {
        "title": "फॉरवर्ड कोर्ट निर्णय",
        "confidence": "विश्वास",
        "source": "स्रोत",
        "fact_checked": "पहले से तथ्य-जांचित: {outlet}",
        "dangerous_title": "खतरनाक लिंक — न खोलें",
        "safety": "OTP/PIN कभी साझा न करें। शिकायत: cybercrime.gov.in / 1930",
        "article_from": "लेख दिनांक: {date}",
    },
}

VERDICT_EMOJI = {
    "True": "✅",
    "False": "❌",
    "Misleading": "⚠️",
    "Unverified": "❔",
    "DANGEROUS": "⚠️",
}


def format_card(verdict) -> str:
    """Render a short shareable verdict card in the verdict's language."""
    labels = LABELS.get(verdict.language, LABELS["en"])
    if verdict.verdict == "DANGEROUS":
        lines = [f"⚠️ *{labels['dangerous_title']}*"]
        if getattr(verdict, "scam_type", None):
            lines.extend(["", verdict.scam_type])
        lines.extend(["", verdict.reason, "", labels["safety"]])
        return "\n".join(lines)
    emoji = VERDICT_EMOJI.get(verdict.verdict, "❔")
    lines = [
        f"{emoji} *{labels['title']}: {verdict.verdict.upper()}*",
        f"{labels['confidence']}: {verdict.confidence}",
    ]
    if getattr(verdict, "fact_checked_by", None):
        lines.append(
            labels["fact_checked"].format(outlet=verdict.fact_checked_by)
        )
    lines.extend(["", verdict.reason])
    if getattr(verdict, "article_date", None):
        lines.extend(
            ["", labels["article_from"].format(date=verdict.article_date)]
        )
    if verdict.key_source:
        lines.extend(["", f"🔗 {labels['source']}: {verdict.key_source}"])
    return "\n".join(lines)
