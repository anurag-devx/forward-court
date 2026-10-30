"""WhatsApp-style verdict card formatting (pure, no LLM calls)."""
from __future__ import annotations

LABELS = {
    "en": {
        "title": "Forward Court Verdict",
        "confidence": "Confidence",
        "source": "Source",
        "fact_checked": "Already fact-checked by {outlet}",
    },
    "hi": {
        "title": "फॉरवर्ड कोर्ट निर्णय",
        "confidence": "विश्वास",
        "source": "स्रोत",
        "fact_checked": "पहले से तथ्य-जांचित: {outlet}",
    },
}

VERDICT_EMOJI = {
    "True": "✅",
    "False": "❌",
    "Misleading": "⚠️",
    "Unverified": "❔",
}


def format_card(verdict) -> str:
    """Render a short shareable verdict card in the verdict's language."""
    labels = LABELS.get(verdict.language, LABELS["en"])
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
    if verdict.key_source:
        lines.extend(["", f"🔗 {labels['source']}: {verdict.key_source}"])
    return "\n".join(lines)
