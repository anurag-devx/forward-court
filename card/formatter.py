"""WhatsApp-style verdict card formatting (pure, no LLM calls)."""
from __future__ import annotations

LABELS = {
    "en": {"title": "Forward Court Verdict", "confidence": "Confidence", "source": "Source"},
    "hi": {"title": "फॉरवर्ड कोर्ट निर्णय", "confidence": "विश्वास", "source": "स्रोत"},
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
    return "\n".join(
        [
            f"{emoji} *{labels['title']}: {verdict.verdict.upper()}*",
            f"{labels['confidence']}: {verdict.confidence}",
            "",
            verdict.reason,
            "",
            f"🔗 {labels['source']}: {verdict.key_source}",
        ]
    )
