"""WhatsApp-style verdict card formatting (pure, no LLM calls)."""
from __future__ import annotations

from card.corroborate import outlet_label, select_corroboration
from court.judge import CLAIM_REVIEW_REASONS

LABELS = {
    "en": {
        "title": "Forward Court Verdict",
        "confidence": "Confidence",
        "source": "Source",
        "fact_checked": "Already fact-checked by {outlet}",
        "also_confirmed_by": "Also confirmed by",
        "dangerous_title": "DANGEROUS LINK — do not open",
        "safety": "Never share OTP/PIN. Report: cybercrime.gov.in / 1930",
        "article_from": "Article from {date}",
    },
    "hi": {
        "title": "फॉरवर्ड कोर्ट निर्णय",
        "confidence": "विश्वास",
        "source": "स्रोत",
        "fact_checked": "पहले से तथ्य-जांचित: {outlet}",
        "also_confirmed_by": "इनसे भी पुष्टि",
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


def _is_claim_review_boilerplate(verdict) -> bool:
    """True when the reason merely restates the fact-checked line."""
    outlet = getattr(verdict, "fact_checked_by", None)
    if not outlet:
        return False
    template = CLAIM_REVIEW_REASONS.get(
        verdict.language, CLAIM_REVIEW_REASONS["en"]
    )
    return verdict.reason == template.format(
        outlet=outlet, rating=getattr(verdict, "fact_check_rating", None)
    )


def _corroboration_block(verdict, claims, evidence, labels) -> list[str]:
    """'Also confirmed by' outlet line + links; [] when none qualify."""
    if not evidence or not claims:
        return []
    extras = select_corroboration(verdict, claims, evidence)
    if not extras:
        return []
    names = []
    for item in extras:
        name = outlet_label(item.source_url)
        if item.claim_review is not None:
            name += f" [{item.claim_review.rating_raw}]"
        names.append(name)
    block = [
        "",
        f"{labels['also_confirmed_by']}: {', '.join(names)}",
    ]
    block.extend(f"🔗 {item.source_url}" for item in extras)
    return block


def format_card(verdict, evidence=None, claims=None) -> str:
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
    checked_by = getattr(verdict, "fact_checked_by", None)
    rating = getattr(verdict, "fact_check_rating", None)
    if checked_by:
        line = labels["fact_checked"].format(outlet=checked_by)
        if rating:
            line += f" [{rating}]"
        lines.append(line)
    if not _is_claim_review_boilerplate(verdict):
        lines.extend(["", verdict.reason])
    if getattr(verdict, "article_date", None):
        lines.extend(
            ["", labels["article_from"].format(date=verdict.article_date)]
        )
    if verdict.key_source:
        lines.extend(["", f"🔗 {labels['source']}: {verdict.key_source}"])
    lines.extend(_corroboration_block(verdict, claims, evidence, labels))
    return "\n".join(lines)
