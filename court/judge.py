"""Judge: weighs only cited points and returns the verdict.

Deterministic rules enforced here (never left to the model alone):
- points whose source_url is not in the evidence list (after URL
  normalization) are dropped before judging;
- with zero cited points remaining, the verdict is Unverified/low;
- key_source must come from the evidence list, else it falls back to the
  most-cited kept URL (then the first evidence URL);
- when intake marks the forward presented-as-recent and the earliest
  non-visual source is more than RECYCLED_DAYS older than the forward's
  implied date, old content presented as new is Misleading, citing the
  earliest source;
- for image forwards without exact-match reverse evidence, the image's origin
  cannot be ruled True, or False when all claims are image claims (capped at
  Unverified/low).
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from config import agent_model
from evidence.urls import normalize_url

from .models import Argument, ImageContext, Verdict, evidence_brief

RECYCLED_DAYS = 30

RECYCLED_NOTES = {
    "en": "Note: the sources predate what the forward describes (recycled old content).",
    "hi": "नोट: स्रोत उस घटना से पुराने हैं जिसका वर्णन फ़ॉरवर्ड में है (पुरानी खबर दोबारा)।",
}

INSTRUCTIONS = (
    "You are the judge in a fact-check court. Both sides argued using only the "
    "evidence list; every point below already cites a URL from it. "
    "Return True or False only on strong cited evidence. Return Misleading "
    "when real content is presented with the wrong context or timeframe, "
    "including old news recycled as new. Otherwise return Unverified. "
    "Never claim 'deepfake' without strong evidence. "
    "Write reason in the forward's language (given below). "
    "key_source must be one source_url copied EXACTLY from the evidence list. "
    "claim_timeframe is the ISO date the forward implies the event happened "
    "(YYYY-MM-DD), or null when the forward implies no timeframe. "
    "Entries marked visually similar are context only, never proof of origin "
    "or date. When told no exact-match reverse evidence exists, do not rule "
    "the image's origin True or False."
)


def _parse_ymd(value: str | None) -> date | None:
    """Parse YYYY-MM-DD, YYYY-MM, or YYYY leniently; else None."""
    if not value:
        return None
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        parts = [int(p) for p in text.split("-")]
    except ValueError:
        return None
    try:
        if len(parts) == 2:
            return date(parts[0], parts[1], 1)
        if len(parts) == 1:
            return date(parts[0], 1, 1)
    except ValueError:
        return None
    return None


def _keep_cited(argument: Argument, evidence_urls: set[str]) -> list:
    return [
        point
        for point in argument.points
        if point.text.strip() and normalize_url(point.source_url) in evidence_urls
    ]


def judge_verdict(
    claims: list[str],
    evidence,
    prosecution: Argument,
    defense: Argument,
    language: str,
    presented_as_recent: bool,
    *,
    client,
    tracker=None,
    today: date | None = None,
    image: ImageContext | None = None,
) -> Verdict:
    evidence_urls = {normalize_url(item.source_url) for item in evidence}
    kept_prosecution = _keep_cited(prosecution, evidence_urls)
    kept_defense = _keep_cited(defense, evidence_urls)

    image_line = ""
    if image is not None:
        image_line = (
            f"Image forward: exact-match reverse evidence "
            f"{'present' if image.reverse_evidence else 'absent'}; "
            f"claims are "
            f"{'all from the image' if image.origin_only else 'image plus caption'}.\n"
        )
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=(
            f"Forward language: {language}\n{image_line}Claims:\n{claims}\n\n"
            f"Evidence:\n{evidence_brief(evidence)}\n\n"
            f"Prosecution points kept: "
            f"{[p.model_dump() for p in kept_prosecution]}\n"
            f"Defense points kept: {[p.model_dump() for p in kept_defense]}"
        ),
        text_format=Verdict,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="judge")
    verdict = response.output_parsed

    if normalize_url(verdict.key_source) not in evidence_urls:
        cited = Counter(
            normalize_url(p.source_url) for p in kept_prosecution + kept_defense
        )
        if cited:
            top = cited.most_common(1)[0][0]
            fallback = next(
                e.source_url for e in evidence if normalize_url(e.source_url) == top
            )
        else:
            fallback = evidence[0].source_url
        verdict = verdict.model_copy(update={"key_source": fallback})

    if not kept_prosecution and not kept_defense:
        return verdict.model_copy(
            update={"verdict": "Unverified", "confidence": "low"}
        )

    if presented_as_recent:
        dated = sorted(
            (parsed, item.source_url)
            for item in evidence
            if not getattr(item, "visual_only", False)
            and (parsed := _parse_ymd(item.date)) is not None
        )
        claim_date = _parse_ymd(verdict.claim_timeframe) or (
            today or date.today()
        )
        if dated and (claim_date - dated[0][0]).days > RECYCLED_DAYS:
            note = RECYCLED_NOTES.get(language, RECYCLED_NOTES["en"])
            verdict = verdict.model_copy(
                update={
                    "verdict": "Misleading",
                    "reason": f"{verdict.reason} {note}",
                    "key_source": dated[0][1],
                }
            )
    if image is not None and not image.reverse_evidence:
        if verdict.verdict == "True" or (
            image.origin_only and verdict.verdict == "False"
        ):
            verdict = verdict.model_copy(
                update={"verdict": "Unverified", "confidence": "low"}
            )
    if verdict.verdict == "Unverified" and verdict.confidence == "high":
        verdict = verdict.model_copy(update={"confidence": "medium"})
    return verdict
