"""Judge: weighs only cited points and returns the verdict.

The model votes 3x on the same brief; the majority verdict wins (1-1-1 ties
become Unverified/low) and confidence is voted (high at 3/3, medium at 2/3).
Deterministic rules enforced here (never left to the model alone):
- points whose source_url is not in the evidence list (after URL
  normalization) are dropped before judging;
- with zero cited points remaining, the verdict is Unverified/low;
- key_source must come from the evidence list and never be tier-3, else
  it falls back to the most-cited kept non-tier-3 URL (then the first
  non-tier-3 evidence URL; most-cited anyway when all evidence is tier-3);
- when intake marks the forward presented-as-recent and the earliest
  eligible source is more than RECYCLED_DAYS older than the forward's
  implied date, old content presented as new is Misleading, citing the
  earliest source (media forwards: exact matches and media-tied
  fact-checks only; topic articles never trigger it);
- for image forwards without exact-match reverse evidence, the image's origin
  cannot be ruled True, or False when all claims are image claims (capped at
  Unverified/low).
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from config import agent_model
from evidence.urls import normalize_url, split_host_path, tier_of

from .models import Argument, ImageContext, Verdict, evidence_brief

RECYCLED_DAYS = 30
VOTES = 3
CLAIM_MATCH_TOKENS = 3

RECYCLED_NOTES = {
    "en": "Note: the sources predate what the forward describes (recycled old content).",
    "hi": "नोट: स्रोत उस घटना से पुराने हैं जिसका वर्णन फ़ॉरवर्ड में है (पुरानी खबर दोबारा)।",
}

NO_EVIDENCE_REASONS = {
    "en": "No relevant evidence was found for these claims.",
    "hi": "इन दावों के लिए कोई प्रासंगिक साक्ष्य नहीं मिला।",
}

CLAIM_REVIEW_REASONS = {
    "en": "Already fact-checked by {outlet}: rated {rating}.",
    "hi": "{outlet} द्वारा पहले से तथ्य-जांचित: रेटिंग {rating}।",
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
    "the image's origin True or False. An existing fact-check with a clear "
    "rating weighs heavily: follow it and cite it as key_source when "
    "decisive, unless other cited evidence contradicts it. "
    "Prefer tier-1 sources; never use a tier-3 source as key_source. "
    "Rate the forward on the media claims first; report the event-claim "
    "assessment in the reason. When told the media does not show the "
    "claimed event, that weighs toward Misleading: say 'media does not "
    "show the claimed event' in the reason."
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


def _majority(ballots: list[Verdict]) -> Verdict:
    """Majority verdict; tie (1-1-1) becomes Unverified/low.

    The winning ballot supplies reason/key_source/timeframe. Confidence is
    voted: high at 3/3, medium at 2/3, low on tie.
    """
    counts = Counter(ballot.verdict for ballot in ballots)
    top_verdict, top_count = counts.most_common(1)[0]
    if top_count == 1:
        return ballots[0].model_copy(
            update={"verdict": "Unverified", "confidence": "low"}
        )
    winner = next(b for b in ballots if b.verdict == top_verdict)
    return winner.model_copy(
        update={"confidence": "high" if top_count == 3 else "medium"}
    )


def _resolve_key_source(key_source: str, evidence, kept_points) -> str:
    """Validate key_source: must be cited evidence, never tier-3.

    Falls back to the most-cited non-tier-3 URL, then the first non-tier-3
    evidence URL. When every source is tier-3, keeps most-cited anyway.
    """
    shortlist = [e.source_url for e in evidence if tier_of(e.source_url) != 3]
    pool = shortlist or [e.source_url for e in evidence]
    pool_urls = {normalize_url(url) for url in pool}
    if normalize_url(key_source) in pool_urls:
        return key_source
    cited = Counter(
        normalize_url(point.source_url)
        for point in kept_points
        if normalize_url(point.source_url) in pool_urls
    )
    if cited:
        top = cited.most_common(1)[0][0]
        return next(url for url in pool if normalize_url(url) == top)
    return pool[0]


def _override_eligible(item, media: bool, exact_urls, media_claims) -> bool:
    """True when the item may trigger the recycled-content override.

    Text forwards use any non-visual dated item. Media forwards only use
    media-provenance evidence: exact reverse matches, or fact-checks tied
    to this media (their page is an exact hit, or their ClaimReview
    matches a media claim). Topic articles never trigger it.
    """
    if getattr(item, "visual_only", False):
        return False
    if not media:
        return True
    if getattr(item, "exact_match", False):
        return True
    if not (getattr(item, "fact_check", False) or item.claim_review is not None):
        return False
    if normalize_url(item.source_url) in (exact_urls or set()):
        return True
    return any(
        _review_matches_claim(item, claim) for claim in (media_claims or [])
    )


def _keep_cited(argument: Argument, evidence_urls: set[str]) -> list:
    return [
        point
        for point in argument.points
        if point.text.strip() and normalize_url(point.source_url) in evidence_urls
    ]


def _significant_tokens(text: str) -> set[str]:
    cleaned = "".join(
        char.lower() if char.isalnum() or char.isspace() else " "
        for char in text or ""
    )
    return {token for token in cleaned.split() if len(token) > 3}


def _review_matches_claim(item, claim: str) -> bool:
    """True when the item's ClaimReview applies to the claim.

    Matches on normalized claim text (containment or shared significant
    tokens), or -- for cross-language cases -- when the item came from the
    lookup query for this exact claim with relevance score 2.
    """
    review = item.claim_review
    if review is None:
        return False
    if (
        getattr(item, "fact_check", False)
        and item.claim == claim
        and getattr(item, "relevance", None) == 2
    ):
        return True
    reviewed = (review.claim_text or "").strip()
    if not reviewed:
        return False
    left = " ".join(reviewed.lower().split())
    right = " ".join((claim or "").lower().split())
    if (
        min(len(left), len(right)) >= 8
        and (left in right or right in left)
    ):
        return True
    shared = _significant_tokens(reviewed) & _significant_tokens(claim)
    return len(shared) >= CLAIM_MATCH_TOKENS


def outlet_host(url: str) -> str:
    """Outlet host for card display (www stripped, full URL on failure)."""
    host, _ = split_host_path(url)
    return host or url


def find_claim_review_verdict(claims, evidence, language: str) -> Verdict | None:
    """Deterministic verdict from tier-1 ClaimReview ratings, else None.

    Collects mapped ratings from tier-1 items whose ClaimReview matches a
    claim. One distinct verdict wins (citing that URL, confidence high);
    two distinct verdicts contradict and fall through to the normal path.
    Makes no model calls.
    """
    hits = []
    for item in evidence:
        if tier_of(item.source_url) != 1 or item.claim_review is None:
            continue
        if any(_review_matches_claim(item, claim) for claim in claims):
            hits.append(item)
    verdicts = {item.claim_review.verdict for item in hits}
    if len(verdicts) != 1:
        return None
    winner = hits[0]
    outlet = outlet_host(winner.source_url)
    template = CLAIM_REVIEW_REASONS.get(
        language, CLAIM_REVIEW_REASONS["en"]
    )
    mapped = winner.claim_review.verdict
    return Verdict(
        verdict=mapped,
        reason=template.format(
            outlet=outlet, rating=winner.claim_review.rating_raw
        ),
        key_source=winner.source_url,
        language=language,
        confidence="medium" if mapped == "Unverified" else "high",
        fact_checked_by=outlet,
    )


def no_evidence_verdict(language: str) -> Verdict:
    """Unverified/low verdict when no relevant evidence survived filtering."""
    return Verdict(
        verdict="Unverified",
        reason=NO_EVIDENCE_REASONS.get(
            language, NO_EVIDENCE_REASONS["en"]
        ),
        key_source="",
        language=language,
        confidence="low",
    )


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
    no_evidence_claims: list[str] | None = None,
    media_claims: list[str] | None = None,
    exact_urls: set[str] | None = None,
    media_mismatch: str | None = None,
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
    no_evidence_line = ""
    if no_evidence_claims:
        no_evidence_line = (
            f"Claims with no kept evidence (treat as Unverified): "
            f"{no_evidence_claims}\n"
        )
    mismatch_line = ""
    if media_mismatch:
        mismatch_line = (
            f"Media mismatch: media does not show the claimed event "
            f"({media_mismatch}).\n"
        )
    media_set = set(media_claims or [])
    media_section = [claim for claim in claims if claim in media_set]
    event_section = [claim for claim in claims if claim not in media_set]
    brief_input = (
        f"Forward language: {language}\n{image_line}{no_evidence_line}"
        f"{mismatch_line}"
        f"Media claims:\n{media_section}\n\n"
        f"Event claims:\n{event_section}\n\n"
        f"Evidence:\n{evidence_brief(evidence)}\n\n"
        f"Prosecution points kept: "
        f"{[p.model_dump() for p in kept_prosecution]}\n"
        f"Defense points kept: {[p.model_dump() for p in kept_defense]}"
    )
    ballots: list[Verdict] = []
    for _ in range(VOTES):
        response = client.responses.parse(
            model=agent_model(),
            instructions=INSTRUCTIONS,
            input=brief_input,
            text_format=Verdict,
        )
        if tracker is not None:
            tracker.add(response.usage, stage="judge")
        ballots.append(response.output_parsed)
    verdict = _majority(ballots)
    if tracker is not None:
        votes = ", ".join(ballot.verdict for ballot in ballots)
        top = Counter(ballot.verdict for ballot in ballots).most_common(1)[0][1]
        tracker.note(f"judge agreement: {top}/{VOTES} ({votes}).")

    key_source = _resolve_key_source(
        verdict.key_source, evidence, kept_prosecution + kept_defense
    )
    if key_source != verdict.key_source:
        verdict = verdict.model_copy(update={"key_source": key_source})

    if not kept_prosecution and not kept_defense:
        return verdict.model_copy(
            update={"verdict": "Unverified", "confidence": "low"}
        )

    if presented_as_recent:
        dated = sorted(
            (parsed, item.source_url)
            for item in evidence
            if _override_eligible(
                item, image is not None, exact_urls, media_claims
            )
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
    if tracker is not None and any(
        getattr(item, "fact_check", False) for item in evidence
    ):
        cited = {
            normalize_url(point.source_url)
            for point in kept_prosecution + kept_defense
        }
        cited.add(normalize_url(verdict.key_source))
        used = any(
            normalize_url(item.source_url) in cited
            for item in evidence
            if getattr(item, "fact_check", False)
        )
        tracker.note(f"existing fact-check cited: {'yes' if used else 'no'}.")
    return verdict
