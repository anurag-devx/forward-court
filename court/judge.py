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

from config import agent_model, reasoning_kwargs
from evidence.grounder import EVIDENCE_WORKERS, DaemonPool
from evidence.match import key_mismatched, review_matches_claim
from evidence.urls import normalize_url, split_host_path, tier_of

from .models import Argument, ImageContext, Verdict, evidence_brief

RECYCLED_DAYS = 30
VOTES = 3

RECYCLED_LEAD = {
    "en": "Real {outlet} content from {date}, shared now as new.",
    "hi": "{outlet} की {date} की असली सामग्री अब नई बताकर साझा की गई।",
}

NO_EVIDENCE_REASONS = {
    "en": "No relevant evidence was found for these claims.",
    "hi": "इन दावों के लिए कोई प्रासंगिक साक्ष्य नहीं मिला।",
}

CLAIM_REVIEW_REASONS = {
    "en": "Already fact-checked by {outlet}: rated {rating}.",
    "hi": "{outlet} द्वारा पहले से तथ्य-जांचित: रेटिंग {rating}।",
}

MISMATCH_REASONS = {
    "en": "Media does not show the claimed event: {reason}",
    "hi": "मीडिया दावा की गई घटना नहीं दिखाता: {reason}",
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
    "assessment in the reason."
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


def _best_key(items, kept_points) -> str | None:
    """Most-cited, then first; None when empty."""
    if not items:
        return None
    cited = Counter(normalize_url(point.source_url) for point in kept_points)
    ranked = sorted(
        range(len(items)),
        key=lambda i: (
            -cited.get(normalize_url(items[i].source_url), 0),
            i,
        ),
    )
    return items[ranked[0]].source_url


def _select_true_key(claims, evidence, resolved_key, kept_points) -> str:
    """True-verdict key: veto mismatched fact-checks, prefer major outlets.

    A fact-check about another claim stays as context but can't be
    key_source. Among eligible items, tier-1 reports win, then most-cited,
    then first. Falls back to the resolved key when nothing better
    qualifies.
    """
    pool = [e for e in evidence if tier_of(e.source_url) != 3]
    pool = pool or list(evidence)
    eligible = [e for e in pool if not key_mismatched(e, claims)]
    current = normalize_url(resolved_key)
    current_ok = any(
        normalize_url(e.source_url) == current for e in eligible
    )
    if not eligible or (current_ok and tier_of(resolved_key) == 1):
        return resolved_key
    if current_ok:
        upgraded = _best_key(
            [e for e in eligible if tier_of(e.source_url) == 1],
            kept_points,
        )
        return upgraded if upgraded is not None else resolved_key
    tier1 = _best_key(
        [e for e in eligible if tier_of(e.source_url) == 1], kept_points
    )
    if tier1 is not None:
        return tier1
    return _best_key(eligible, kept_points) or resolved_key


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
        review_matches_claim(item, claim) for claim in (media_claims or [])
    )


def _cast_vote(client, tracker, brief_input: str) -> Verdict:
    """Cast one judge vote. Ballots reassemble in submission order."""
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=brief_input,
        text_format=Verdict,
        **reasoning_kwargs("judge"),
    )
    if tracker is not None:
        tracker.add(response.usage, stage="judge")
    return response.output_parsed


def _keep_cited(argument: Argument, evidence_urls: set[str]) -> list:
    return [
        point
        for point in argument.points
        if point.text.strip() and normalize_url(point.source_url) in evidence_urls
    ]


def outlet_host(url: str) -> str:
    """Outlet host for card display (www stripped, full URL on failure)."""
    host, _ = split_host_path(url)
    return host or url


def find_claim_review_verdict(
    claims, evidence, language: str, *, media: bool = False,
    exact_urls: set[str] | None = None, tracker=None,
) -> Verdict | None:
    """Deterministic verdict from tier-1 ClaimReview ratings, else None.

    Collects mapped ratings from tier-1 items whose ClaimReview matches a
    claim. Unverified mappings never short-circuit. On media forwards the
    winner must be media-tied (an exact reverse hit on its page); untied
    reviews stay as normal debate evidence. One distinct verdict wins
    (citing that URL, high confidence); two distinct verdicts contradict
    and fall through to the normal path. Makes no model calls.
    """
    hits = []
    for item in evidence:
        if tier_of(item.source_url) != 1 or item.claim_review is None:
            continue
        if item.claim_review.verdict == "Unverified":
            continue
        if any(review_matches_claim(item, claim) for claim in claims):
            hits.append(item)
    verdicts = {item.claim_review.verdict for item in hits}
    if tracker is not None:
        tracker.record(
            "claimreview_hits",
            [
                {
                    "source_url": item.source_url,
                    "verdict": item.claim_review.verdict,
                    "rating": item.claim_review.rating_raw,
                }
                for item in hits
            ],
        )
    if len(verdicts) != 1:
        return None
    winner = hits[0]
    if media and normalize_url(winner.source_url) not in (exact_urls or set()):
        return None
    outlet = outlet_host(winner.source_url)
    template = CLAIM_REVIEW_REASONS.get(
        language, CLAIM_REVIEW_REASONS["en"]
    )
    return Verdict(
        verdict=winner.claim_review.verdict,
        reason=template.format(
            outlet=outlet, rating=winner.claim_review.rating_raw
        ),
        key_source=winner.source_url,
        language=language,
        confidence="high",
        fact_checked_by=outlet,
        fact_check_rating=winner.claim_review.rating_raw,
    )


def mismatch_verdict(reason: str, language: str, agreement: int) -> Verdict:
    """Misleading verdict from a voted media mismatch. No key source:
    the verdict rests on the media itself, not on a cited URL."""
    template = MISMATCH_REASONS.get(language, MISMATCH_REASONS["en"])
    return Verdict(
        verdict="Misleading",
        reason=template.format(reason=reason),
        key_source="",
        language=language,
        confidence="high" if agreement >= 3 else "medium",
        claim_timeframe=None,
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
    article_url: str | None = None,
    article_date: str | None = None,
    article_bare: bool = False,
    executor=None,
) -> Verdict:
    if tracker is not None:
        tracker.mark("judge")
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
        if image.impersonated_domain:
            image_line += (
                f"Note: impersonated domain: the media visibly shows "
                f"{image.impersonated_domain}, which spoofs a known outlet.\n"
            )
    no_evidence_line = ""
    if no_evidence_claims:
        no_evidence_line = (
            f"Claims with no kept evidence (treat as Unverified): "
            f"{no_evidence_claims}\n"
        )
    media_set = set(media_claims or [])
    media_section = [claim for claim in claims if claim in media_set]
    event_section = [claim for claim in claims if claim not in media_set]
    brief_input = (
        f"Forward language: {language}\n{image_line}{no_evidence_line}"
        f"Media claims:\n{media_section}\n\n"
        f"Event claims:\n{event_section}\n\n"
        f"Evidence:\n{evidence_brief(evidence)}\n\n"
        f"Prosecution points kept: "
        f"{[p.model_dump() for p in kept_prosecution]}\n"
        f"Defense points kept: {[p.model_dump() for p in kept_defense]}"
    )
    pool = executor if executor is not None else DaemonPool(
        max_workers=EVIDENCE_WORKERS
    )
    try:
        ballots: list[Verdict] = [
            future.result()
            for future in [
                pool.submit(_cast_vote, client, tracker, brief_input)
                for _ in range(VOTES)
            ]
        ]
    finally:
        if executor is None:
            pool.shutdown()
    verdict = _majority(ballots)
    if tracker is not None:
        votes = ", ".join(ballot.verdict for ballot in ballots)
        top = Counter(ballot.verdict for ballot in ballots).most_common(1)[0][1]
        tracker.note(f"judge agreement: {top}/{VOTES} ({votes}).")
        tracker.record(
            "votes",
            [
                {
                    "verdict": ballot.verdict,
                    "reason": ballot.reason,
                    "key_source": ballot.key_source,
                    "confidence": ballot.confidence,
                }
                for ballot in ballots
            ],
        )

    key_source = _resolve_key_source(
        verdict.key_source, evidence, kept_prosecution + kept_defense
    )
    if key_source != verdict.key_source:
        verdict = verdict.model_copy(update={"key_source": key_source})
    if verdict.verdict == "True":
        true_key = _select_true_key(
            claims, evidence, verdict.key_source,
            kept_prosecution + kept_defense,
        )
        if true_key != verdict.key_source:
            verdict = verdict.model_copy(update={"key_source": true_key})
            if tracker is not None:
                tracker.note("key_source moved to an on-claim report.")

    if not kept_prosecution and not kept_defense:
        return verdict.model_copy(
            update={"verdict": "Unverified", "confidence": "low"}
        )

    recycled: tuple[str, str, str] | None = None  # outlet, date, key
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
            recycled = (
                outlet_host(dated[0][1]),
                dated[0][0].isoformat(),
                dated[0][1],
            )
    parsed_article = _parse_ymd(article_date) if article_date else None
    if parsed_article is not None and article_url:
        claim_date = _parse_ymd(verdict.claim_timeframe) or (
            today or date.today()
        )
        if (claim_date - parsed_article).days > RECYCLED_DAYS:
            if presented_as_recent and not article_bare:
                recycled = (
                    outlet_host(article_url), article_date, article_url
                )
            elif article_bare:
                verdict = verdict.model_copy(
                    update={"article_date": article_date}
                )
    if recycled is not None:
        outlet, recycled_date, key = recycled
        lead = RECYCLED_LEAD.get(language, RECYCLED_LEAD["en"]).format(
            outlet=outlet, date=recycled_date
        )
        verdict = verdict.model_copy(
            update={
                "verdict": "Misleading",
                "reason": f"{lead} {verdict.reason}",
                "key_source": key,
            }
        )
    if verdict.verdict == "True":
        shared = {
            normalize_url(item.source_url)
            for item in evidence
            if getattr(item, "shared_article", False)
        }
        if shared and normalize_url(verdict.key_source) in shared:
            independent = next(
                (
                    item.source_url
                    for item in evidence
                    if not getattr(item, "shared_article", False)
                ),
                None,
            )
            if independent is not None:
                verdict = verdict.model_copy(
                    update={"key_source": independent}
                )
                if tracker is not None:
                    tracker.note("key_source moved off shared article.")
            else:
                verdict = verdict.model_copy(
                    update={"verdict": "Unverified", "confidence": "medium"}
                )
                if tracker is not None:
                    tracker.note("True needs independent corroboration.")
    if (
        image is not None
        and not image.reverse_evidence
        and not image.has_content_claims
    ):
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
