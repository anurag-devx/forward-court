"""CLI entry: python main.py "<forward text>"

Runs the full pipeline: intake -> evidence -> debate -> judge -> card.
Prints the verdict card plus per-run token usage to stdout.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from card.formatter import format_card
from config import (
    MAX_CLAIMS,
    UsageTracker,
    get_client,
    reverse_search_enabled,
    serpapi_key,
    stage_timer,
    zoom_enabled,
)
from court.defender import argue_true
from court.judge import (
    find_claim_review_verdict,
    judge_verdict,
    mismatch_verdict,
    no_evidence_verdict,
)
from court.models import ImageContext, Verdict
from court.prosecutor import argue_false
from evidence.factcheck import lookup_factchecks
from evidence.grounder import (
    EVIDENCE_WORKERS,
    DaemonPool,
    GroundingError,
    gather_evidence,
)
from evidence.relevance import filter_evidence
from evidence.select import select_evidence
from evidence.reverse_image import (
    ReverseSearchError,
    SerpApiLensProvider,
    matches_to_evidence,
    search_images,
)
from evidence.urls import impersonates_outlet, normalize_url, same_source
from links.article import (
    article_tier_note,
    fetch_article,
    is_bare_link,
    to_evidence as article_evidence,
)
from links.check import check_links
from intake.extractor import extract_intake
from intake.recency import is_presented_as_recent
from media.audio import TranscribeError, extract_audio, transcribe_audio
from media.segment import (
    TEXT_PROMPTS,
    RankedRegion,
    rank_regions,
    tight_crops,
    zoom_crops,
)
from media.video import (
    ZOOM_FRAMES,
    describe_video,
    extract_key_frames,
    merge_video_zoom,
    read_video,
)
from media.vision import (
    describe_image,
    merge_zoom,
    read_image,
    zoom_views,
)


RUN_DEADLINE_S = 240.0

TIMEOUT_REASONS = {
    "en": "Couldn't verify right now, please try again.",
    "hi": "अभी सत्यापित नहीं हो सका, कृपया पुनः प्रयास करें।",
}


def _media_claim_kinds(media_claims: list[str], kinds: list[str]) -> list[str]:
    """Kind per media claim; missing tags default to origin_claim (capped)."""
    padded = list(kinds) + ["origin_claim"] * len(media_claims)
    return padded[: len(media_claims)]


def _impersonated_domain(visible_domain: str | None, tracker) -> str | None:
    if visible_domain and impersonates_outlet(visible_domain):
        tracker.note(f"impersonated domain: {visible_domain}.")
        return visible_domain
    return None


def _reverse_provider(provider, http_client):
    """Explicit provider, or SerpAPI from env when enabled; else None."""
    if provider is not None:
        return provider
    if reverse_search_enabled() and serpapi_key() is not None:
        return SerpApiLensProvider(
            serpapi_key(), http_client=http_client
        )
    return None


def _rank_media(
    image_bytes: bytes, subjects: list[str], mime: str,
    has_content_claims: bool, *, client, tracker,
) -> list[RankedRegion]:
    """One SAM ranking for zoom + reverse; text prompts on screenshots."""
    extra = list(TEXT_PROMPTS) if (
        zoom_enabled() and has_content_claims
    ) else []
    with stage_timer(tracker, "segment"):
        return rank_regions(
            image_bytes, subjects, extra, mime=mime, client=client,
            tracker=tracker,
        )


def _apply_zoom(
    image_bytes: bytes, base, ranked: list[RankedRegion], merge, *,
    mime: str, client, tracker,
):
    """Zoom pass over SAM crops. Returns (merged result, domain conflict).

    Never raises for zoom problems: failures keep the pass-1 result.
    """
    if not zoom_enabled():
        tracker.note("zoom disabled (ZOOM=off).")
        return base, False
    crops = zoom_crops(image_bytes, ranked)
    if not crops:
        tracker.note("zoom skipped (SAM found no regions).")
        return base, False
    reading = zoom_views(
        image_bytes, [crop.data for crop in crops], mime=mime,
        client=client, tracker=tracker,
    )
    if reading is None:
        tracker.note("zoom skipped (zoom call failed).")
        return base, False
    merged, conflicted = merge(base, reading)
    tracker.note(f"zoom merged {len(crops)} crop(s).")
    return merged, conflicted


def _run_with_deadline(tracker, fn, deadline_s: float):
    """Run fn on a worker thread; on expiry return a timeout card.

    The hung worker is abandoned as a daemon thread. The stalled stage
    comes from the tracker's last mark().
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(deadline_s)
    if worker.is_alive():
        tracker.note(
            f"run timed out during stage: {tracker.current or 'startup'}."
        )
        verdict = Verdict(
            verdict="Unverified",
            reason=TIMEOUT_REASONS.get(
                tracker.language, TIMEOUT_REASONS["en"]
            ),
            key_source="",
            language=tracker.language,
            confidence="low",
        )
        return format_card(verdict), tracker
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_forward(
    text: str, *, client=None, http_client=None, factcheck=True,
    claimreview=True,
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on forward text. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        with stage_timer(tracker, "intake"):
            intake = extract_intake(text, client=client, tracker=tracker)
        tracker.language = intake.language
        link_verdict, link_info = check_links(
            text, intake.language, client=client, http_client=http_client,
            tracker=tracker,
        )
        if link_verdict is not None:
            return format_card(link_verdict), tracker
        blocked = frozenset(link_info["flagged_hosts"])
        article = None
        if link_info["safe_urls"]:
            with stage_timer(tracker, "article"):
                article = fetch_article(
                    link_info["safe_urls"][0], http_client
                )
        article_bare = bool(article) and is_bare_link(text, link_info["urls"])
        if article is not None:
            note = article_tier_note(article)
            if note is not None:
                tracker.note(note)
        claims = list(intake.claims)
        if article is not None and article.headline:
            claims = [article.headline] + [
                claim for claim in claims if claim != article.headline
            ]
        claims = claims[:MAX_CLAIMS]
        if not claims:
            return "No checkable claims found in this forward.", tracker
        with DaemonPool(max_workers=EVIDENCE_WORKERS) as pool:
            grounding = pool.submit(
                gather_evidence, claims, client=client,
                tracker=tracker, http_client=http_client, executor=pool,
                blocked_hosts=blocked,
            )
            lookup = None
            if factcheck:
                lookup = pool.submit(
                    lookup_factchecks, claims, client=client,
                    tracker=tracker, http_client=http_client,
                    language=intake.language, executor=pool,
                    blocked_hosts=blocked,
                )
            with stage_timer(tracker, "grounding"):
                evidence = grounding.result()
            if lookup is not None:
                with stage_timer(tracker, "factcheck"):
                    evidence = lookup.result() + evidence
            else:
                tracker.note("fact-check lookup disabled.")
            evidence = select_evidence(
                claims, evidence, tracker=tracker
            )
            with stage_timer(tracker, "relevance"):
                evidence, bare_claims = filter_evidence(
                    claims, evidence, client=client, tracker=tracker
                )
            if article is not None:
                evidence.append(article_evidence(article, claims[0]))
            if not evidence:
                tracker.note("no relevant evidence: verdict Unverified.")
                return format_card(no_evidence_verdict(intake.language)), tracker
            review_verdict = None
            if claimreview:
                review_verdict = find_claim_review_verdict(
                    claims, evidence, intake.language
                )
            else:
                tracker.note("ClaimReview short-circuit disabled.")
            if review_verdict is not None:
                tracker.note(
                    f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                    f"{review_verdict.verdict}."
                )
                return format_card(review_verdict), tracker
            prosecution_job = pool.submit(
                argue_false, claims, evidence, client=client,
                tracker=tracker,
            )
            defense_job = pool.submit(
                argue_true, claims, evidence, client=client,
                tracker=tracker,
            )
            debate_start = time.monotonic()
            prosecution = prosecution_job.result()
            tracker.add_time("prosecutor", time.monotonic() - debate_start)
            defense = defense_job.result()
            tracker.add_time("defender", time.monotonic() - debate_start)
            with stage_timer(tracker, "judge"):
                verdict = judge_verdict(
                    claims,
                    evidence,
                    prosecution,
                    defense,
                    intake.language,
                    intake.presented_as_recent,
                    client=client,
                    tracker=tracker,
                    no_evidence_claims=bare_claims,
                    article_url=article.url if article else None,
                    article_date=article.date if article else None,
                    article_bare=article_bare,
                    executor=pool,
                )
            return format_card(verdict), tracker
    return _run_with_deadline(tracker, _go, deadline_s)


def run_image_forward(
    image_path: str, caption: str | None = None, *, client=None,
    http_client=None, provider=None, factcheck=True, claimreview=True,
    ground_truth: list[str] | None = None,
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on an image forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        image_bytes, mime = read_image(image_path)
        with stage_timer(tracker, "vision"):
            vision = describe_image(
                image_bytes, mime=mime, client=client, tracker=tracker,
                context=caption,
            )
        caption_claims: list[str] = []
        caption_media: list[str] = []
        language = vision.language
        if caption:
            with stage_timer(tracker, "intake"):
                intake = extract_intake(caption, client=client, tracker=tracker)
            caption_claims = intake.claims
            caption_media = intake.media_claims
            language = intake.language
        tracker.language = language
        link_blocked: frozenset = frozenset()
        if caption:
            link_verdict, link_info = check_links(
                caption, language, client=client, http_client=http_client,
                tracker=tracker,
            )
            if link_verdict is not None:
                return format_card(link_verdict), tracker
            link_blocked = frozenset(link_info["flagged_hosts"])
        claims = (vision.claims_from_image + caption_claims)[:MAX_CLAIMS]
        kept_kinds = (
            _media_claim_kinds(vision.claims_from_image, vision.claim_kinds)
            + ["content_claim"] * len(caption_claims)
        )[:MAX_CLAIMS]
        has_content_claims = "content_claim" in kept_kinds
        if not claims:
            return "No checkable claims found in this forward.", tracker
        if vision.shows_claimed_event == "no":
            reason = vision.mismatch_reason or "media shows other content"
            tracker.note(f"media_mismatch: {reason}")
            agreement = vision.consistency_agreement
            tracker.note(f"media mismatch verdict (agreement {agreement}/3).")
            return (
                format_card(
                    mismatch_verdict(reason, language, agreement)
                ),
                tracker,
            )
        active_provider = _reverse_provider(provider, http_client)
        ranked: list[RankedRegion] = []
        if zoom_enabled() or active_provider is not None:
            ranked = _rank_media(
                image_bytes, vision.people_places_objects, mime,
                has_content_claims, client=client, tracker=tracker,
            )
        vision, domain_conflicted = _apply_zoom(
            image_bytes, vision, ranked, merge_zoom, mime=mime,
            client=client, tracker=tracker,
        )
        if domain_conflicted:
            tracker.note(
                "visible-domain conflict: impersonation check skipped."
            )
            spoofed = None
        else:
            spoofed = _impersonated_domain(vision.visible_domain, tracker)
        presented_as_recent = is_presented_as_recent(
            [vision.visible_text, caption or ""]
        )
        media_for_judge = [
            claim for claim in claims
            if claim in set(vision.claims_from_image + caption_media)
        ]
        reverse_items, reverse_evidence = [], False
        exact_urls: set[str] = set()
        if active_provider is None:
            if not reverse_search_enabled():
                tracker.note("reverse search disabled (REVERSE_SEARCH=off).")
            else:
                tracker.note("reverse search skipped: SERPAPI_KEY not set.")
        else:
            try:
                crops = tight_crops(image_bytes, ranked)
                tracker.mark("reverse")
                with stage_timer(tracker, "reverse"):
                    matches = search_images(
                        image_bytes, [crop.data for crop in crops],
                        provider=active_provider,
                    )
                reverse_items = matches_to_evidence(
                    matches, claims[0], http_client=http_client
                )
                reverse_evidence = any(match.exact for match in matches)
                exact_urls = {
                    normalize_url(match.source_url)
                    for match in matches
                    if match.exact
                }
                tracker.note(
                    f"reverse matches: {len(matches)} "
                    f"({sum(1 for m in matches if m.exact)} exact) "
                    f"from {1 + len(crops)} image(s)."
                )
                if ground_truth:
                    known = {normalize_url(url) for url in ground_truth}
                    found = any(
                        normalize_url(match.source_url) in known
                        for match in matches
                        if match.exact
                    )
                    tracker.note(
                        f"found ground truth: {'yes' if found else 'no'}.")
            except ReverseSearchError as exc:
                tracker.note(f"reverse search unavailable: {exc}")
        with DaemonPool(max_workers=EVIDENCE_WORKERS) as pool:
            grounding = pool.submit(
                gather_evidence, claims, client=client,
                tracker=tracker, http_client=http_client, executor=pool,
                blocked_hosts=link_blocked,
            )
            lookup = None
            if factcheck:
                lookup = pool.submit(
                    lookup_factchecks, claims, client=client,
                    tracker=tracker, http_client=http_client,
                    language=language, executor=pool,
                    blocked_hosts=link_blocked,
                )
            with stage_timer(tracker, "grounding"):
                evidence = reverse_items + grounding.result()
            if lookup is not None:
                with stage_timer(tracker, "factcheck"):
                    evidence = lookup.result() + evidence
            else:
                tracker.note("fact-check lookup disabled.")
            evidence = select_evidence(
                claims, evidence, tracker=tracker
            )
            with stage_timer(tracker, "relevance"):
                evidence, bare_claims = filter_evidence(
                    claims, evidence, client=client, tracker=tracker
                )
            if not evidence:
                tracker.note("no relevant evidence: verdict Unverified.")
                return format_card(no_evidence_verdict(language)), tracker
            review_verdict = None
            if claimreview:
                review_verdict = find_claim_review_verdict(
                    claims, evidence, language, media=True,
                    exact_urls=exact_urls,
                )
            else:
                tracker.note("ClaimReview short-circuit disabled.")
            if review_verdict is not None:
                tracker.note(
                    f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                    f"{review_verdict.verdict}."
                )
                return format_card(review_verdict), tracker
            prosecution_job = pool.submit(
                argue_false, claims, evidence, client=client,
                tracker=tracker,
            )
            defense_job = pool.submit(
                argue_true, claims, evidence, client=client,
                tracker=tracker,
            )
            debate_start = time.monotonic()
            prosecution = prosecution_job.result()
            tracker.add_time("prosecutor", time.monotonic() - debate_start)
            defense = defense_job.result()
            tracker.add_time("defender", time.monotonic() - debate_start)
            with stage_timer(tracker, "judge"):
                verdict = judge_verdict(
                    claims,
                    evidence,
                    prosecution,
                    defense,
                    language,
                    presented_as_recent,
                    client=client,
                    tracker=tracker,
                    image=ImageContext(
                        origin_only=not caption_claims,
                        reverse_evidence=reverse_evidence,
                        has_content_claims=has_content_claims,
                        impersonated_domain=spoofed,
                    ),
                    no_evidence_claims=bare_claims,
                    executor=pool,
                    media_claims=media_for_judge,
                    exact_urls=exact_urls,
                )
            return format_card(verdict), tracker
    return _run_with_deadline(tracker, _go, deadline_s)


class _CountingProvider:
    """Wraps a reverse-search provider and counts image searches."""

    def __init__(self, provider):
        self.provider = provider
        self.searches = 0

    def search(self, image):
        self.searches += 1
        return self.provider.search(image)


def run_video_forward(
    video_path: str, caption: str | None = None, *, client=None,
    http_client=None, provider=None, factcheck=True, claimreview=True,
    ground_truth: list[str] | None = None,
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on a video forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        video_bytes, resolved = read_video(video_path)
        with stage_timer(tracker, "video"):
            video = describe_video(
                video_bytes, client=client, tracker=tracker, context=caption
            )
        language = video.language
        caption_claims: list[str] = []
        caption_media: list[str] = []
        if caption:
            with stage_timer(tracker, "intake"):
                intake = extract_intake(caption, client=client, tracker=tracker)
            caption_claims = intake.claims
            caption_media = intake.media_claims
            language = intake.language
        transcript_claims: list[str] = []
        transcript_media: list[str] = []
        transcript = ""
        transcript_language: str | None = None
        audio = extract_audio(resolved)
        if audio is not None:
            try:
                with stage_timer(tracker, "transcribe"):
                    transcript = transcribe_audio(
                        audio, http_client=http_client, tracker=tracker
                    )
            except TranscribeError as exc:
                tracker.note(f"transcription unavailable: {exc}")
                transcript = ""
            if transcript.strip():
                with stage_timer(tracker, "intake"):
                    spoken = extract_intake(transcript, client=client, tracker=tracker)
                transcript_claims = spoken.claims
                transcript_media = spoken.media_claims
                transcript_language = spoken.language
        else:
            tracker.note("no usable audio track: transcript skipped.")
        if not caption and transcript_language is not None:
            language = transcript_language
        tracker.language = language
        link_blocked: frozenset = frozenset()
        if caption:
            link_verdict, link_info = check_links(
                caption, language, client=client, http_client=http_client,
                tracker=tracker,
            )
            if link_verdict is not None:
                return format_card(link_verdict), tracker
            link_blocked = frozenset(link_info["flagged_hosts"])
        claims = (
            video.claims_from_video + transcript_claims + caption_claims
        )[:MAX_CLAIMS]
        kept_kinds = (
            _media_claim_kinds(video.claims_from_video, video.claim_kinds)
            + ["content_claim"] * (len(transcript_claims) + len(caption_claims))
        )[:MAX_CLAIMS]
        has_content_claims = "content_claim" in kept_kinds
        if not claims:
            return "No checkable claims found in this forward.", tracker
        if video.shows_claimed_event == "no":
            reason = video.mismatch_reason or "media shows other content"
            tracker.note(f"media_mismatch: {reason}")
            agreement = video.consistency_agreement
            tracker.note(f"media mismatch verdict (agreement {agreement}/3).")
            return (
                format_card(
                    mismatch_verdict(reason, language, agreement)
                ),
                tracker,
            )
        active_provider = _reverse_provider(provider, http_client)
        frames: list[bytes] = []
        if zoom_enabled() or active_provider is not None:
            frames = extract_key_frames(resolved)
            tracker.note(f"key frames extracted: {len(frames)}.")
        ranked_frames: list[list[RankedRegion]] = []
        for frame in frames:
            ranked_frames.append(
                _rank_media(
                    frame, video.people_places_objects, "image/jpeg",
                    has_content_claims, client=client, tracker=tracker,
                )
            )
        zoom_targets = frames[:ZOOM_FRAMES]
        if zoom_targets:
            video, domain_conflicted = _apply_zoom(
                zoom_targets[0], video, ranked_frames[0], merge_video_zoom,
                mime="image/jpeg", client=client, tracker=tracker,
            )
        else:
            domain_conflicted = False
            if zoom_enabled():
                tracker.note("zoom skipped (no key frames).")
            else:
                tracker.note("zoom disabled (ZOOM=off).")
        if domain_conflicted:
            tracker.note(
                "visible-domain conflict: impersonation check skipped."
            )
            spoofed = None
        else:
            spoofed = _impersonated_domain(video.visible_domain, tracker)
        presented_as_recent = is_presented_as_recent(
            [video.visible_text, transcript, caption or ""]
        )
        media_for_judge = [
            claim for claim in claims
            if claim in set(
                video.claims_from_video + transcript_media + caption_media
            )
        ]
        reverse_items = []
        reverse_evidence = False
        exact_urls: set[str] = set()
        if active_provider is None:
            if not reverse_search_enabled():
                tracker.note("reverse search disabled (REVERSE_SEARCH=off).")
            else:
                tracker.note("reverse search skipped: SERPAPI_KEY not set.")
        else:
            counting = _CountingProvider(active_provider)
            seen: set[str] = set()
            try:
                for frame, ranked in zip(frames, ranked_frames):
                    crops = tight_crops(frame, ranked)
                    try:
                        tracker.mark("reverse")
                        with stage_timer(tracker, "reverse"):
                            matches = search_images(
                                frame, [crop.data for crop in crops],
                                provider=counting,
                            )
                    except ReverseSearchError as exc:
                        tracker.note(f"reverse search unavailable: {exc}")
                        break
                    if any(match.exact for match in matches):
                        reverse_evidence = True
                    exact_urls.update(
                        normalize_url(match.source_url)
                        for match in matches
                        if match.exact
                    )
                    for item in matches_to_evidence(
                        matches, claims[0], http_client=http_client
                    ):
                        key = normalize_url(item.source_url)
                        if key in seen:
                            continue
                        seen.add(key)
                        reverse_items.append(item)
            finally:
                tracker.note(f"SerpAPI image searches: {counting.searches}.")
            if ground_truth:
                found = any(
                    same_source(known, exact)
                    for known in ground_truth
                    for exact in exact_urls
                )
                tracker.note(
                    f"found ground truth: {'yes' if found else 'no'}.")
        with DaemonPool(max_workers=EVIDENCE_WORKERS) as pool:
            grounding = pool.submit(
                gather_evidence, claims, client=client,
                tracker=tracker, http_client=http_client, executor=pool,
                blocked_hosts=link_blocked,
            )
            lookup = None
            if factcheck:
                lookup = pool.submit(
                    lookup_factchecks, claims, client=client,
                    tracker=tracker, http_client=http_client,
                    language=language, executor=pool,
                    blocked_hosts=link_blocked,
                )
            with stage_timer(tracker, "grounding"):
                evidence = reverse_items + grounding.result()
            if lookup is not None:
                with stage_timer(tracker, "factcheck"):
                    evidence = lookup.result() + evidence
            else:
                tracker.note("fact-check lookup disabled.")
            evidence = select_evidence(
                claims, evidence, tracker=tracker
            )
            with stage_timer(tracker, "relevance"):
                evidence, bare_claims = filter_evidence(
                    claims, evidence, client=client, tracker=tracker
                )
            if not evidence:
                tracker.note("no relevant evidence: verdict Unverified.")
                return format_card(no_evidence_verdict(language)), tracker
            review_verdict = None
            if claimreview:
                review_verdict = find_claim_review_verdict(
                    claims, evidence, language, media=True,
                    exact_urls=exact_urls,
                )
            else:
                tracker.note("ClaimReview short-circuit disabled.")
            if review_verdict is not None:
                tracker.note(
                    f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                    f"{review_verdict.verdict}."
                )
                return format_card(review_verdict), tracker
            prosecution_job = pool.submit(
                argue_false, claims, evidence, client=client,
                tracker=tracker,
            )
            defense_job = pool.submit(
                argue_true, claims, evidence, client=client,
                tracker=tracker,
            )
            debate_start = time.monotonic()
            prosecution = prosecution_job.result()
            tracker.add_time("prosecutor", time.monotonic() - debate_start)
            defense = defense_job.result()
            tracker.add_time("defender", time.monotonic() - debate_start)
            with stage_timer(tracker, "judge"):
                verdict = judge_verdict(
                    claims,
                    evidence,
                    prosecution,
                    defense,
                    language,
                    presented_as_recent,
                    client=client,
                    tracker=tracker,
                    image=ImageContext(
                        origin_only=not caption_claims and not transcript_claims,
                        reverse_evidence=reverse_evidence,
                        has_content_claims=has_content_claims,
                        impersonated_domain=spoofed,
                    ),
                    no_evidence_claims=bare_claims,
                    executor=pool,
                    media_claims=media_for_judge,
                    exact_urls=exact_urls,
                )
            return format_card(verdict), tracker
    return _run_with_deadline(tracker, _go, deadline_s)


def run_demo() -> int:
    """Run the 4 bundled demo samples in sequence. Returns exit code."""
    import os
    import tempfile

    from demo import SAMPLES, make_screenshot

    demo_dir = Path(__file__).resolve().parent / "demo"
    failures = 0
    total_start = time.monotonic()
    # The demo never depends on non-Meta services, whatever .env says.
    saved = {
        key: os.environ.get(key) for key in ("REVERSE_SEARCH", "GSB")
    }
    os.environ["REVERSE_SEARCH"] = "off"
    os.environ["GSB"] = "off"
    try:
        with tempfile.TemporaryDirectory(
            prefix="forwardcourt-demo-"
        ) as tmp:
            for number, name in enumerate(SAMPLES, start=1):
                print(f"=== Demo {number}/{len(SAMPLES)}: {name} ===")
                start = time.monotonic()
                try:
                    if name == "text":
                        text = (demo_dir / "text.txt").read_text().strip()
                        print(f"input: {text}")
                        card, _ = run_forward(text)
                    elif name == "screenshot":
                        shot = make_screenshot(
                            str(Path(tmp) / "demo_shot.png")
                        )
                        print(f"input: generated screenshot {shot}")
                        card, _ = run_image_forward(shot)
                    elif name == "link":
                        text = (demo_dir / "link.txt").read_text().strip()
                        print(f"input: {text}")
                        card, _ = run_forward(text)
                    elif name == "scam":
                        text = (demo_dir / "scam.txt").read_text().strip()
                        print(f"input: {text}")
                        card, _ = run_forward(text)
                    else:  # pragma: no cover - SAMPLES is fixed
                        raise ValueError(f"unknown demo sample: {name}")
                except (GroundingError, RuntimeError, ValueError) as exc:
                    print(f"sample {name} failed: {exc}")
                    failures += 1
                else:
                    print(card)
                print(
                    f"--- {name} took "
                    f"{time.monotonic() - start:.0f}s ---\n"
                )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(f"demo total: {time.monotonic() - total_start:.0f}s, "
          f"{len(SAMPLES) - failures}/{len(SAMPLES)} samples ok.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fact-check a WhatsApp forward with a debating court."
    )
    parser.add_argument("text", nargs="?", help="Forward text (else read stdin).")
    parser.add_argument("--demo", action="store_true",
                        help="Run the 4 bundled demo samples in sequence.")
    parser.add_argument("--image", default=None, help="Image forward file path.")
    parser.add_argument("--video", default=None, help="Video forward file path.")
    parser.add_argument(
        "--caption", default=None, help="Caption for --image or --video."
    )
    parser.add_argument("--no-factcheck", action="store_true",
                        help="Skip the existing-fact-check lookup stage.")
    parser.add_argument("--no-claimreview", action="store_true",
                        help="Bench/debug only: disable the ClaimReview "
                        "short-circuit so runs take the debate path.")
    parser.add_argument("--card-out", default=None, metavar="PATH",
                        help="Also write the final card text to PATH.")
    args = parser.parse_args(argv)
    if args.demo:
        if args.text or args.image or args.video or args.caption:
            print("Error: --demo takes no other arguments.", file=sys.stderr)
            return 2
        try:
            return run_demo()
        except RuntimeError as exc:  # e.g. MODEL_API_KEY missing
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    if sum(bool(part) for part in (args.text, args.image, args.video)) > 1:
        print("Error: pass forward text, --image, or --video — only one.",
              file=sys.stderr)
        return 2
    if args.caption and not (args.image or args.video):
        print("Error: --caption needs --image or --video.", file=sys.stderr)
        return 2
    use_factcheck = not args.no_factcheck
    use_claimreview = not args.no_claimreview
    try:
        if args.image:
            card, tracker = run_image_forward(
                args.image, args.caption, factcheck=use_factcheck, claimreview=use_claimreview
            )
        elif args.video:
            card, tracker = run_video_forward(
                args.video, args.caption, factcheck=use_factcheck, claimreview=use_claimreview
            )
        else:
            text = args.text if args.text is not None else sys.stdin.read()
            if not text.strip():
                print("Error: provide forward text as an argument or via stdin.",
                      file=sys.stderr)
                return 2
            card, tracker = run_forward(text, factcheck=use_factcheck, claimreview=use_claimreview)
    except GroundingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(card)
    if args.card_out:
        Path(args.card_out).write_text(card + "\n")
    print()
    for line in tracker.breakdown():
        print(line)
    for note in tracker.notes:
        print(f"note: {note}")
    print(tracker.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
