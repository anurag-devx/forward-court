"""CLI entry: python main.py "<forward text>"

Runs the full pipeline: intake -> evidence -> debate -> judge -> card.
Prints the verdict card plus per-run token usage to stdout.
"""
from __future__ import annotations

import argparse
import sys
import threading

from card.formatter import format_card
from config import MAX_CLAIMS, UsageTracker, get_client, serpapi_key
from court.defender import argue_true
from court.judge import (
    find_claim_review_verdict,
    has_media_tied_evidence,
    judge_verdict,
    mismatch_verdict,
    no_evidence_verdict,
)
from court.models import ImageContext, Verdict
from court.prosecutor import argue_false
from evidence.factcheck import lookup_factchecks
from evidence.grounder import GroundingError, gather_evidence
from evidence.relevance import filter_evidence
from evidence.reverse_image import (
    ReverseSearchError,
    SerpApiLensProvider,
    matches_to_evidence,
    search_images,
)
from evidence.urls import normalize_url
from intake.extractor import extract_intake
from intake.recency import is_presented_as_recent
from media.audio import TranscribeError, extract_audio, transcribe_audio
from media.segment import segment_subjects
from media.video import describe_video, extract_key_frames, read_video
from media.vision import describe_image, read_image


RUN_DEADLINE_S = 240.0

TIMEOUT_REASONS = {
    "en": "Couldn't verify right now, please try again.",
    "hi": "अभी सत्यापित नहीं हो सका, कृपया पुनः प्रयास करें।",
}


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
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on forward text. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        intake = extract_intake(text, client=client, tracker=tracker)
        tracker.language = intake.language
        if not intake.claims:
            return "No checkable claims found in this forward.", tracker
        evidence = gather_evidence(
            intake.claims, client=client, tracker=tracker, http_client=http_client
        )
        if factcheck:
            evidence = lookup_factchecks(
                intake.claims, client=client, tracker=tracker,
                http_client=http_client, language=intake.language,
            ) + evidence
        else:
            tracker.note("fact-check lookup disabled.")
        evidence, bare_claims = filter_evidence(
            intake.claims, evidence, client=client, tracker=tracker
        )
        if not evidence:
            tracker.note("no relevant evidence: verdict Unverified.")
            return format_card(no_evidence_verdict(intake.language)), tracker
        review_verdict = find_claim_review_verdict(
            intake.claims, evidence, intake.language
        )
        if review_verdict is not None:
            tracker.note(
                f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                f"{review_verdict.verdict}."
            )
            return format_card(review_verdict), tracker
        prosecution = argue_false(intake.claims, evidence, client=client, tracker=tracker)
        defense = argue_true(intake.claims, evidence, client=client, tracker=tracker)
        verdict = judge_verdict(
            intake.claims,
            evidence,
            prosecution,
            defense,
            intake.language,
            intake.presented_as_recent,
            client=client,
            tracker=tracker,
            no_evidence_claims=bare_claims,
        )
        return format_card(verdict), tracker
    return _run_with_deadline(tracker, _go, deadline_s)


def run_image_forward(
    image_path: str, caption: str | None = None, *, client=None,
    http_client=None, provider=None, factcheck=True,
    ground_truth: list[str] | None = None,
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on an image forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        image_bytes, mime = read_image(image_path)
        vision = describe_image(
            image_bytes, mime=mime, client=client, tracker=tracker,
            context=caption,
        )
        media_mismatch: str | None = None
        if vision.shows_claimed_event == "no":
            media_mismatch = vision.mismatch_reason or "media shows other content"
            tracker.note(f"media_mismatch: {media_mismatch}")
        caption_claims: list[str] = []
        caption_media: list[str] = []
        language = vision.language
        if caption:
            intake = extract_intake(caption, client=client, tracker=tracker)
            caption_claims = intake.claims
            caption_media = intake.media_claims
            language = intake.language
        tracker.language = language
        presented_as_recent = is_presented_as_recent(
            [vision.visible_text, caption or ""]
        )
        claims = (vision.claims_from_image + caption_claims)[:MAX_CLAIMS]
        if not claims:
            return "No checkable claims found in this forward.", tracker
        media_for_judge = [
            claim for claim in claims
            if claim in set(vision.claims_from_image + caption_media)
        ]
        reverse_items, reverse_evidence = [], False
        exact_urls: set[str] = set()
        active_provider = provider
        if active_provider is None and serpapi_key() is not None:
            active_provider = SerpApiLensProvider(
                serpapi_key(), http_client=http_client
            )
        if active_provider is None:
            tracker.note("reverse search skipped: SERPAPI_KEY not set.")
        else:
            try:
                crops = segment_subjects(
                    image_bytes, vision.people_places_objects, mime=mime,
                    client=client, tracker=tracker,
                )
                tracker.mark("reverse")
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
        evidence = reverse_items + gather_evidence(
            claims, client=client, tracker=tracker, http_client=http_client
        )
        if factcheck:
            evidence = lookup_factchecks(
                claims, client=client, tracker=tracker, http_client=http_client,
                language=language,
            ) + evidence
        else:
            tracker.note("fact-check lookup disabled.")
        evidence, bare_claims = filter_evidence(
            claims, evidence, client=client, tracker=tracker
        )
        if not evidence:
            tracker.note("no relevant evidence: verdict Unverified.")
            return format_card(no_evidence_verdict(language)), tracker
        review_verdict = find_claim_review_verdict(
            claims, evidence, language, media=True, exact_urls=exact_urls
        )
        if review_verdict is not None:
            tracker.note(
                f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                f"{review_verdict.verdict}."
            )
            return format_card(review_verdict), tracker
        if media_mismatch is not None and not has_media_tied_evidence(
            evidence, exact_urls, media_for_judge
        ):
            agreement = vision.consistency_agreement
            tracker.note(f"media mismatch verdict (agreement {agreement}/3).")
            return (
                format_card(mismatch_verdict(media_mismatch, language, agreement)),
                tracker,
            )
        prosecution = argue_false(claims, evidence, client=client, tracker=tracker)
        defense = argue_true(claims, evidence, client=client, tracker=tracker)
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
            ),
            no_evidence_claims=bare_claims,
            media_claims=media_for_judge,
            exact_urls=exact_urls,
            media_mismatch=media_mismatch,
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
    http_client=None, provider=None, factcheck=True,
    deadline_s: float = RUN_DEADLINE_S,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on a video forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    def _go() -> tuple[str, UsageTracker]:
        video_bytes, resolved = read_video(video_path)
        video = describe_video(
            video_bytes, client=client, tracker=tracker, context=caption
        )
        media_mismatch: str | None = None
        if video.shows_claimed_event == "no":
            media_mismatch = video.mismatch_reason or "media shows other content"
            tracker.note(f"media_mismatch: {media_mismatch}")
        transcript_claims: list[str] = []
        transcript_media: list[str] = []
        transcript = ""
        language = video.language
        audio = extract_audio(resolved)
        if audio is not None:
            try:
                transcript = transcribe_audio(
                    audio, http_client=http_client, tracker=tracker
                )
            except TranscribeError as exc:
                tracker.note(f"transcription unavailable: {exc}")
                transcript = ""
            if transcript.strip():
                spoken = extract_intake(transcript, client=client, tracker=tracker)
                transcript_claims = spoken.claims
                transcript_media = spoken.media_claims
                language = spoken.language
        else:
            tracker.note("no usable audio track: transcript skipped.")
        caption_claims: list[str] = []
        caption_media: list[str] = []
        if caption:
            intake = extract_intake(caption, client=client, tracker=tracker)
            caption_claims = intake.claims
            caption_media = intake.media_claims
            language = intake.language
        tracker.language = language
        presented_as_recent = is_presented_as_recent(
            [video.visible_text, transcript, caption or ""]
        )
        claims = (
            video.claims_from_video + transcript_claims + caption_claims
        )[:MAX_CLAIMS]
        if not claims:
            return "No checkable claims found in this forward.", tracker
        media_for_judge = [
            claim for claim in claims
            if claim in set(
                video.claims_from_video + transcript_media + caption_media
            )
        ]
        reverse_items = []
        reverse_evidence = False
        exact_urls: set[str] = set()
        active_provider = provider
        if active_provider is None and serpapi_key() is not None:
            active_provider = SerpApiLensProvider(
                serpapi_key(), http_client=http_client
            )
        if active_provider is None:
            tracker.note("reverse search skipped: SERPAPI_KEY not set.")
        else:
            counting = _CountingProvider(active_provider)
            frames = extract_key_frames(resolved)
            tracker.note(f"key frames extracted: {len(frames)}.")
            seen: set[str] = set()
            try:
                for frame in frames:
                    crops = segment_subjects(
                        frame, video.people_places_objects,
                        client=client, tracker=tracker,
                    )
                    try:
                        tracker.mark("reverse")
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
        evidence = reverse_items + gather_evidence(
            claims, client=client, tracker=tracker, http_client=http_client
        )
        if factcheck:
            evidence = lookup_factchecks(
                claims, client=client, tracker=tracker, http_client=http_client,
                language=language,
            ) + evidence
        else:
            tracker.note("fact-check lookup disabled.")
        evidence, bare_claims = filter_evidence(
            claims, evidence, client=client, tracker=tracker
        )
        if not evidence:
            tracker.note("no relevant evidence: verdict Unverified.")
            return format_card(no_evidence_verdict(language)), tracker
        review_verdict = find_claim_review_verdict(
            claims, evidence, language, media=True, exact_urls=exact_urls
        )
        if review_verdict is not None:
            tracker.note(
                f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
                f"{review_verdict.verdict}."
            )
            return format_card(review_verdict), tracker
        if media_mismatch is not None and not has_media_tied_evidence(
            evidence, exact_urls, media_for_judge
        ):
            agreement = video.consistency_agreement
            tracker.note(f"media mismatch verdict (agreement {agreement}/3).")
            return (
                format_card(mismatch_verdict(media_mismatch, language, agreement)),
                tracker,
            )
        prosecution = argue_false(claims, evidence, client=client, tracker=tracker)
        defense = argue_true(claims, evidence, client=client, tracker=tracker)
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
            ),
            no_evidence_claims=bare_claims,
            media_claims=media_for_judge,
            exact_urls=exact_urls,
            media_mismatch=media_mismatch,
        )
        return format_card(verdict), tracker
    return _run_with_deadline(tracker, _go, deadline_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fact-check a WhatsApp forward with a debating court."
    )
    parser.add_argument("text", nargs="?", help="Forward text (else read stdin).")
    parser.add_argument("--image", default=None, help="Image forward file path.")
    parser.add_argument("--video", default=None, help="Video forward file path.")
    parser.add_argument(
        "--caption", default=None, help="Caption for --image or --video."
    )
    parser.add_argument("--no-factcheck", action="store_true",
                        help="Skip the existing-fact-check lookup stage.")
    args = parser.parse_args(argv)
    if sum(bool(part) for part in (args.text, args.image, args.video)) > 1:
        print("Error: pass forward text, --image, or --video — only one.",
              file=sys.stderr)
        return 2
    if args.caption and not (args.image or args.video):
        print("Error: --caption needs --image or --video.", file=sys.stderr)
        return 2
    use_factcheck = not args.no_factcheck
    try:
        if args.image:
            card, tracker = run_image_forward(
                args.image, args.caption, factcheck=use_factcheck
            )
        elif args.video:
            card, tracker = run_video_forward(
                args.video, args.caption, factcheck=use_factcheck
            )
        else:
            text = args.text if args.text is not None else sys.stdin.read()
            if not text.strip():
                print("Error: provide forward text as an argument or via stdin.",
                      file=sys.stderr)
                return 2
            card, tracker = run_forward(text, factcheck=use_factcheck)
    except GroundingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(card)
    print()
    for line in tracker.breakdown():
        print(line)
    for note in tracker.notes:
        print(f"note: {note}")
    print(tracker.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
