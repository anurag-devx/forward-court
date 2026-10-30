"""CLI entry: python main.py "<forward text>"

Runs the full pipeline: intake -> evidence -> debate -> judge -> card.
Prints the verdict card plus per-run token usage to stdout.
"""
from __future__ import annotations

import argparse
import sys

from card.formatter import format_card
from config import MAX_CLAIMS, UsageTracker, get_client, serpapi_key
from court.defender import argue_true
from court.judge import (
    find_claim_review_verdict,
    judge_verdict,
    no_evidence_verdict,
)
from court.models import ImageContext
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
from media.segment import segment_subjects
from media.vision import describe_image, read_image


def run_forward(
    text: str, *, client=None, http_client=None, factcheck=True
) -> tuple[str, UsageTracker]:
    """Run the pipeline on forward text. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    intake = extract_intake(text, client=client, tracker=tracker)
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


def run_image_forward(
    image_path: str, caption: str | None = None, *, client=None,
    http_client=None, provider=None, factcheck=True,
    ground_truth: list[str] | None = None,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on an image forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    image_bytes, mime = read_image(image_path)
    vision = describe_image(image_bytes, mime=mime, client=client, tracker=tracker)
    caption_claims: list[str] = []
    language = vision.language
    if caption:
        intake = extract_intake(caption, client=client, tracker=tracker)
        caption_claims = intake.claims
        language = intake.language
    presented_as_recent = is_presented_as_recent(
        [vision.visible_text, caption or ""]
    )
    claims = (vision.claims_from_image + caption_claims)[:MAX_CLAIMS]
    if not claims:
        return "No checkable claims found in this forward.", tracker
    reverse_items, reverse_evidence = [], False
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
            matches = search_images(
                image_bytes, [crop.data for crop in crops],
                provider=active_provider,
            )
            reverse_items = matches_to_evidence(
                matches, claims[0], http_client=http_client
            )
            reverse_evidence = any(match.exact for match in matches)
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
    review_verdict = find_claim_review_verdict(claims, evidence, language)
    if review_verdict is not None:
        tracker.note(
            f"ClaimReview verdict from {review_verdict.fact_checked_by}: "
            f"{review_verdict.verdict}."
        )
        return format_card(review_verdict), tracker
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
    )
    return format_card(verdict), tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fact-check a WhatsApp forward with a debating court."
    )
    parser.add_argument("text", nargs="?", help="Forward text (else read stdin).")
    parser.add_argument("--image", default=None, help="Image forward file path.")
    parser.add_argument("--caption", default=None, help="Caption for --image.")
    parser.add_argument("--no-factcheck", action="store_true",
                        help="Skip the existing-fact-check lookup stage.")
    args = parser.parse_args(argv)
    if args.image and args.text:
        print("Error: pass either forward text or --image, not both.",
              file=sys.stderr)
        return 2
    if args.caption and not args.image:
        print("Error: --caption needs --image.", file=sys.stderr)
        return 2
    use_factcheck = not args.no_factcheck
    try:
        if args.image:
            card, tracker = run_image_forward(
                args.image, args.caption, factcheck=use_factcheck
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
