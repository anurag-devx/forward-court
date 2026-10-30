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
from court.judge import judge_verdict
from court.models import ImageContext
from court.prosecutor import argue_false
from evidence.grounder import GroundingError, gather_evidence
from evidence.reverse_image import (
    ReverseSearchError,
    SerpApiLensProvider,
    matches_to_evidence,
    search_images,
)
from intake.extractor import extract_intake
from media.segment import segment_subjects
from media.vision import describe_image, read_image


def run_forward(text: str, *, client=None, http_client=None) -> tuple[str, UsageTracker]:
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
    )
    return format_card(verdict), tracker


def run_image_forward(
    image_path: str, caption: str | None = None, *, client=None,
    http_client=None, provider=None,
) -> tuple[str, UsageTracker]:
    """Run the pipeline on an image forward. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    image_bytes, mime = read_image(image_path)
    vision = describe_image(image_bytes, mime=mime, client=client, tracker=tracker)
    caption_claims: list[str] = []
    language = vision.language
    presented_as_recent = False
    if caption:
        intake = extract_intake(caption, client=client, tracker=tracker)
        caption_claims = intake.claims
        language = intake.language
        presented_as_recent = intake.presented_as_recent
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
        except ReverseSearchError as exc:
            tracker.note(f"reverse search unavailable: {exc}")
    evidence = reverse_items + gather_evidence(
        claims, client=client, tracker=tracker, http_client=http_client
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
    )
    return format_card(verdict), tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fact-check a WhatsApp forward with a debating court."
    )
    parser.add_argument("text", nargs="?", help="Forward text (else read stdin).")
    parser.add_argument("--image", default=None, help="Image forward file path.")
    parser.add_argument("--caption", default=None, help="Caption for --image.")
    args = parser.parse_args(argv)
    if args.image and args.text:
        print("Error: pass either forward text or --image, not both.",
              file=sys.stderr)
        return 2
    if args.caption and not args.image:
        print("Error: --caption needs --image.", file=sys.stderr)
        return 2
    try:
        if args.image:
            card, tracker = run_image_forward(args.image, args.caption)
        else:
            text = args.text if args.text is not None else sys.stdin.read()
            if not text.strip():
                print("Error: provide forward text as an argument or via stdin.",
                      file=sys.stderr)
                return 2
            card, tracker = run_forward(text)
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
