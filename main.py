"""CLI entry: python main.py "<forward text>"

Runs the full pipeline: intake -> evidence -> debate -> judge -> card.
Prints the verdict card plus per-run token usage to stdout.
"""
from __future__ import annotations

import argparse
import sys

from card.formatter import format_card
from config import UsageTracker, get_client
from court.defender import argue_true
from court.judge import judge_verdict
from court.prosecutor import argue_false
from evidence.grounder import GroundingError, gather_evidence
from intake.extractor import extract_intake


def run_forward(text: str, *, client=None) -> tuple[str, UsageTracker]:
    """Run the pipeline on forward text. Returns (card, usage)."""
    if client is None:
        client = get_client()
    tracker = UsageTracker()
    intake = extract_intake(text, client=client, tracker=tracker)
    if not intake.claims:
        return "No checkable claims found in this forward.", tracker
    evidence = gather_evidence(intake.claims, client=client, tracker=tracker)
    prosecution = argue_false(intake.claims, evidence, client=client, tracker=tracker)
    defense = argue_true(intake.claims, evidence, client=client, tracker=tracker)
    verdict = judge_verdict(
        intake.claims,
        evidence,
        prosecution,
        defense,
        intake.language,
        client=client,
        tracker=tracker,
    )
    return format_card(verdict), tracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fact-check a WhatsApp forward with a debating court."
    )
    parser.add_argument("text", nargs="?", help="Forward text (else read stdin).")
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        print("Error: provide forward text as an argument or via stdin.",
              file=sys.stderr)
        return 2
    try:
        card, tracker = run_forward(text)
    except GroundingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(card)
    print()
    print(tracker.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
