"""Relevance gate: one structured call scores each evidence item 0/1/2.

Items scoring 0 (off-topic for their claim) are dropped before the debate;
claims left with no items have no evidence. Scores are stored on the kept
items for the ClaimReview lookup+relevance match rule.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from config import agent_model

SNIPPET_CHARS = 300

INSTRUCTIONS = (
    "Score how relevant each evidence item is to its claim. "
    "2 = directly about the claim (supports, refutes, or fact-checks it). "
    "1 = same topic but not the claim itself. "
    "0 = off-topic or unrelated to the claim. "
    "Score every item by its index."
)


class RelevanceScore(BaseModel):
    index: int
    score: int = Field(ge=0, le=2)


class RelevanceScores(BaseModel):
    scores: list[RelevanceScore]


def filter_evidence(
    claims: list[str], evidence, *, client, tracker=None
) -> tuple[list, list[str]]:
    """Score items and drop 0s. Returns (kept, claims_with_no_evidence)."""
    if tracker is not None:
        tracker.mark("relevance")
    if not evidence:
        return [], list(claims)
    payload = [
        {
            "index": index,
            "claim": item.claim,
            "source_url": item.source_url,
            "snippet": item.snippet[:SNIPPET_CHARS],
        }
        for index, item in enumerate(evidence)
    ]
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
        text_format=RelevanceScores,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="relevance")
    by_index = {
        score.index: max(0, min(2, score.score))
        for score in response.output_parsed.scores
    }
    kept = []
    for index, item in enumerate(evidence):
        score = by_index.get(index, 1)
        if score == 0:
            continue
        kept.append(item.model_copy(update={"relevance": score}))
    kept_claims = {item.claim for item in kept}
    dropped_claims = [claim for claim in claims if claim not in kept_claims]
    if tracker is not None:
        tracker.note(
            f"relevance filter dropped {len(evidence) - len(kept)}/"
            f"{len(evidence)} items."
        )
        for claim in dropped_claims:
            tracker.note(f"no evidence kept for claim: {claim}")
    return kept, dropped_claims
