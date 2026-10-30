"""Defender: argues the forward's claims are TRUE from shared evidence."""
from __future__ import annotations

from config import agent_model

from .models import Argument, evidence_brief

INSTRUCTIONS = (
    "You are the defender in a fact-check court. Argue that the claims are "
    "TRUE, using ONLY the evidence list below. Every point must cite one "
    "source_url copied EXACTLY from that list; a point with any other URL -- "
    "or none -- is dropped. Never invent URLs, dates, or quotes. If the "
    "evidence does not support the claims being true, return few or no points. "
    "Entries marked visually similar are context only; cite them solely as "
    "similarity, never as proof of origin."
)


def argue_true(claims: list[str], evidence, *, client, tracker=None) -> Argument:
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=f"Claims:\n{claims}\n\nEvidence:\n{evidence_brief(evidence)}",
        text_format=Argument,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="defender")
    return response.output_parsed
