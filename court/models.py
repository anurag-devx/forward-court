"""Court data shapes: arguments and verdicts."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from evidence.urls import tier_of


class ArgPoint(BaseModel):
    text: str
    source_url: str = Field(description="Must be copied exactly from the evidence list.")


class Argument(BaseModel):
    points: list[ArgPoint]


class Verdict(BaseModel):
    verdict: Literal["True", "False", "Misleading", "Unverified"]
    reason: str
    key_source: str
    language: str
    confidence: Literal["low", "medium", "high"]
    claim_timeframe: str | None = Field(
        default=None,
        description="ISO date the forward implies the event happened, else null.",
    )
    fact_checked_by: str | None = Field(
        default=None,
        description="Outlet host when a ClaimReview rating decided the verdict.",
    )


class ImageContext(BaseModel):
    origin_only: bool = Field(
        description="All claims come from the image; the caption added none."
    )
    reverse_evidence: bool = Field(
        description="Reverse search produced exact-match origin evidence."
    )


def evidence_brief(evidence) -> str:
    """Render the shared evidence list for agent prompts."""
    entries = []
    for item in evidence:
        entry = {
            "source_url": item.source_url,
            "date": item.date,
            "snippet": item.snippet,
            "tier": tier_of(item.source_url),
        }
        if item.visual_only:
            entry["note"] = (
                "visually similar match: context only, "
                "not proof of origin or date"
            )
        elif item.fact_check:
            entry["note"] = "existing fact-check from a trusted outlet"
        entries.append(entry)
    return json.dumps(entries, ensure_ascii=False, indent=2)
