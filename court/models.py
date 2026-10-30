"""Court data shapes: arguments and verdicts."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


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


def evidence_brief(evidence) -> str:
    """Render the shared evidence list for agent prompts."""
    return json.dumps(
        [
            {
                "source_url": item.source_url,
                "date": item.date,
                "snippet": item.snippet,
            }
            for item in evidence
        ],
        ensure_ascii=False,
        indent=2,
    )
