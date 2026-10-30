"""Grounded-evidence data shapes."""
from typing import Literal

from pydantic import BaseModel, Field


class ClaimReview(BaseModel):
    """A parsed schema.org ClaimReview rating from the evidence page."""

    rating_raw: str = Field(description="Raw alternateName/ratingValue text.")
    verdict: Literal["True", "False", "Misleading", "Unverified"] = Field(
        description="Rating mapped to a Forward Court verdict."
    )
    claim_text: str = Field(
        description="claimReviewed/itemReviewed text the rating applies to."
    )


class EvidenceItem(BaseModel):
    claim: str
    source_url: str
    date: str | None = Field(
        default=None, description="ISO YYYY-MM-DD publication date when available."
    )
    snippet: str
    visual_only: bool = Field(
        default=False,
        description="Visually-similar reverse match: context for debaters, "
        "never origin evidence and never triggers the recycled override.",
    )
    fact_check: bool = Field(
        default=False,
        description="Existing fact-check from a trusted outlet: weighs "
        "heavily with the judge when its rating is clear.",
    )
    relevance: int | None = Field(
        default=None,
        description="Relevance score 0/1/2 from the relevance filter; "
        "None when the filter has not run.",
    )
    claim_review: ClaimReview | None = Field(
        default=None,
        description="Parsed ClaimReview rating from the page, when present.",
    )
    exact_match: bool = Field(
        default=False,
        description="Exact reverse-search hit: media-provenance evidence.",
    )
