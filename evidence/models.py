"""Grounded-evidence data shapes."""
from pydantic import BaseModel, Field


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
