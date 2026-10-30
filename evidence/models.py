"""Grounded-evidence data shapes."""
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    claim: str
    source_url: str
    date: str | None = Field(
        default=None, description="ISO YYYY-MM-DD publication date when available."
    )
    snippet: str
