"""Checkable-claim and language data shapes for intake."""
from pydantic import BaseModel, Field


class IntakeResult(BaseModel):
    language: str = Field(
        description="ISO 639-1 language code of the forward, e.g. 'en', 'hi'."
    )
    claims: list[str] = Field(
        description="Checkable factual claims, most important first."
    )
    event_claims: list[str] = Field(
        default_factory=list,
        description="Claims asserting real-world facts "
        "(e.g. 'floods hit Patna today').",
    )
    media_claims: list[str] = Field(
        default_factory=list,
        description="Claims asserting what attached media shows "
        "(e.g. 'this video shows the Patna floods').",
    )
    presented_as_recent: bool = Field(
        description="True when the forward presents events as breaking, "
        "just-happened, or otherwise current."
    )
