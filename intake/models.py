"""Checkable-claim and language data shapes for intake."""
from pydantic import BaseModel, Field


class IntakeResult(BaseModel):
    language: str = Field(
        description="ISO 639-1 language code of the forward, e.g. 'en', 'hi'."
    )
    claims: list[str] = Field(
        description="Checkable factual claims, most important first."
    )
