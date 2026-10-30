"""Media-forward data shapes."""
from typing import Literal

from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    scene_description: str
    visible_text: str = Field(
        description="Text visible in the image, verbatim; empty string if none."
    )
    people_places_objects: list[str] = Field(
        description="Main depicted subjects as short noun phrases."
    )
    claims_from_image: list[str] = Field(
        description="Checkable claims the image itself makes (what/where/when)."
    )
    language: str = Field(
        description="ISO 639-1 language of visible text, or 'en' if none."
    )
    image_role: Literal[
        "viral_claim", "original_source", "fact_check_marked", "other"
    ] = Field(
        description="viral_claim: the image as shared in the viral forward. "
        "original_source: the undoctored original the claim misuses. "
        "fact_check_marked: carries fact-check stamps, comparisons, or "
        "outlet logos/watermarks. other: anything else."
    )


class VideoResult(BaseModel):
    scene_summary: str
    visible_text: str = Field(
        description="Text visible across the video frames, verbatim; empty if none."
    )
    spoken_claims_hint: str = Field(
        description="What the soundtrack seems to claim, briefly; empty if unclear."
    )
    people_places_objects: list[str] = Field(
        description="Main depicted subjects as short noun phrases."
    )
    claims_from_video: list[str] = Field(
        description="Checkable claims the video itself makes (what/where/when)."
    )
    language: str = Field(
        description="ISO 639-1 language of visible text, or 'en' if none."
    )
