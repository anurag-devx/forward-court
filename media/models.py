"""Image-forward data shapes."""
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
    fact_check_marked: bool = Field(
        description="True when the image carries fact-check markings: "
        "FAKE/FALSE/MISLEADING stamps, side-by-side comparison layouts, "
        "or fact-check outlet logos, watermarks, or tickers."
    )
