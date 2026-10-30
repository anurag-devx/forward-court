"""Extract checkable claims and detect language via structured output."""
from __future__ import annotations

from config import MAX_CLAIMS, agent_model

from .models import IntakeResult

INSTRUCTIONS = (
    "You read viral WhatsApp forwards. Detect the language of the user's message "
    "and extract its checkable factual claims. "
    "Return an ISO 639-1 language code (e.g. 'en' for English, 'hi' for Hindi). "
    "List each distinct checkable factual claim once, most important first. "
    "Skip greetings, opinions, prayers, jokes, and calls to forward. "
    "If there is no checkable factual claim, return an empty list. "
    "Set presented_as_recent true when the forward presents events as "
    "breaking, just-happened, or otherwise current "
    "(e.g. 'breaking', 'just now', 'today', 'this morning'); false otherwise."
)


def extract_intake(text: str, *, client, tracker=None) -> IntakeResult:
    """Run intake on forward text. Never raises for empty claims (returns [])."""
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=text,
        text_format=IntakeResult,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="intake")
    result = response.output_parsed
    return IntakeResult(
        language=result.language,
        claims=list(result.claims)[:MAX_CLAIMS],
        presented_as_recent=result.presented_as_recent,
    )
