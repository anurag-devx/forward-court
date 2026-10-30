"""Extract checkable claims and detect language via structured output."""
from __future__ import annotations

from config import MAX_CLAIMS, agent_model

from .models import IntakeResult

INSTRUCTIONS = (
    "You read viral WhatsApp forwards. Detect the language of the user's message "
    "and extract its checkable factual claims. "
    "Return an ISO 639-1 language code (e.g. 'en' for English, 'hi' for Hindi). "
    "List each distinct checkable factual claim once, most important first. "
    "Split the claims: media_claims assert what an attached photo or video "
    "shows (e.g. 'this video shows the Patna floods'); event_claims assert "
    "real-world facts (e.g. 'floods hit Patna today'). Claims that do both "
    "go in both lists. "
    "Skip greetings, opinions, prayers, jokes, and calls to forward. "
    "If there is no checkable factual claim, return an empty list. "
    "Set presented_as_recent true when the forward presents events as "
    "breaking, just-happened, or otherwise current "
    "(e.g. 'breaking', 'just now', 'today', 'this morning'); false otherwise."
)


def extract_intake(text: str, *, client, tracker=None) -> IntakeResult:
    """Run intake on forward text. Never raises for empty claims (returns [])."""
    if tracker is not None:
        tracker.mark("intake")
    response = client.responses.parse(
        model=agent_model(),
        instructions=INSTRUCTIONS,
        input=text,
        text_format=IntakeResult,
    )
    if tracker is not None:
        tracker.add(response.usage, stage="intake")
    result = response.output_parsed
    kept = list(result.claims)[:MAX_CLAIMS]
    return IntakeResult(
        language=result.language,
        claims=kept,
        event_claims=[claim for claim in result.event_claims if claim in kept],
        media_claims=[claim for claim in result.media_claims if claim in kept],
        presented_as_recent=result.presented_as_recent,
    )
