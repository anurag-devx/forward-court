"""Central configuration for Forward Court.

Model names live ONLY in this file. Every other module imports them from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.meta.ai/v1"
API_KEY_ENV = "MODEL_API_KEY"
SERPAPI_KEY_ENV = "SERPAPI_KEY"

AGENT_MODEL = "muse-spark-1.3"
CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"
SEGMENTATION_MODEL = "sam-3.1"
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"

CONTRIBUTOR_ENV = "FC_CONTRIBUTOR"
CONTRIBUTOR_OFF = frozenset({"0", "false", "no", "off"})

MAX_CLAIMS = 3


def agent_model() -> str:
    """Model for agents/judge/grounding. Contributor mode is default-on;
    set FC_CONTRIBUTOR=0 (or false/no/off) for the standard model."""
    if os.environ.get(CONTRIBUTOR_ENV, "1").strip().lower() in CONTRIBUTOR_OFF:
        return AGENT_MODEL
    return CONTRIBUTOR_MODEL


def serpapi_key() -> str | None:
    """SerpAPI key for reverse image search, or None when not configured."""
    return os.environ.get(SERPAPI_KEY_ENV) or None


def get_client() -> OpenAI:
    """Build an OpenAI SDK client pointed at the Meta Model API."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Put it in .env and retry. "
            "(Never print or commit the key itself.)"
        )
    return OpenAI(base_url=BASE_URL, api_key=api_key)


@dataclass
class UsageTracker:
    """Accumulates token usage across the LLM calls of one run."""

    input_tokens: int = 0
    output_tokens: int = 0
    stages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def add(self, usage, stage: str | None = None) -> None:
        if usage is None:
            return
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        if stage is not None:
            stage_in, stage_out = self.stages.get(stage, (0, 0))
            self.stages[stage] = (stage_in + in_tokens, stage_out + out_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (
            f"Tokens: {self.input_tokens} in / "
            f"{self.output_tokens} out / {self.total_tokens} total"
        )

    def breakdown(self) -> list[str]:
        """Per-stage token lines, in first-seen order."""
        return [
            f"  {stage}: {in_tokens} in / {out_tokens} out / "
            f"{in_tokens + out_tokens} total"
            for stage, (in_tokens, out_tokens) in self.stages.items()
        ]
