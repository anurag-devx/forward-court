"""Central configuration for Forward Court.

Model names live ONLY in this file. Every other module imports them from here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.meta.ai/v1"
API_KEY_ENV = "MODEL_API_KEY"

AGENT_MODEL = "muse-spark-1.3"
SEGMENTATION_MODEL = "sam-3.1"
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"

MAX_CLAIMS = 3


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

    def add(self, usage) -> None:
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (
            f"Tokens: {self.input_tokens} in / "
            f"{self.output_tokens} out / {self.total_tokens} total"
        )
