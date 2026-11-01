"""Pluggable WhatsApp senders (Twilio live, Meta Cloud stub)."""
from __future__ import annotations

from typing import Protocol

import httpx

MAX_PART_CHARS = 1600


class SenderError(RuntimeError):
    """Outbound message delivery failed."""


class Sender(Protocol):
    def send(self, to: str, body: str) -> None:
        """Deliver body to the recipient address."""
        ...


def split_message(body: str, limit: int = MAX_PART_CHARS) -> list[str]:
    """Greedy word-wrap a message into parts of at most limit chars."""
    if len(body) <= limit:
        return [body]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in body.split(" "):
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > limit:
            parts.append(" ".join(current))
            current, current_len = [], 0
            extra = len(word)
        current.append(word)
        current_len += extra
    if current:
        parts.append(" ".join(current))
    return parts


class TwilioSender:
    """Sends via the Twilio Messages API with basic auth."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_address: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{account_sid}/Messages.json"
        )
        self._auth = (account_sid, auth_token)
        self._from = from_address
        self._client = client

    def send(self, to: str, body: str) -> None:
        if self._client is None:
            with httpx.Client() as client:
                self._send_all(to, body, client)
        else:
            self._send_all(to, body, self._client)

    def _send_all(self, to: str, body: str, client: httpx.Client) -> None:
        for part in split_message(body):
            try:
                response = client.post(
                    self._url,
                    data={"From": self._from, "To": to, "Body": part},
                    auth=self._auth,
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise SenderError(f"Twilio send failed: {exc}") from exc


class MetaCloudSender:
    """WhatsApp Cloud API sender (stub for later)."""

    def send(self, to: str, body: str) -> None:
        raise NotImplementedError("Meta Cloud sender is not implemented yet.")
