"""Twilio webhook request validation (HMAC-SHA1, no SDK)."""
from __future__ import annotations

import base64
import hashlib
import hmac


def compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """X-Twilio-Signature for a URL + POST params (Twilio's algorithm)."""
    data = url + "".join(
        key + params[key] for key in sorted(params)
    )
    digest = hmac.new(
        auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def valid_signature(
    auth_token: str, url: str, params: dict[str, str], signature: str
) -> bool:
    """True when the header matches the expected signature."""
    if not signature:
        return False
    expected = compute_signature(auth_token, url, params)
    return hmac.compare_digest(expected, signature)
