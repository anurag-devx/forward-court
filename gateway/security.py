"""Webhook request validation: Twilio HMAC-SHA1, Vonage signed JWT (no SDK)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

VONAGE_JWT_MAX_AGE_S = 24 * 60 * 60


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


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def valid_vonage_jwt(secret: str, token: str, raw_body: bytes) -> bool:
    """True when the Vonage signed-webhook JWT is authentic and fresh.

    Verifies the HMAC-SHA256 signature, the payload_hash claim against the
    raw body, and the iat freshness window.
    """
    if not secret or not token:
        return False
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            return False
        expected = hmac.new(
            secret.encode("utf-8"),
            f"{header_b64}.{payload_b64}".encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
            return False
        claims = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return False
    digest = hashlib.sha256(raw_body).hexdigest()
    if not hmac.compare_digest(str(claims.get("payload_hash", "")), digest):
        return False
    try:
        age = time.time() - float(claims["iat"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= age <= VONAGE_JWT_MAX_AGE_S
