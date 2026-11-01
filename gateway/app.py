"""FastAPI WhatsApp webhook (Twilio form payloads)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response

from gateway.security import valid_signature, valid_vonage_jwt
from gateway.senders import Sender, TwilioSender, VonageSender
from gateway.worker import handle_message

load_dotenv()

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "")
PROVIDER = os.environ.get("WHATSAPP_PROVIDER", "twilio").lower()
VONAGE_API_KEY = os.environ.get("VONAGE_API_KEY", "")
VONAGE_API_SECRET = os.environ.get("VONAGE_API_SECRET", "")
VONAGE_WHATSAPP_NUMBER = os.environ.get("VONAGE_WHATSAPP_NUMBER", "")
VONAGE_SIGNATURE_SECRET = os.environ.get("VONAGE_SIGNATURE_SECRET", "")
VONAGE_WEBHOOK_TOKEN = os.environ.get("VONAGE_WEBHOOK_TOKEN", "")

if not PUBLIC_BASE_URL:
    raise RuntimeError("PUBLIC_BASE_URL is not set; cannot verify webhooks.")
if not (ACCOUNT_SID and AUTH_TOKEN and WHATSAPP_FROM):
    raise RuntimeError("Twilio credentials are not set in .env.")

app = FastAPI()
seen_sids: set[str] = set()


def _sender() -> Sender:
    if PROVIDER == "vonage":
        if not (VONAGE_API_KEY and VONAGE_API_SECRET
                and VONAGE_WHATSAPP_NUMBER):
            raise RuntimeError("Vonage credentials are not set in .env.")
        return VonageSender(
            VONAGE_API_KEY, VONAGE_API_SECRET, VONAGE_WHATSAPP_NUMBER
        )
    return TwilioSender(ACCOUNT_SID, AUTH_TOKEN, WHATSAPP_FROM)


def _signed_url(request: Request) -> str:
    url = PUBLIC_BASE_URL + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    return url


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/webhook")
async def webhook(
    request: Request, tasks: BackgroundTasks
) -> Response:
    form = await request.form()
    params = {key: str(value) for key, value in form.multi_items()}
    signature = request.headers.get("x-twilio-signature", "")
    if not valid_signature(AUTH_TOKEN, _signed_url(request), params,
                           signature):
        return Response(status_code=403)
    sid = params.get("MessageSid", "")
    if not sid or sid in seen_sids:
        return Response(status_code=200)
    seen_sids.add(sid)
    tasks.add_task(
        handle_message,
        _sender(),
        to=params.get("From", ""),
        body=params.get("Body", ""),
        media_url=params.get("MediaUrl0") or None,
        media_type=params.get("MediaContentType0", ""),
        media_auth=(ACCOUNT_SID, AUTH_TOKEN),
    )
    return Response(status_code=200)


def _vonage_authorized(request: Request, raw_body: bytes) -> bool:
    if VONAGE_SIGNATURE_SECRET:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        return valid_vonage_jwt(VONAGE_SIGNATURE_SECRET, token, raw_body)
    if VONAGE_WEBHOOK_TOKEN:
        return request.query_params.get("t", "") == VONAGE_WEBHOOK_TOKEN
    return False


def _vonage_parts(payload: dict) -> tuple[str, str, str | None, str]:
    """(sender, text, media_url, media_type) from an inbound payload."""
    kind = payload.get("message_type", "")
    sender = (payload.get("from") or {})
    sender = sender.get("number", "") if isinstance(sender, dict) else sender
    if kind == "text":
        return sender, (payload.get("text") or ""), None, ""
    if kind in ("image", "video"):
        media = payload.get(kind) or {}
        return (
            sender,
            media.get("caption", ""),
            media.get("url"),
            f"{kind}/mp4" if kind == "video" else "image/jpeg",
        )
    return sender, "", None, ""


@app.post("/webhook/vonage")
async def webhook_vonage(
    request: Request, tasks: BackgroundTasks
) -> Response:
    raw_body = await request.body()
    if not _vonage_authorized(request, raw_body):
        return Response(status_code=403)
    try:
        payload = await request.json()
    except ValueError:
        return Response(status_code=400)
    uuid = payload.get("message_uuid", "")
    if not uuid or uuid in seen_sids:
        return Response(status_code=200)
    seen_sids.add(uuid)
    sender, text, media_url, media_type = _vonage_parts(payload)
    tasks.add_task(
        handle_message,
        _sender(),
        to=sender,
        body=text,
        media_url=media_url,
        media_type=media_type,
    )
    return Response(status_code=200)


@app.post("/webhook/vonage/status")
async def webhook_vonage_status(request: Request) -> Response:
    print(f"vonage status: {(await request.body()).decode()[:300]}")
    return Response(status_code=200)
