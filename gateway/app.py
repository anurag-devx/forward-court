"""FastAPI WhatsApp webhook (Twilio form payloads)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response

from gateway.security import valid_signature
from gateway.senders import Sender, TwilioSender
from gateway.worker import handle_message

load_dotenv()

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "")

if not PUBLIC_BASE_URL:
    raise RuntimeError("PUBLIC_BASE_URL is not set; cannot verify webhooks.")
if not (ACCOUNT_SID and AUTH_TOKEN and WHATSAPP_FROM):
    raise RuntimeError("Twilio credentials are not set in .env.")

app = FastAPI()
seen_sids: set[str] = set()


def _sender() -> Sender:
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
