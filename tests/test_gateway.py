import os

# Forced (not setdefault): config.load_dotenv() may lodge empty .env values
# before this module imports.
os.environ["PUBLIC_BASE_URL"] = "https://test.example"
os.environ["TWILIO_ACCOUNT_SID"] = "AC" + "0" * 32
os.environ["TWILIO_AUTH_TOKEN"] = "test-token"
os.environ["TWILIO_WHATSAPP_FROM"] = "whatsapp:+10000000000"

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway.app as app_module
from gateway.app import app
from gateway.security import compute_signature, valid_signature
from gateway.senders import (
    MetaCloudSender,
    SenderError,
    TwilioSender,
    split_message,
)
from gateway.worker import ack_language

AUTH = "test-token"
BASE = "https://test.example"
FROM = "whatsapp:+12223334444"


class _RecordingSender:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send(self, to, body):
        self.sent.append((to, body))


def _form(**extra):
    params = {
        "MessageSid": "SM123",
        "From": FROM,
        "Body": "hello forward",
        "NumMedia": "0",
    }
    params.update(extra)
    return params


def _signed_post(client, params):
    signature = compute_signature(AUTH, BASE + "/webhook", params)
    return client.post(
        "/webhook", data=params, headers={"X-Twilio-Signature": signature}
    )


@pytest.fixture
def sender(monkeypatch):
    recording = _RecordingSender()
    monkeypatch.setattr(app_module, "_sender", lambda: recording)
    return recording


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_bad_signature_rejected(client):
    response = client.post(
        "/webhook", data=_form(), headers={"X-Twilio-Signature": "bogus"}
    )
    assert response.status_code == 403


def test_valid_signature_accepted():
    params = _form()
    signature = compute_signature(AUTH, BASE + "/webhook", params)
    assert valid_signature(AUTH, BASE + "/webhook", params, signature)
    assert not valid_signature(AUTH, BASE + "/webhook", params, "bogus")


def test_text_flow_sends_ack_then_card(client, sender, monkeypatch):
    import gateway.worker as worker_module

    seen_args: list[list[str]] = []

    def fake_run(args, card_path):
        seen_args.append(args)
        card_path.write_text("CARD")
        return "CARD"

    monkeypatch.setattr(worker_module, "run_pipeline", fake_run)
    response = _signed_post(client, _form(MessageSid="SM-text"))
    assert response.status_code == 200
    assert [body for _, body in sender.sent] == [
        "Checking this forward…", "CARD"]
    assert seen_args == [["hello forward"]]


def test_dedupe_ignores_replay(client, sender, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "run_pipeline", lambda args, path: "CARD")
    params = _form(MessageSid="SM-dupe")
    assert _signed_post(client, params).status_code == 200
    assert _signed_post(client, params).status_code == 200
    assert len(sender.sent) == 2  # one ack + one card, not four


def test_image_flow_downloads_and_passes_caption(
    client, sender, monkeypatch
):
    import gateway.worker as worker_module

    seen: dict = {}

    def fake_run(args, card_path):
        seen["args"] = args
        return "CARD"

    monkeypatch.setattr(worker_module, "run_pipeline", fake_run)
    monkeypatch.setattr(
        worker_module, "download_media", lambda url, auth: b"img-bytes")
    params = _form(
        MessageSid="SM-img", NumMedia="1",
        MediaUrl0="https://api.twilio.com/media/1",
        MediaContentType0="image/jpeg",
    )
    assert _signed_post(client, params).status_code == 200
    assert seen["args"][0] == "--image"
    assert seen["args"][2:] == ["--caption", "hello forward"]
    assert sender.sent[-1][1] == "CARD"


def test_unsupported_media_gets_couldnt_verify(client, sender, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "download_media", lambda url, auth: b"data")
    params = _form(
        MessageSid="SM-pdf", NumMedia="1",
        MediaUrl0="https://api.twilio.com/media/2",
        MediaContentType0="application/pdf",
    )
    assert _signed_post(client, params).status_code == 200
    assert "Couldn't verify" in sender.sent[-1][1]


def test_pipeline_failure_gets_couldnt_verify(client, sender, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "run_pipeline", lambda args, path: None)
    params = _form(MessageSid="SM-fail")
    assert _signed_post(client, params).status_code == 200
    assert "Couldn't verify" in sender.sent[-1][1]


def test_hindi_ack_for_devanagari_body(client, sender, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "run_pipeline", lambda args, path: "CARD")
    params = _form(MessageSid="SM-hi", Body="यह सच है क्या")
    assert _signed_post(client, params).status_code == 200
    assert ack_language("यह सच है क्या") == "hi"
    assert ack_language("is this true") == "en"
    assert sender.sent[0][1] == "यह फ़ॉरवर्ड जाँचा जा रहा है…"


def test_sender_splits_long_messages():
    bodies: list[str] = []

    def handler(request):
        bodies.append(request.read().decode())
        return httpx.Response(200, json={"sid": "SM1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = TwilioSender("AC" + "0" * 32, "tok", "whatsapp:+1000",
                          client=client)
    sender.send("whatsapp:+2000", "word " * 400)
    assert len(bodies) == 2
    assert len(split_message("short")) == 1


def test_sender_error_raises():
    def handler(request):
        return httpx.Response(500, text="oops")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = TwilioSender("AC" + "0" * 32, "tok", "whatsapp:+1000",
                          client=client)
    with pytest.raises(SenderError):
        sender.send("whatsapp:+2000", "hi")


def test_meta_sender_is_stub():
    with pytest.raises(NotImplementedError):
        MetaCloudSender().send("whatsapp:+2000", "hi")
