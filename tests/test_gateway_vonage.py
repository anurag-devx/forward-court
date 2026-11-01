import base64
import hashlib
import hmac
import json
import os
import time

os.environ["PUBLIC_BASE_URL"] = "https://test.example"
os.environ["TWILIO_ACCOUNT_SID"] = "AC" + "0" * 32
os.environ["TWILIO_AUTH_TOKEN"] = "test-token"
os.environ["TWILIO_WHATSAPP_FROM"] = "whatsapp:+10000000000"

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway.app as app_module
from gateway.app import app
from gateway.security import valid_vonage_jwt
from gateway.senders import SenderError, VonageSender

SECRET = "vonage-signature-secret"
FROM = "919999999999"


class _RecordingSender:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    def send(self, to, body):
        self.sent.append((to, body))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _jwt(secret, raw_body: bytes, *, age_s=10, payload_hash=None):
    header = _b64url(json.dumps(
        {"alg": "HS256", "typ": "JWT"}).encode())
    claims = {
        "payload_hash": payload_hash
        if payload_hash is not None
        else hashlib.sha256(raw_body).hexdigest(),
        "iat": time.time() - age_s,
    }
    payload = _b64url(json.dumps(claims).encode())
    signature = _b64url(hmac.new(
        secret.encode(), f"{header}.{payload}".encode(),
        hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _text_payload(uuid="uuid-1", text="hello forward"):
    return {
        "message_uuid": uuid, "from": {"number": FROM},
        "to": {"number": "918888888888"}, "message_type": "text",
        "text": text, "timestamp": "2026-11-01T00:00:00.000Z",
    }


@pytest.fixture
def sender(monkeypatch):
    recording = _RecordingSender()
    monkeypatch.setattr(app_module, "_sender", lambda: recording)
    return recording


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def use_jwt(monkeypatch):
    monkeypatch.setattr(app_module, "VONAGE_SIGNATURE_SECRET", SECRET)
    monkeypatch.setattr(app_module, "VONAGE_WEBHOOK_TOKEN", "")


@pytest.fixture
def use_token(monkeypatch):
    monkeypatch.setattr(app_module, "VONAGE_SIGNATURE_SECRET", "")
    monkeypatch.setattr(app_module, "VONAGE_WEBHOOK_TOKEN", "hook-token")


def _post_jwt(client, payload, secret=SECRET, **jwt_kwargs):
    raw = json.dumps(payload).encode()
    token = _jwt(secret, raw, **jwt_kwargs)
    return client.post(
        "/webhook/vonage", content=raw,
        headers={"Authorization": f"Bearer {token}"},
    )


def test_jwt_accepts_valid_text_message(client, sender, use_jwt, monkeypatch):
    import gateway.worker as worker_module

    seen: list = []
    monkeypatch.setattr(
        worker_module, "run_pipeline",
        lambda args, path: seen.append(args) or "CARD")
    response = _post_jwt(client, _text_payload(uuid="uuid-text"))
    assert response.status_code == 200
    assert [body for _, body in sender.sent] == [
        "Checking this forward…", "CARD"]
    assert seen == [["hello forward"]]


def test_risky_file_attachment_replies_dangerous(client, sender, use_jwt):
    payload = _text_payload(uuid="uuid-file")
    payload["message_type"] = "file"
    payload["file"] = {"name": "update.apk", "caption": "install this"}
    response = _post_jwt(client, payload)
    assert response.status_code == 200
    assert len(sender.sent) == 2
    assert sender.sent[0][1] == "Checking this forward…"
    assert "DANGEROUS LINK" in sender.sent[1][1].splitlines()[0]


def test_safe_file_attachment_falls_through(client, sender, use_jwt, monkeypatch):
    import gateway.worker as worker_module

    seen: list = []
    monkeypatch.setattr(
        worker_module, "run_pipeline",
        lambda args, path: seen.append(args) or "CARD")
    payload = _text_payload(uuid="uuid-file-ok")
    payload["message_type"] = "file"
    payload["file"] = {"name": "notes.pdf"}
    response = _post_jwt(client, payload)
    assert response.status_code == 200
    assert "DANGEROUS" not in "".join(body for _, body in sender.sent)


def test_jwt_rejects_tampered_payload(client, sender, use_jwt):
    payload = _text_payload(uuid="uuid-tamper")
    raw = json.dumps(payload).encode()
    token = _jwt(SECRET, b"something else")
    response = client.post(
        "/webhook/vonage", content=raw,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert sender.sent == []


def test_jwt_rejects_stale_and_unsigned(client, sender, use_jwt):
    old = _post_jwt(client, _text_payload(uuid="u1"), age_s=100000)
    assert old.status_code == 403
    missing = client.post("/webhook/vonage", json=_text_payload(uuid="u2"))
    assert missing.status_code == 403
    assert sender.sent == []


def test_valid_vonage_jwt_unit():
    raw = b'{"a": 1}'
    assert valid_vonage_jwt(SECRET, _jwt(SECRET, raw), raw)
    assert not valid_vonage_jwt(SECRET, "not.a.jwt", raw)
    assert not valid_vonage_jwt("", _jwt(SECRET, raw), raw)


def test_token_gate_mode(client, sender, use_token, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "run_pipeline", lambda args, path: "CARD")
    payload = _text_payload(uuid="uuid-token")
    denied = client.post("/webhook/vonage", json=payload)
    assert denied.status_code == 403
    allowed = client.post("/webhook/vonage?t=hook-token", json=payload)
    assert allowed.status_code == 200
    assert len(sender.sent) == 2


def test_fail_closed_without_secrets(client, sender, monkeypatch):
    monkeypatch.setattr(app_module, "VONAGE_SIGNATURE_SECRET", "")
    monkeypatch.setattr(app_module, "VONAGE_WEBHOOK_TOKEN", "")
    response = client.post("/webhook/vonage", json=_text_payload())
    assert response.status_code == 403
    assert sender.sent == []


def test_image_flow_maps_url_and_caption(client, sender, use_jwt, monkeypatch):
    import gateway.worker as worker_module

    seen: dict = {}

    def fake_run(args, card_path):
        seen["args"] = args
        return "CARD"

    monkeypatch.setattr(worker_module, "run_pipeline", fake_run)
    monkeypatch.setattr(
        worker_module, "download_media", lambda url, auth=None: b"img")
    payload = {
        "message_uuid": "uuid-img", "from": {"number": FROM},
        "message_type": "image",
        "image": {"url": "https://api.nexmo.com/v3/media/x",
                  "caption": "look here"},
        "timestamp": "2026-11-01T00:00:00.000Z",
    }
    assert _post_jwt(client, payload).status_code == 200
    assert seen["args"][0] == "--image"
    assert seen["args"][2:] == ["--caption", "look here"]
    assert sender.sent[-1][1] == "CARD"


def test_dedupe_by_message_uuid(client, sender, use_jwt, monkeypatch):
    import gateway.worker as worker_module

    monkeypatch.setattr(
        worker_module, "run_pipeline", lambda args, path: "CARD")
    payload = _text_payload(uuid="uuid-dupe")
    assert _post_jwt(client, payload).status_code == 200
    assert _post_jwt(client, payload).status_code == 200
    assert len(sender.sent) == 2


def test_status_route_returns_200(client):
    response = client.post(
        "/webhook/vonage/status", json={"message_uuid": "x"})
    assert response.status_code == 200


def test_vonage_sender_posts_exact_json(monkeypatch):
    bodies: list[dict] = []
    headers: list[str] = []

    def handler(request):
        bodies.append(json.loads(request.read()))
        headers.append(request.headers["authorization"])
        return httpx.Response(202, json={"message_uuid": "m1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = VonageSender("key1", "secret1", "91111", client=client)
    sender.send("91222", "hi")
    assert bodies == [{
        "from": "91111", "to": "91222", "channel": "whatsapp",
        "message_type": "text", "text": "hi",
    }]
    assert headers[0].startswith("Basic ")
    assert str(client._transport) != ""


def test_vonage_sender_paces_split_parts(monkeypatch):
    calls = {"posts": 0}
    sleeps: list[float] = []

    def handler(request):
        calls["posts"] += 1
        return httpx.Response(202, json={})

    monkeypatch.setattr("gateway.senders.time.sleep", sleeps.append)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = VonageSender("k", "s", "91111", client=client)
    sender.send("91222", "word " * 400)
    assert calls["posts"] == 2
    assert sleeps == [1.0]


def test_vonage_sender_error_raises():
    def handler(request):
        return httpx.Response(429, text="slow down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sender = VonageSender("k", "s", "91111", client=client)
    with pytest.raises(SenderError):
        sender.send("91222", "hi")


def test_status_webhook_logs_uuid_and_status_only(client, capsys):
    response = client.post(
        "/webhook/vonage/status",
        json={
            "message_uuid": "uuid-status-1",
            "status": "delivered",
            "to": {"number": "919999999999"},
            "from": {"number": "918888888888"},
        },
    )
    assert response.status_code == 200
    out = capsys.readouterr().out
    assert "uuid-status-1" in out and "delivered" in out
    assert "919999999999" not in out
    assert "918888888888" not in out


def test_status_webhook_tolerates_bad_payload(client, capsys):
    assert client.post("/webhook/vonage/status", content=b"nope").status_code == 200
    assert "vonage status: ? ?" in capsys.readouterr().out


def test_mask_phones_keeps_last_four():
    from gateway.worker import mask_phones

    assert mask_phones("919999999999") == "********9999"
    assert mask_phones("to whatsapp:+919999999999 ok") == (
        "to whatsapp:+********9999 ok"
    )
    assert mask_phones("short 12345") == "short 12345"
    assert mask_phones("") == ""


def test_provider_switch_selects_vonage(monkeypatch):
    monkeypatch.setattr(app_module, "PROVIDER", "vonage")
    monkeypatch.setattr(app_module, "VONAGE_API_KEY", "k")
    monkeypatch.setattr(app_module, "VONAGE_API_SECRET", "s")
    monkeypatch.setattr(app_module, "VONAGE_WHATSAPP_NUMBER", "91111")
    assert isinstance(app_module._sender(), VonageSender)
    monkeypatch.setattr(app_module, "PROVIDER", "twilio")
    from gateway.senders import TwilioSender
    assert isinstance(app_module._sender(), TwilioSender)
