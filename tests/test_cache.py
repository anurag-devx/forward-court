"""Offline tests for the repeat cache (isolated SQLite per test)."""
import sqlite3
from types import SimpleNamespace

import pytest

import cache as cache_module
import main
from intake.models import IntakeResult
from links.scamtext import ScamReading
from tests.fakes import FakeClient, dead_http_client


@pytest.fixture
def cache_on(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE", "on")
    monkeypatch.setenv("FC_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _verdict(name, fact_checked_by=None):
    return {"verdict": name, "fact_checked_by": fact_checked_by}


def test_enabled_flag(monkeypatch):
    monkeypatch.delenv("CACHE", raising=False)
    assert cache_module.enabled() is True
    monkeypatch.setenv("CACHE", "0")
    assert cache_module.enabled() is False
    monkeypatch.setenv("CACHE", "off")
    assert cache_module.enabled() is False


def test_normalize_key_text():
    assert cache_module.normalize_key_text("  Hello   WORLD\n") == "Hello WORLD"


def test_keys_differ_by_content_and_flags(tmp_path):
    base = cache_module.text_key("hello", "fc=1")
    assert cache_module.text_key("hello ", "fc=1") == base
    assert cache_module.text_key("hello", "fc=0") != base
    assert cache_module.text_key("other", "fc=1") != base
    img = tmp_path / "a.png"
    img.write_bytes(b"bytes-one")
    key = cache_module.media_key_for_file(str(img), None, "fc=1")
    assert cache_module.media_key_for_file(str(img), "cap", "fc=1") != key
    img.write_bytes(b"bytes-two")
    assert cache_module.media_key_for_file(str(img), None, "fc=1") != key
    assert cache_module.media_key_for_file(str(tmp_path / "nope"), None, "x") is None


def test_ttl_classes():
    week = 7 * 24 * 3600
    assert cache_module.ttl_for_verdict(_verdict("DANGEROUS")) == week
    assert cache_module.ttl_for_verdict(_verdict("False", "altnews.in")) == week
    assert cache_module.ttl_for_verdict(_verdict("False")) == 24 * 3600
    assert cache_module.ttl_for_verdict(_verdict("Unverified")) == 3600
    assert cache_module.ttl_for_verdict(_verdict("True")) == 24 * 3600
    assert cache_module.ttl_for_verdict(_verdict("SUSPICIOUS")) == 24 * 3600


def test_store_lookup_roundtrip_and_expiry(cache_on):
    tracker = SimpleNamespace(trace={"verdict": _verdict("True")})
    cache_module.maybe_store("k1", "CARD", tracker)
    assert cache_module.lookup("k1") == {"card": "CARD", "verdict": _verdict("True")}
    assert cache_module.lookup("missing") is None
    db_path = cache_on / "cache" / "forwardcourt.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE cache SET created = 0 WHERE key = 'k1'")
    conn.commit()
    conn.close()
    assert cache_module.lookup("k1") is None


def test_store_skips_verdictless_runs(cache_on):
    tracker = SimpleNamespace(trace={})
    cache_module.maybe_store("k", "CARD", tracker)
    assert not (cache_on / "cache" / "forwardcourt.sqlite").exists()


def test_db_errors_miss_through(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("CACHE", "on")
    monkeypatch.setenv("FC_CACHE_DIR", str(blocker))
    assert cache_module.lookup("k") is None
    tracker = SimpleNamespace(trace={"verdict": _verdict("True")})
    cache_module.maybe_store("k", "CARD", tracker)  # must not raise


def _dangerous_client():
    return FakeClient(
        parsed_by_model={
            IntakeResult: IntakeResult(
                language="en", claims=["some claim"], presented_as_recent=False
            ),
            ScamReading: ScamReading(
                scam_type="Fake KYC update", signals=["otp_pin", "urgency"]
            ),
        }
    )


def test_repeat_run_hits_cache(cache_on):
    first = _dangerous_client()
    card, _ = main.run_forward(
        "hello", client=first, http_client=dead_http_client()
    )
    calls_after_first = len(first.responses.calls)
    assert calls_after_first > 0
    second = _dangerous_client()
    card2, tracker2 = main.run_forward(
        "hello", client=second, http_client=dead_http_client()
    )
    assert card2 == card
    assert second.responses.calls == []
    assert "cache hit." in tracker2.notes


def test_no_cache_param_bypasses(cache_on):
    first = _dangerous_client()
    main.run_forward(
        "hello", client=first, http_client=dead_http_client(), cache=False
    )
    second = _dangerous_client()
    main.run_forward(
        "hello", client=second, http_client=dead_http_client(), cache=False
    )
    assert len(second.responses.calls) > 0


def _png_file(tmp_path):
    from demo import make_screenshot

    return make_screenshot(
        str(tmp_path / "shot.png"), headline="H", standfirst="S",
        url_path="/x",
    )


def test_media_hit_needs_no_pipeline(cache_on):
    shot = _png_file(cache_on)
    key = cache_module.media_key_for_file(
        shot, None, main._cache_flags(True, True)
    )
    prime = SimpleNamespace(trace={"verdict": _verdict("True")})
    cache_module.maybe_store(key, "CACHED CARD", prime)
    bare = FakeClient(parsed_by_model={})
    card, _ = main.run_image_forward(
        shot, client=bare, http_client=dead_http_client()
    )
    assert card == "CACHED CARD"
    assert bare.responses.calls == []


def test_record_mode_bypasses_cache(monkeypatch, tmp_path):
    seen = {}

    def fake_run(text, **kwargs):
        seen.update(kwargs)
        tracker = SimpleNamespace(trace={})
        return "CARD", tracker

    monkeypatch.setattr(main, "run_forward", fake_run)
    main._run_demo_sample(
        {"kind": "text", "id": "t", "text": "hi"}, str(tmp_path),
        cache=False,
    )
    assert seen.get("cache") is False


def test_ground_truth_bypasses_cache(cache_on):
    shot = _png_file(cache_on)
    key = cache_module.media_key_for_file(
        shot, None, main._cache_flags(True, True)
    )
    prime = SimpleNamespace(trace={"verdict": _verdict("True")})
    cache_module.maybe_store(key, "CACHED CARD", prime)
    bare = FakeClient(parsed_by_model={})
    with pytest.raises(KeyError):
        main.run_image_forward(
            shot, client=bare, http_client=dead_http_client(),
            ground_truth=["https://orig.com/pic"],
        )
