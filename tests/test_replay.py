import json
import socket

import main
import demo.trace as trace_module
import demo.replay as replay_module
from config import UsageTracker
from demo.replay import (
    load_samples,
    provenance_line,
    render_comparison,
    run_replay,
)
from demo.trace import (
    build_recording,
    save_recording,
    scrub,
    validate_recording,
)


def _lived_tracker(**trace):
    tracker = UsageTracker()
    tracker.stages["intake"] = (10, 5)
    tracker.times["intake"] = 1.2
    tracker.note("done.")
    for key, payload in trace.items():
        tracker.record(key, payload)
    return tracker


def _full_trace(verdict="True", key="https://t1.example/x"):
    return {
        "claims": [{"text": "c", "kind": "text_claim"}],
        "link_safety": {
            "urls": [], "expanded": {}, "flagged_hosts": [],
            "signals": {}, "scam_signals": [],
        },
        "article": None,
        "evidence": [{
            "claim": "c", "source_url": key, "outlet": "t1.example",
            "tier": 1, "date": "2026-10-01", "snippet": "s",
            "fact_check": False, "exact_match": False,
            "shared_article": False, "claim_review": None,
        }],
        "relevance_drops": [],
        "claimreview_hits": [],
        "debate": {
            "prosecution": [{"text": "p", "source_url": key}],
            "defense": [{"text": "d", "source_url": key}],
        },
        "votes": [
            {
                "verdict": verdict, "reason": "r", "key_source": key,
                "confidence": "high",
            }
        ] * 3,
        "verdict": {
            "verdict": verdict, "reason": "r", "key_source": key,
            "language": "en", "confidence": "high",
            "fact_checked_by": None, "scam_type": None,
            "article_date": None,
        },
    }


def _recording(sample_id="text_real", **overrides):
    base = build_recording(
        sample_id=sample_id, kind="text", label="L", expected="True",
        input_desc={"text": "c"}, card="CARD",
        tracker=_lived_tracker(**_full_trace()), elapsed_s=3.0,
    )
    base.update(overrides)
    return base


# --- recorder ---


def test_build_save_roundtrip_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_module, "RECORDINGS_DIR", tmp_path)
    recording = _recording()
    assert validate_recording(recording) == []
    path = save_recording(recording)
    assert path.parent == tmp_path
    assert validate_recording(json.loads(path.read_text())) == []


def test_scrub_strips_keys_phones_and_tunnel(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-12345")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://tunnel-abc.trycloudflare.com")
    recording = _recording()
    recording["notes"] = [
        "key sk-test-12345 used",
        "call 919999999999",
        "hook https://tunnel-abc.trycloudflare.com/webhook",
    ]
    cleaned = scrub(recording)
    text = json.dumps(cleaned)
    assert "sk-test-12345" not in text
    assert "919999999999" not in text
    assert "9999" in text  # last 4 kept
    assert "tunnel-abc" not in text


def test_validate_catches_bad_shapes():
    assert validate_recording({}) != []
    bad_verdict = _recording()
    bad_verdict["verdict"]["verdict"] = "Maybe"
    assert any("bad verdict" in e for e in validate_recording(bad_verdict))
    bad_vote = _recording()
    bad_vote["votes"][0]["verdict"] = "Maybe"
    assert any("bad vote" in e for e in validate_recording(bad_vote))
    bad_evidence = _recording()
    del bad_evidence["evidence"][0]["tier"]
    assert any("evidence" in e for e in validate_recording(bad_evidence))


# --- record mode ---


def test_demo_record_writes_all_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trace_module, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(
        main, "run_forward",
        lambda text, **_: ("CARD", _lived_tracker(**_full_trace())),
    )
    monkeypatch.setattr(
        main, "run_image_forward",
        lambda path, *_, **__: ("CARD", _lived_tracker(**_full_trace())),
    )
    assert main.main(["--demo", "--record"]) == 0
    ids = {sample["id"] for sample in load_samples()}
    assert {path.stem for path in tmp_path.glob("*.json")} == ids
    for path in tmp_path.glob("*.json"):
        assert validate_recording(json.loads(path.read_text())) == []
    assert "recorded:" in capsys.readouterr().out


def test_demo_record_rejects_invalid_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trace_module, "RECORDINGS_DIR", tmp_path)
    bare = UsageTracker()  # no verdict recorded
    monkeypatch.setattr(main, "run_forward", lambda t, **_: ("CARD", bare))
    monkeypatch.setattr(
        main, "run_image_forward", lambda p, *_, **__: ("CARD", bare)
    )
    assert main.main(["--demo", "--record"]) == 1
    assert "trace invalid" in capsys.readouterr().out
    assert list(tmp_path.glob("*.json")) == []


def test_record_replay_arg_guards():
    assert main.main(["--record"]) == 2
    assert main.main(["--replay"]) == 2
    assert main.main(["--demo", "--record", "--replay"]) == 2
    assert main.main(["--demo", "--replay", "extra"]) == 2


# --- replay ---


def _write_all(monkeypatch, tmp_path):
    monkeypatch.setattr(replay_module, "RECORDINGS_DIR", tmp_path)
    verdicts = {
        "text_real": "True", "text_fake": "False",
        "shot_real": "True", "shot_fake": "False",
        "link_real": "True", "link_fake": "Misleading",
        "malicious": "DANGEROUS",
    }
    for sample in load_samples():
        verdict = verdicts[sample["id"]]
        trace = _full_trace(verdict=verdict)
        if sample["id"] == "malicious":
            trace["verdict"]["key_source"] = ""
            trace["verdict"]["scam_type"] = "Fake KYC update"
            trace["votes"] = []
        recording = build_recording(
            sample_id=sample["id"], kind=sample["kind"],
            label=sample["label"], expected=sample["expected"],
            input_desc={"text": sample.get("text", "")},
            card=f"CARD {sample['id']}",
            tracker=_lived_tracker(**trace), elapsed_s=1.0,
        )
        (tmp_path / f"{sample['id']}.json").write_text(json.dumps(recording))


def test_replay_offline_no_key_no_sockets(tmp_path, monkeypatch, capsys):
    _write_all(monkeypatch, tmp_path)
    import os

    for key in [
        "MODEL_API_KEY", "SERPAPI_KEY", "GSB_API_KEY",
        "TWILIO_AUTH_TOKEN", "VONAGE_API_SECRET",
    ]:
        monkeypatch.delenv(key, raising=False)

    class BlockedSocket:
        def __init__(self, *args, **kwargs):
            raise AssertionError("socket used during replay")

    monkeypatch.setattr(socket, "socket", BlockedSocket)
    assert os.environ.get("MODEL_API_KEY") is None
    assert main.main(["--demo", "--replay"]) == 0
    out = capsys.readouterr().out
    assert "REPLAY — recorded live on " in out
    assert "no models called" in out
    assert "| text | True | False |" in out
    assert "| malicious | — | DANGEROUS |" in out
    assert "provenance: 1 claim(s) (text_claim) →" in out
    assert "CARD text_real" in out


def test_replay_provenance_variants():
    debate = _recording()
    line = provenance_line(debate)
    assert "1 claim(s) (text_claim)" in line
    assert "t1.example (tier 1)" in line
    assert "votes True, True, True → True" in line

    short = _recording()
    short["votes"] = []
    short["claimreview_hits"] = [{
        "source_url": "https://fc.example/x", "verdict": "False",
        "rating": "False",
    }]
    short["verdict"] = {
        "verdict": "False", "reason": "r",
        "key_source": "https://fc.example/x", "language": "en",
        "confidence": "high", "fact_checked_by": "fc.example",
        "scam_type": None, "article_date": None,
    }
    assert "ClaimReview fc.example → False" in provenance_line(short)

    scam = _recording()
    scam["votes"] = []
    scam["claims"] = []
    scam["link_safety"] = {
        "urls": ["http://evil.xyz"], "expanded": {},
        "flagged_hosts": [], "signals": {"plain_http": "u"},
        "scam_signals": ["otp_pin", "urgency"],
    }
    scam["verdict"] = {
        "verdict": "DANGEROUS", "reason": "r", "key_source": "",
        "language": "en", "confidence": "high",
        "fact_checked_by": None, "scam_type": "Fake KYC update",
        "article_date": None,
    }
    line = provenance_line(scam)
    assert "no claims" in line and "scam rule (Fake KYC update)" in line
    assert "link rules (" in line and line.endswith("DANGEROUS")


def test_render_comparison_uses_fake_key_source():
    recordings = {
        "text_real": _recording("text_real"),
        "text_fake": _recording(
            "text_fake",
            verdict={
                "verdict": "False", "reason": "r",
                "key_source": "https://debunk.example/y", "language": "en",
                "confidence": "high", "fact_checked_by": None,
                "scam_type": None, "article_date": None,
            },
        ),
    }
    table = render_comparison(recordings)
    assert "| text | True | False | debunk.example |" in table


def test_replay_tolerates_missing_recordings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(replay_module, "RECORDINGS_DIR", tmp_path)
    assert run_replay() == 0
    out = capsys.readouterr().out
    assert "REPLAY — recorded live on" in out
    assert "| text | not recorded | not recorded | — |" in out
    assert "| malicious | — | not recorded | — |" in out
    assert out.count("not recorded yet. Run: python main.py --demo --record") == 7


def test_replay_partial_recordings_shows_rest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(replay_module, "RECORDINGS_DIR", tmp_path)
    recording = _recording("text_real")
    (tmp_path / "text_real.json").write_text(json.dumps(recording))
    assert run_replay() == 0
    out = capsys.readouterr().out
    assert "| text | True | not recorded | — |" in out
    assert "CARD" in out  # present sample still renders


def test_replay_committed_recordings_pass(capsys):
    assert run_replay() == 0
    out = capsys.readouterr().out
    assert "REPLAY — recorded live on " in out
    assert "not recorded" not in out
    for sample in load_samples():
        assert f"=== {sample['id']} " in out


def test_committed_recordings_validate():
    from demo.trace import RECORDINGS_DIR

    paths = sorted(RECORDINGS_DIR.glob("*.json"))
    assert sorted(path.stem for path in paths) == sorted(
        sample["id"] for sample in load_samples()
    )
    for path in paths:
        recording = json.loads(path.read_text())
        assert validate_recording(recording) == []
        assert recording["verdict"]["verdict"] == recording["expected"]


def test_committed_recordings_hold_no_secrets():
    import os
    import re

    from demo.trace import RECORDINGS_DIR

    secrets = [
        value
        for key in (
            "MODEL_API_KEY", "SERPAPI_KEY", "GSB_API_KEY",
            "TWILIO_AUTH_TOKEN", "VONAGE_API_SECRET",
            "VONAGE_SIGNATURE_SECRET",
        )
        if (value := os.environ.get(key, ""))
    ]
    for path in sorted(RECORDINGS_DIR.glob("*.json")):
        text = path.read_text()
        for secret in secrets:
            assert secret not in text
        assert not re.search(r"\d{8,}", text)
