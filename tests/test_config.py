import re

import pytest

from config import (
    AGENT_MODEL,
    CONTRIBUTOR_MODEL,
    UsageTracker,
    agent_model,
    get_client,
    gsb_api_key,
    gsb_enabled,
    max_claims,
    reverse_search_enabled,
    stage_timer,
    zoom_enabled,
)


def test_tracker_accumulates_and_summarizes():
    from types import SimpleNamespace

    tracker = UsageTracker()
    tracker.add(SimpleNamespace(input_tokens=100, output_tokens=50))
    tracker.add(SimpleNamespace(input_tokens=10, output_tokens=5))
    assert tracker.total_tokens == 165
    assert tracker.summary() == "Tokens: 110 in / 55 out / 165 total"


def test_tracker_ignores_none_and_missing_fields():
    from types import SimpleNamespace

    tracker = UsageTracker()
    tracker.add(None)
    tracker.add(SimpleNamespace())
    assert tracker.summary() == "Tokens: 0 in / 0 out / 0 total"


def test_breakdown_reports_per_stage_in_order():
    from types import SimpleNamespace

    tracker = UsageTracker()
    tracker.add(SimpleNamespace(input_tokens=100, output_tokens=50), stage="intake")
    tracker.add(SimpleNamespace(input_tokens=10, output_tokens=5), stage="judge")
    tracker.add(SimpleNamespace(input_tokens=1, output_tokens=1), stage="intake")
    assert tracker.breakdown() == [
        "  intake: 101 in / 51 out / 152 total",
        "  judge: 10 in / 5 out / 15 total",
    ]
    assert tracker.total_tokens == 167


def test_unlabeled_usage_counts_toward_total_only():
    from types import SimpleNamespace

    tracker = UsageTracker()
    tracker.add(SimpleNamespace(input_tokens=4, output_tokens=2))
    assert tracker.breakdown() == []
    assert tracker.total_tokens == 6


def test_get_client_requires_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        get_client()


def test_contributor_mode_default_on(monkeypatch):
    monkeypatch.delenv("FC_CONTRIBUTOR", raising=False)
    assert agent_model() == CONTRIBUTOR_MODEL


@pytest.mark.parametrize("value", ["0", "false", "NO", "off"])
def test_contributor_mode_explicit_off(monkeypatch, value):
    monkeypatch.setenv("FC_CONTRIBUTOR", value)
    assert agent_model() == AGENT_MODEL


def test_contributor_mode_explicit_on(monkeypatch):
    monkeypatch.setenv("FC_CONTRIBUTOR", "1")
    assert agent_model() == CONTRIBUTOR_MODEL


def test_client_has_timeout_and_retries(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    client = get_client()
    assert client.timeout == 60.0
    assert client.max_retries == 2


def test_tracker_marks_current_stage():
    tracker = UsageTracker()
    assert tracker.current is None
    tracker.mark("grounding")
    assert tracker.current == "grounding"


def test_timed_accumulates_and_breakdown_shows_seconds():
    import time
    from types import SimpleNamespace

    tracker = UsageTracker()
    with tracker.timed("judge"):
        time.sleep(0.05)
    with tracker.timed("judge"):
        pass
    assert tracker.times["judge"] >= 0.05
    tracker.add(SimpleNamespace(input_tokens=10, output_tokens=5), stage="judge")
    tracker.add(SimpleNamespace(input_tokens=1, output_tokens=1), stage="intake")
    lines = tracker.breakdown()
    assert re.search(r"total \(\d+\.\ds\)$", lines[0])
    assert lines[1] == "  intake: 1 in / 1 out / 2 total"


def test_stage_timer_tolerates_missing_tracker():
    with stage_timer(None, "judge"):
        pass


@pytest.mark.parametrize("value", ["on", "1", "true", "yes", "ON"])
def test_feature_flags_on_values(monkeypatch, value):
    monkeypatch.setenv("REVERSE_SEARCH", value)
    monkeypatch.setenv("GSB", value)
    assert reverse_search_enabled() is True
    assert gsb_enabled() is True


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "", "maybe"])
def test_feature_flags_default_off(monkeypatch, value):
    monkeypatch.setenv("REVERSE_SEARCH", value)
    monkeypatch.setenv("GSB", value)
    assert reverse_search_enabled() is False
    assert gsb_enabled() is False


def test_feature_flags_unset_default_off(monkeypatch):
    monkeypatch.delenv("REVERSE_SEARCH", raising=False)
    monkeypatch.delenv("GSB", raising=False)
    assert reverse_search_enabled() is False
    assert gsb_enabled() is False


def test_zoom_defaults_on(monkeypatch):
    monkeypatch.delenv("ZOOM", raising=False)
    assert zoom_enabled() is True
    monkeypatch.setenv("ZOOM", "on")
    assert zoom_enabled() is True


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "OFF"])
def test_zoom_off_values(monkeypatch, value):
    monkeypatch.setenv("ZOOM", value)
    assert zoom_enabled() is False


def test_gsb_api_key_gated_by_flag(monkeypatch):
    monkeypatch.setenv("GSB_API_KEY", "k")
    monkeypatch.setenv("GSB", "off")
    assert gsb_api_key() == ""
    monkeypatch.setenv("GSB", "on")
    assert gsb_api_key() == "k"


def test_max_claims_default_and_override(monkeypatch):
    monkeypatch.delenv("FC_MAX_CLAIMS", raising=False)
    assert max_claims() == 3
    monkeypatch.setenv("FC_MAX_CLAIMS", "2")
    assert max_claims() == 2


def test_max_claims_clamps_garbage(monkeypatch):
    monkeypatch.setenv("FC_MAX_CLAIMS", "nope")
    assert max_claims() == 3
    monkeypatch.setenv("FC_MAX_CLAIMS", "0")
    assert max_claims() == 1
    monkeypatch.setenv("FC_MAX_CLAIMS", "99")
    assert max_claims() == 3
