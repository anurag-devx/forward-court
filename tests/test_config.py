import pytest

from config import AGENT_MODEL, CONTRIBUTOR_MODEL, UsageTracker, agent_model, get_client


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
