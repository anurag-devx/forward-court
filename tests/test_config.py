import pytest

from config import UsageTracker, get_client


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


def test_get_client_requires_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        get_client()
