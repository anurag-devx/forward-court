"""Suite-wide fixtures: the repeat cache stays off unless a test opts in."""
import pytest


@pytest.fixture(autouse=True)
def _no_cache_by_default(monkeypatch):
    monkeypatch.setenv("CACHE", "off")
