"""Fake Meta Model API client for offline tests."""
from __future__ import annotations

from types import SimpleNamespace

import httpx


class FakeUsage:
    input_tokens = 100
    output_tokens = 50
    total_tokens = 150


def search_call(urls, status="completed"):
    """Build a fake web_search_call output item from [(url, title, snippet)]."""
    return SimpleNamespace(
        type="web_search_call",
        status=status,
        results=[
            SimpleNamespace(type="text_result", url=u, title=t, snippet=s)
            for u, t, s in urls
        ],
    )


class FakeResponses:
    """Mimics client.responses with canned parse/create results."""

    def __init__(self, parsed_by_model=None, outputs=None):
        self.parsed_by_model = dict(parsed_by_model or {})
        self.outputs = list(outputs or [])
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        fmt = kwargs["text_format"]
        return SimpleNamespace(
            output_parsed=self.parsed_by_model[fmt], usage=FakeUsage()
        )

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(output=self.outputs.pop(0), usage=FakeUsage())


class FakeClient:
    def __init__(self, parsed_by_model=None, outputs=None):
        self.responses = FakeResponses(parsed_by_model, outputs)


def dead_http_client():
    """HTTP client that fails every request: keeps page-fetch tests offline."""

    def handler(request):
        raise httpx.ConnectError("offline")

    return httpx.Client(transport=httpx.MockTransport(handler))
