"""Fake Meta Model API client for offline tests."""
from __future__ import annotations

import threading
from concurrent.futures import Future
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
    """Mimics client.responses with canned parse/create results.

    Thread-safe. Lookup queries (input starting with "fact check:") pop
    from lookup_outputs when provided, so concurrent grounding and lookup
    stages get deterministic results; everything else shares outputs.
    """

    def __init__(
        self, parsed_by_model=None, outputs=None, parse_queue=None,
        lookup_outputs=None,
    ):
        self.parsed_by_model = dict(parsed_by_model or {})
        self.outputs = list(outputs or [])
        self.lookup_outputs = (
            list(lookup_outputs) if lookup_outputs is not None else None
        )
        self.parse_queue = list(parse_queue or [])
        self.calls = []
        self.lock = threading.Lock()

    def parse(self, **kwargs):
        with self.lock:
            self.calls.append(("parse", kwargs))
            if self.parse_queue:
                parsed = self.parse_queue.pop(0)
            else:
                parsed = self.parsed_by_model[kwargs["text_format"]]
        return SimpleNamespace(output_parsed=parsed, usage=FakeUsage())

    def create(self, **kwargs):
        with self.lock:
            self.calls.append(("create", kwargs))
            queue = self.outputs
            if (
                self.lookup_outputs is not None
                and str(kwargs.get("input", "")).startswith("fact check:")
            ):
                queue = self.lookup_outputs
            output = queue.pop(0)
        return SimpleNamespace(output=output, usage=FakeUsage())


class FakeFiles:
    """Mimics client.files with canned upload ids."""

    def __init__(self, file_id="file-test-123"):
        self.file_id = file_id
        self.calls = []
        self.deleted = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(id=self.file_id)

    def delete(self, file_id):
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


class FakeClient:
    def __init__(
        self, parsed_by_model=None, outputs=None, parse_queue=None,
        lookup_outputs=None,
    ):
        self.responses = FakeResponses(
            parsed_by_model, outputs, parse_queue, lookup_outputs
        )
        self.files = FakeFiles()


class SyncExecutor:
    """Runs submitted jobs immediately; deterministic stand-in for a pool."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirrors Future
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


def dead_http_client():
    """HTTP client that fails every request: keeps page-fetch tests offline."""

    def handler(request):
        raise httpx.ConnectError("offline")

    return httpx.Client(transport=httpx.MockTransport(handler))
