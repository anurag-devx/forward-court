"""Search-grounded evidence collection.

Uses the Responses API with the `web_search` tool plus
`include=["web_search_call.results"]` (per Meta's search-grounding docs).
Evidence URLs come ONLY from `web_search_call.results` metadata; the model's
text output is never used for URLs. Any grounding failure raises
GroundingError -- there is deliberately no ungrounded fallback.
"""
from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker

from config import agent_model

from dates import extract_date
from .models import EvidenceItem
from .pages import enrich_from_page
from .responses import field, web_search_results
from .urls import normalize_url, split_host_path


def _blocked(url: str, blocked_hosts: frozenset) -> bool:
    host, _ = split_host_path(url)
    return host is not None and host in blocked_hosts

SEARCH_TOOL = {"type": "web_search", "search_context_size": "low"}
INCLUDE_RESULTS = ["web_search_call.results"]
TOP_FETCH_PER_CLAIM = 3
MAX_TOOL_CALLS = 2
EVIDENCE_WORKERS = 6


class DaemonPool(ThreadPoolExecutor):
    """Thread pool whose workers never block process exit.

    Evidence pools may be abandoned by the run deadline while calls are
    still in flight; daemon threads let the process exit instead of
    waiting out socket timeouts.
    """

    def _adjust_thread_count(self):
        # Mirror of ThreadPoolExecutor._adjust_thread_count, except workers
        # are daemons and are never registered in _threads_queues, so the
        # interpreter never joins them at exit.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (
                self._thread_name_prefix or self, num_threads
            )
            thread = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            thread.start()
            self._threads.add(thread)

    def abandon(self) -> None:
        """Stop the pool without waiting: cancel queued work, orphan the rest.

        For the ClaimReview early exit: the run returns without waiting
        for grounding. Daemon threads finish (or die with the process)
        in the background; their results are never collected.
        """
        self._abandoned = True
        self.shutdown(wait=False, cancel_futures=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "_abandoned", False):
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)

SUPPORT_QUERY = "Find web evidence SUPPORTING this claim: {claim}"
REFUTE_QUERY = (
    "Find web evidence REFUTING, debunking, or fact-checking this claim: {claim}"
)


class GroundingError(RuntimeError):
    """Search grounding failed. Stop the run; never use ungrounded URLs."""


def _interleave(per_query: list[list]) -> list:
    """Round-robin across per-query result lists so both sides rank evenly."""
    ordered: list = []
    for index in range(max((len(q) for q in per_query), default=0)):
        for results in per_query:
            if index < len(results):
                ordered.append(results[index])
    return ordered


def _search_query(client, tracker, stage: str, claim: str, query: str):
    """Run one grounded query. Returns (results, searched)."""
    try:
        response = client.responses.create(
            model=agent_model(),
            input=query,
            tools=[dict(SEARCH_TOOL)],
            include=list(INCLUDE_RESULTS),
            max_tool_calls=MAX_TOOL_CALLS,
        )
    except Exception as exc:
        raise GroundingError(
            f"Search grounding failed for claim {claim!r}: {exc}"
        ) from exc
    if tracker is not None:
        tracker.add(response.usage, stage=stage)
    return web_search_results(response.output)


def fetch_page_unit(args) -> tuple:
    """Fetch one page for excerpt + date + ClaimReview. Never raises."""
    url, title, search_snippet, http_client = args
    return enrich_from_page(url, title, search_snippet, http_client=http_client)


def _ordered_unique(per_query: list[list], seen: set[str]) -> list[tuple]:
    """Interleaved (url, title, search_snippet) deduped against seen."""
    ordered: list[tuple] = []
    for result in _interleave(per_query):
        url = field(result, "url")
        if not url:
            continue
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(
            (url, field(result, "title") or "", field(result, "snippet") or "")
        )
    return ordered


def gather_evidence(
    claims: list[str], *, client, tracker=None, http_client=None,
    executor=None, blocked_hosts: frozenset = frozenset(),
) -> list[EvidenceItem]:
    """Collect grounded evidence for each claim (supporting + refuting queries).

    Searches and the top TOP_FETCH_PER_CLAIM page fetches per claim run on
    the executor (a private pool when none is given); results reassemble
    in submission order so output is deterministic. Fetch failures keep
    the search snippet and never abort the run. Raises GroundingError
    when a query performs no grounded search, when the API call fails,
    or when a claim ends up with zero grounded items.
    """
    if tracker is not None:
        tracker.mark("grounding")
    pool = executor if executor is not None else DaemonPool(
        max_workers=EVIDENCE_WORKERS
    )
    try:
        return _gather(
            claims, pool, client=client, tracker=tracker,
            http_client=http_client, blocked_hosts=blocked_hosts,
        )
    finally:
        if executor is None:
            pool.shutdown()


def _gather(claims, pool, *, client, tracker, http_client, blocked_hosts) -> list[EvidenceItem]:
    jobs: list[tuple] = []
    for number, claim in enumerate(claims, start=1):
        jobs.append(
            (number, claim, "support", SUPPORT_QUERY.format(claim=claim))
        )
        jobs.append(
            (number, claim, "refute", REFUTE_QUERY.format(claim=claim))
        )
    outcomes = [
        future.result()
        for future in [
            pool.submit(
                _search_query, client, tracker,
                f"grounding:claim{number}:{side}", claim, query,
            )
            for number, claim, side, query in jobs
        ]
    ]
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for index in range(0, len(jobs), 2):
        claim = jobs[index][1]
        per_query: list[list] = []
        for results, searched in outcomes[index:index + 2]:
            if not searched:
                raise GroundingError(
                    f"Model performed no grounded search for claim {claim!r}; stopping."
                )
            per_query.append(results)
        ordered = _ordered_unique(per_query, seen)
        if not ordered:
            raise GroundingError(
                f"No grounded evidence found for claim {claim!r}; stopping."
            )
        candidates = ordered[:TOP_FETCH_PER_CLAIM]
        selected = [
            triple
            for triple in candidates
            if not _blocked(triple[0], blocked_hosts)
        ]
        if len(selected) < len(candidates) and tracker is not None:
            tracker.note(
                f"skipped {len(candidates) - len(selected)} "
                "flagged URL(s) (unsafe)."
            )
        fetched = [
            future.result()
            for future in [
                pool.submit(fetch_page_unit, (url, title, snippet, http_client))
                for url, title, snippet in selected
            ]
        ]
        enriched = {
            normalize_url(url): triple
            for (url, _, _), triple in zip(selected, fetched)
        }
        for url, title, search_snippet in ordered:
            if _blocked(url, blocked_hosts):
                continue
            if normalize_url(url) in enriched:
                snippet, date, review = enriched[normalize_url(url)]
            else:
                snippet = search_snippet or title
                date = extract_date(f"{title}\n{snippet}")
                review = None
            items.append(
                EvidenceItem(
                    claim=claim, source_url=url, date=date, snippet=snippet,
                    claim_review=review,
                )
            )
    return items
