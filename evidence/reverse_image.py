"""Reverse image search with a pluggable provider.

SerpApiLensProvider is the first provider: it uploads local bytes to SerpAPI's
Image API (500KB max, hence client-side downscaling), then runs a Google Lens
search with the returned image_id. exact_matches identify the image's origin;
visual_matches are context only. Provider errors raise ReverseSearchError so
the pipeline can continue on vision + text grounding.
"""
from __future__ import annotations

import io
from typing import Protocol

import httpx
from PIL import Image
from pydantic import BaseModel

from .dates import extract_date
from .models import EvidenceItem
from .pages import enrich_from_page
from .urls import normalize_url

UPLOAD_URL = "https://serpapi.com/image"
SEARCH_URL = "https://serpapi.com/search.json"
SEARCH_TIMEOUT = 30.0
UPLOAD_MAX_BYTES = 500 * 1024
UPLOAD_TARGET_BYTES = 480 * 1024
MATCHES_PER_IMAGE = 10
EXACT_EVIDENCE_CAP = 8
VISUAL_EVIDENCE_CAP = 4
ENRICH_EXACT = 4
ENRICH_VISUAL = 2
CROP_SEARCH_EXACT_THRESHOLD = 3


class ReverseMatch(BaseModel):
    source_url: str
    page_title: str
    date: str | None = None
    thumbnail: str | None = None
    exact: bool


class ReverseSearchError(RuntimeError):
    """Reverse image search failed. The pipeline continues without it."""


class ReverseProvider(Protocol):
    def search(self, image: bytes) -> list[ReverseMatch]:
        """Search one image. Raises ReverseSearchError on provider failure."""
        ...


def fit_for_upload(image: bytes) -> bytes:
    """Downscale/compress image bytes to SerpAPI's 500KB upload limit."""
    if len(image) <= UPLOAD_TARGET_BYTES:
        return image
    img = Image.open(io.BytesIO(image)).convert("RGB")
    for max_side, quality in ((1600, 80), (1200, 70), (800, 60), (800, 40)):
        scale = min(1.0, max_side / max(img.size))
        resized = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=quality)
        if len(buffer.getvalue()) <= UPLOAD_TARGET_BYTES:
            return buffer.getvalue()
    raise ReverseSearchError("Image exceeds SerpAPI's 500KB upload limit.")


class SerpApiLensProvider:
    """Google Lens via SerpAPI: upload bytes, then search by image_id."""

    def __init__(self, api_key: str, *, http_client: httpx.Client | None = None):
        self.api_key = api_key
        self.http_client = http_client

    def _post(self, url: str, **kwargs):
        if self.http_client is not None:
            return self.http_client.post(url, timeout=SEARCH_TIMEOUT, **kwargs)
        with httpx.Client() as client:
            return client.post(url, timeout=SEARCH_TIMEOUT, **kwargs)

    def _get(self, url: str, **kwargs):
        if self.http_client is not None:
            return self.http_client.get(url, timeout=SEARCH_TIMEOUT, **kwargs)
        with httpx.Client() as client:
            return client.get(url, timeout=SEARCH_TIMEOUT, **kwargs)

    def search(self, image: bytes) -> list[ReverseMatch]:
        try:
            payload = fit_for_upload(image)
            upload = self._post(
                UPLOAD_URL,
                files={"image": ("image.jpg", payload, "image/jpeg")},
                data={"api_key": self.api_key},
            )
            upload.raise_for_status()
            image_id = upload.json().get("image_id")
            if not image_id:
                raise ReverseSearchError(
                    f"SerpAPI upload failed: {upload.text[:200]}"
                )
            response = self._get(
                SEARCH_URL,
                params={
                    "engine": "google_lens",
                    "type": "all",
                    "image_id": image_id,
                    "api_key": self.api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
        except ReverseSearchError:
            raise
        except Exception as exc:
            # Never include str(exc): httpx errors can echo the request URL,
            # which carries the API key as a query parameter.
            raise ReverseSearchError("SerpAPI Lens request failed.") from exc
        if "error" in data:
            raise ReverseSearchError(f"SerpAPI Lens failed: {data['error'][:200]}")
        matches: list[ReverseMatch] = []
        for exact, key in ((True, "exact_matches"), (False, "visual_matches")):
            for entry in data.get(key, []) or []:
                link = entry.get("link")
                if not link or not link.startswith(("http://", "https://")):
                    continue
                matches.append(
                    ReverseMatch(
                        source_url=link,
                        page_title=entry.get("title") or "",
                        thumbnail=entry.get("thumbnail"),
                        exact=exact,
                    )
                )
                if len(matches) >= MATCHES_PER_IMAGE:
                    return matches
        return matches


def search_images(
    full: bytes, crops: list[bytes], *, provider: ReverseProvider
) -> list[ReverseMatch]:
    """Search the full image, then crops only if <3 exact matches so far.

    A failing crop search is skipped; only a full-image failure raises.
    Deduplicates by normalized URL with exact matches winning over visual.
    """
    matches = provider.search(full)
    if sum(1 for match in matches if match.exact) < CROP_SEARCH_EXACT_THRESHOLD:
        for crop in crops:
            try:
                matches.extend(provider.search(crop))
            except ReverseSearchError:
                continue
    deduped: dict[str, ReverseMatch] = {}
    for match in matches:
        key = normalize_url(match.source_url)
        if key not in deduped or (match.exact and not deduped[key].exact):
            deduped[key] = match
    return list(deduped.values())


def matches_to_evidence(
    matches: list[ReverseMatch], claim: str, *, http_client=None
) -> list[EvidenceItem]:
    """Convert matches to evidence: exact first, visual labeled visual-only.

    Only the top matches get page-fetch enrichment; the rest keep titles.
    """
    exact = [m for m in matches if m.exact][:EXACT_EVIDENCE_CAP]
    visual = [m for m in matches if not m.exact][:VISUAL_EVIDENCE_CAP]
    items: list[EvidenceItem] = []
    for index, match in enumerate(exact):
        if index < ENRICH_EXACT:
            snippet, date = enrich_from_page(
                match.source_url, match.page_title, match.page_title,
                http_client=http_client,
            )
        else:
            snippet, date = match.page_title, extract_date(match.page_title)
        items.append(
            EvidenceItem(
                claim=claim, source_url=match.source_url, date=date,
                snippet=snippet,
            )
        )
    for index, match in enumerate(visual):
        if index < ENRICH_VISUAL:
            snippet, date = enrich_from_page(
                match.source_url, match.page_title, match.page_title,
                http_client=http_client,
            )
        else:
            snippet, date = match.page_title, extract_date(match.page_title)
        items.append(
            EvidenceItem(
                claim=claim, source_url=match.source_url, date=date,
                snippet=snippet, visual_only=True,
            )
        )
    return items
