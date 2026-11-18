"""Bounded FINN research tools; the MCP client supplies the AI judgment."""

import asyncio
import json
import math
import re
from datetime import UTC, datetime
from typing import Any, Literal, get_args

import requests
from mcp.server.fastmcp.utilities.types import Image

from finn_deals.mcp.client import FinnListingClient
from finn_deals.mcp.deals import compare_listings, review_signals
from finn_deals.mcp.filters import (
    FilterGroup,
    FinnFilters,
    discover_options,
    validate_filters,
    verify_selected,
)
from finn_deals.mcp.output import (
    View,
    compact_comparison,
    compact_listing,
    compact_shortlist,
    search_window,
)

DEFAULT_PRICE_MAX = 1_000_000
DEFAULT_MAX_PAGES = 2
SortOrder = Literal[
    "RELEVANCE", "PUBLISHED_DESC", "PUBLISHED_ASC", "PRICE_ASC", "PRICE_DESC"
]
_client: FinnListingClient | None = None


def get_finn_client() -> FinnListingClient:
    """Return the shared FinnListingClient (lazily created singleton)."""
    global _client
    if _client is None:
        _client = FinnListingClient()
    return _client


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, default=str, allow_nan=False, separators=(",", ":")
    )


def _validate_search(
    query: str,
    price_from: float,
    price_to: float,
    max_pages: int,
    finn_filters: FinnFilters,
) -> None:
    validate_filters(finn_filters)
    if not query.strip() and not finn_filters:
        raise ValueError("Provide a query or at least one FINN filter")
    if not all(math.isfinite(x) for x in (price_from, price_to)):
        raise ValueError("prices must be finite")
    if not 0 <= price_from <= price_to:
        raise ValueError("prices must satisfy 0 <= price_from <= price_to")
    if not 1 <= max_pages <= 5:
        raise ValueError("max_pages must be between 1 and 5")


def _listing_to_dict(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize prices without emitting nonstandard NaN/Infinity JSON."""
    try:
        price = float(row["price_amount"])
    except (TypeError, ValueError, KeyError):
        return None
    if not math.isfinite(price) or price < 0:
        return None
    return {
        "ad_id": str(row.get("id") or ""),
        **{
            key: row.get(key)
            for key in (
                "title",
                "url",
                "location",
                "price_currency",
                "price_unit",
                "price_display",
                "image_url",
                "timestamp",
            )
        },
        "price_amount": price,
        "promoted": bool(row.get("promoted", False)),
    }


def _matches_title(title: str, required: list[str], excluded: list[str]) -> bool:
    """Match words/phrases so e.g. 13 does not accidentally match 130."""
    title = title.casefold()

    def contains(term: str) -> bool:
        return bool(
            re.search(
                r"(?<!\w)" + re.escape(term.strip().casefold()) + r"(?!\w)", title
            )
        )

    return all(contains(t) for t in required if t.strip()) and not any(
        contains(t) for t in excluded if t.strip()
    )


def _search(
    query: str,
    price_from: float,
    price_to: float,
    location: str,
    max_pages: int,
    sort: SortOrder,
    required_terms: list[str],
    exclude_terms: list[str],
    finn_filters: FinnFilters,
) -> dict[str, Any]:
    _validate_search(query, price_from, price_to, max_pages, finn_filters)
    if sort not in (
        "RELEVANCE",
        "PUBLISHED_DESC",
        "PUBLISHED_ASC",
        "PRICE_ASC",
        "PRICE_DESC",
    ):
        raise ValueError("Unsupported sort order")
    client = get_finn_client()
    listings: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = scanned = pages = unpriced = 0
    complete = False
    for result in client.iter_search(
        query=query.strip(),
        max_pages=max_pages,
        price_from=math.floor(price_from),
        price_to=math.ceil(price_to),
        sort=sort,
        trade_type=1,
        **finn_filters,
    ):
        verify_selected(result, finn_filters)
        pages += 1
        total = result.total_matches
        complete = result.is_end or not result.listings
        for listing in result.listings:
            if listing.id in seen:
                continue
            seen.add(listing.id)
            scanned += 1
            row = _listing_to_dict(
                {
                    "id": listing.id,
                    "title": listing.title,
                    "url": listing.url,
                    "location": listing.location,
                    "price_amount": listing.price_amount,
                    "price_currency": listing.price_currency,
                    "price_unit": listing.price_unit,
                    "price_display": listing.price_display,
                    "image_url": listing.image_url,
                    "timestamp": (
                        listing.timestamp.isoformat() if listing.timestamp else None
                    ),
                    "promoted": listing.promoted,
                }
            )
            if row is None:
                unpriced += 1
                continue
            if not price_from <= row["price_amount"] <= price_to:
                continue
            if location.strip().casefold() not in (listing.location or "").casefold():
                continue
            if not _matches_title(listing.title, required_terms, exclude_terms):
                continue
            listings.append(row)
    if sort in ("PRICE_ASC", "PRICE_DESC"):
        listings.sort(key=lambda row: row["price_amount"], reverse=sort == "PRICE_DESC")
    elif sort in ("PUBLISHED_ASC", "PUBLISHED_DESC"):
        listings.sort(
            key=lambda row: row["timestamp"] or "", reverse=sort == "PUBLISHED_DESC"
        )
    return {
        "query": query.strip(),
        "count": len(listings),
        "listings": listings,
        "fetched_at": datetime.now(UTC).isoformat(),
        "total_matches": total,
        "scanned_count": scanned,
        "pages_fetched": pages,
        "truncated": not complete or scanned < total,
        "unpriced_skipped": unpriced,
        "filters": {
            "price_from": price_from,
            "price_to": price_to,
            "location": location,
            "finn_filters": finn_filters,
            "required_terms": required_terms,
            "exclude_terms": exclude_terms,
            "sort": sort,
        },
        "coverage_note": "Bounded sample of Torget sale ads. FINN filters apply before paging. Location substring and title filters apply after fetching; empty local results do not prove none exist. Asking prices exclude shipping and payment fees.",
    }


def _discover_filters(
    query: str, group: FilterGroup, text: str, limit: int, finn_filters: FinnFilters
) -> dict[str, Any]:
    if group not in get_args(FilterGroup) or not 1 <= limit <= 50:
        raise ValueError("Use a supported filter group and a limit between 1 and 50")
    validate_filters(finn_filters)
    result = get_finn_client().search(query.strip(), trade_type=1, **finn_filters)
    verify_selected(result, finn_filters)
    return discover_options(result, group, text, limit)


async def finn_discover_filters(
    group: FilterGroup,
    query: str = "",
    text: str = "",
    limit: int = 8,
    finn_filters: FinnFilters | None = None,
) -> str:
    """Look up native FINN filters by Norwegian label, e.g. group="location", text="Stavanger".

    Pass an option's finn_filters object to finn_search or finn_find_deals, or combine
    keys from different groups. Discovery takes one page. Hits reflect the upstream query
    and selected FINN filters. Text matches any part of the hierarchy and includes zero-hit
    options. Without text, returns the most populated options.
    """
    return _json(
        await asyncio.to_thread(
            _discover_filters, query, group, text, limit, finn_filters or {}
        )
    )


async def finn_search(
    query: str = "",
    price_from: float = 0.0,
    price_to: float = DEFAULT_PRICE_MAX,
    location: str = "",
    max_pages: int = DEFAULT_MAX_PAGES,
    sort: SortOrder = "RELEVANCE",
    required_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    finn_filters: FinnFilters | None = None,
    view: View = "compact",
    limit: int | None = None,
    offset: int = 0,
) -> str:
    """Search sale ads. Returns 10 compact rows by default. Pass next_offset as offset for more sampled rows.

    Use finn_filters from finn_discover_filters to narrow before paging (1–5 pages).
    Location substring and required/excluded title terms filter the sample locally.
    With view="full" and no limit, returns all matched rows and listing fields.
    """
    if offset < 0 or (limit is not None and not 1 <= limit <= 250):
        raise ValueError("Use a nonnegative offset and a limit between 1 and 250")
    search = await asyncio.to_thread(
        _search,
        query,
        price_from,
        price_to,
        location,
        max_pages,
        sort,
        required_terms or [],
        exclude_terms or [],
        finn_filters or {},
    )
    return _json(search_window(search, view, limit, offset))


async def finn_get_listing(ad_id: str, view: View = "compact") -> str:
    """Inspect one numeric ad ID. Compact includes a description excerpt, condition, availability and photo count.
    Request view="full" for the entire description, photo URLs, seller fields and metadata.
    """
    detail = await asyncio.to_thread(get_finn_client().get_item, ad_id)
    return _json(detail if view == "full" else compact_listing(detail))


def _get_image(ad_id: str, image_index: int) -> Image:
    if image_index < 0:
        raise ValueError("image_index must be nonnegative")
    client = get_finn_client()
    detail = client.get_item(ad_id)
    images = detail.get("image_urls") or (
        [detail["image_url"]] if detail.get("image_url") else []
    )
    if image_index >= len(images):
        raise ValueError(
            f"Image {image_index} unavailable; listing has {len(images)} images"
        )
    response = client.session.get(images[image_index], timeout=15)
    response.raise_for_status()
    data = response.content
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Image exceeds the 10 MiB tool limit; inspect another photo")
    if data.startswith(b"\xff\xd8\xff"):
        image_format = "jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        image_format = "png"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        image_format = "gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        image_format = "webp"
    else:
        raise ValueError("Unsupported or invalid image data from FINN")
    return Image(data=data, format=image_format)


async def finn_get_image(ad_id: str, image_index: int = 0) -> Image:
    """View a listing photo (zero-based image_index, default main photo).

    Inspect wear, model labels, accessories and damage. Image format is detected
    from bytes because FINN image URLs often have no extension.
    """
    return await asyncio.to_thread(_get_image, ad_id, image_index)


def _find_deals(
    query: str,
    price_from: float,
    price_to: float,
    location: str,
    max_pages: int,
    sort: SortOrder,
    required_terms: list[str],
    exclude_terms: list[str],
    limit: int,
    finn_filters: FinnFilters,
) -> dict[str, Any]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    search = _search(
        query,
        price_from,
        price_to,
        location,
        max_pages,
        sort,
        required_terms,
        exclude_terms,
        finn_filters,
    )
    rows = search.pop("listings")
    candidates = []
    for row in rows[:limit]:
        try:
            detail = get_finn_client().get_item(row["ad_id"])
            candidates.append(
                {
                    **row,
                    **detail,
                    "detail_status": "ok",
                    "review_signals": review_signals(detail),
                }
            )
        except (requests.RequestException, ValueError) as exc:
            candidates.append(
                {
                    **row,
                    "detail_status": "error",
                    "detail_error": str(exc),
                    "review_signals": ["Listing details could not be verified"],
                }
            )
    return {
        **search,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "search_sample": rows,
        "assessment": "Research leads, not confirmed deals. No market-value estimate or bargain score is inferred from a broad search.",
        "next_steps": [
            "Narrow broad searches to exact products/models, sizes and variants before comparing prices.",
            "Reject unavailable ads, wanted ads, accessories, incomplete bundles and incompatible variants.",
            "Read descriptions and photos; compare condition, quantity, included parts and seller claims.",
            "Search the same product without your budget/location restriction to find independent comparison ads.",
            "Call finn_compare_listings with candidate and selected comparable ad IDs; explain any differences.",
            "Account for shipping, payment fees, travel and repairs before recommending a purchase.",
        ],
    }


async def finn_find_deals(
    query: str = "",
    price_from: float = 1.0,
    price_to: float = DEFAULT_PRICE_MAX,
    location: str = "",
    max_pages: int = DEFAULT_MAX_PAGES,
    limit: int = 5,
    sort: SortOrder = "PRICE_ASC",
    required_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
    finn_filters: FinnFilters | None = None,
    view: View = "compact",
) -> str:
    """Shortlist up to 10 research leads with condition, seller excerpts and review signals.
    Use finn_filters from finn_discover_filters for native filters. PRICE_ASC finds cheap leads.
    Request view="full" for complete candidates, photo URLs and the entire search sample.
    """
    shortlist = await asyncio.to_thread(
        _find_deals,
        query,
        price_from,
        price_to,
        location,
        max_pages,
        sort,
        required_terms or [],
        exclude_terms or [],
        limit,
        finn_filters or {},
    )
    return _json(shortlist if view == "full" else compact_shortlist(shortlist))


def _compare(ad_ids: list[str]) -> dict[str, Any]:
    ids = list(dict.fromkeys(ad_ids))
    if not 2 <= len(ids) <= 10:
        raise ValueError(
            "Provide 2–10 distinct ad IDs: candidates and selected comparables"
        )
    listings = []
    errors = []
    for ad_id in ids:
        try:
            listings.append(get_finn_client().get_item(ad_id))
        except (requests.RequestException, ValueError) as exc:
            errors.append({"ad_id": ad_id, "error": str(exc)})
    return {
        **compare_listings(listings),
        "fetch_errors": errors,
        "fetched_at": datetime.now(UTC).isoformat(),
    }


async def finn_compare_listings(ad_ids: list[str], view: View = "compact") -> str:
    """Compare asking prices for 2–10 distinct, caller-selected ads. Percentages require 3 active priced NOK peers.
    Verify model, condition and bundle equivalence. Request view="full" for all listing fields.
    """
    comparison = await asyncio.to_thread(_compare, ad_ids)
    return _json(comparison if view == "full" else compact_comparison(comparison))
