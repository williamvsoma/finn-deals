"""MCP tools backed by the Finn.no listing client."""

import json
import logging
import math
from typing import Any

from mcp.server.fastmcp.utilities.types import Image

from finn_deals.mcp.client import FinnListingClient

logger = logging.getLogger(__name__)

DEFAULT_PRICE_MAX = 1_000_000
DEFAULT_MAX_PAGES = 2


_client: FinnListingClient | None = None


def get_finn_client() -> FinnListingClient:
    """Return the shared FinnListingClient (lazily created singleton)."""
    global _client
    if _client is None:
        _client = FinnListingClient()
    return _client


def _listing_to_dict(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single search-result row into a JSON-friendly listing dict."""
    ad_id = str(row.get("id") or "")
    raw_price = row.get("price_amount")
    if raw_price is None:
        return None
    try:
        price_amount = float(raw_price)
    except (TypeError, ValueError):
        return None
    if math.isnan(price_amount):
        return None
    return {
        "ad_id": ad_id,
        "title": row.get("title"),
        "url": row.get("url"),
        "location": row.get("location"),
        "price_amount": price_amount,
        "price_currency": row.get("price_currency"),
        "price_unit": row.get("price_unit"),
        "price_display": row.get("price_display"),
        "image_url": row.get("image_url"),
        "timestamp": row.get("timestamp"),
        "promoted": bool(row.get("promoted")),
    }


async def finn_search(
    query: str,
    price_from: float = 0.0,
    price_to: float = DEFAULT_PRICE_MAX,
    location: str = "",
    max_pages: int = DEFAULT_MAX_PAGES,
) -> str:
    """Search Finn.no listings and filter them by price range and location.

    Runs a Finn.no search for `query`, collects listings across up to
    `max_pages` result pages (price-range sharding is handled internally when
    the result set is large), and returns the listings that fall within
    [price_from, price_to] and, optionally, whose location matches `location`.

    Args:
        query: Search term (e.g. "gitar", "tromme", "sykkel").
        price_from: Minimum price in NOK (inclusive, default 0).
        price_to: Maximum price in NOK (inclusive, default 1000000).
        location: Optional location substring filter, e.g. "oslo" or "berg".
        max_pages: Maximum number of search result pages to scan.

    Returns:
        JSON string: {"query": ..., "count": ..., "listings": [...]}
        Each listing has ad_id, title, url, location, price_*, image_url, timestamp, promoted.
    """
    client = get_finn_client()
    df = client.search_dataframe_sharded(
        query=query,
        price_min=int(price_from),
        price_max=int(price_to),
        max_pages=max_pages,
    )

    listings: list[dict[str, Any]] = []
    if not df.empty:
        filter_price_from = float(price_from)
        filter_price_to = float(price_to)
        filter_location = (location or "").strip().lower()
        for _, row in df.iterrows():
            listing = _listing_to_dict(dict(row))
            if listing is None:
                continue
            if (
                listing["price_amount"] < filter_price_from
                or listing["price_amount"] > filter_price_to
            ):
                continue
            if (
                filter_location
                and filter_location not in str(listing.get("location") or "").lower()
            ):
                continue
            listings.append(listing)

    return json.dumps(
        {"query": query, "count": len(listings), "listings": listings},
        indent=2,
        default=str,
    )


async def finn_get_listing(ad_id: str) -> str:
    """Fetch full details for a single Finn.no listing by ad id.

    Returns everything the model knows about the listing: description, all
    image urls, price, location (postal code + coordinates), condition/extras,
    category hierarchy, transaction flags, timestamp and seller info.

    Args:
        ad_id: Finn.no ad id of the listing (e.g. "451778168").

    Returns:
        JSON string with the full listing detail.
    """
    client = get_finn_client()
    detail = client.get_item(ad_id)
    return json.dumps(detail, indent=2, default=str)


async def finn_get_image(ad_id: str) -> Image:
    """Fetch a Finn.no listing image and return it so it can be viewed.

    Args:
        ad_id: Finn.no ad id of the listing (e.g. "451778168").

    Returns:
        The listing's main image content, ready for the model to see.
    """
    client = get_finn_client()
    detail = client.get_item(ad_id)
    image_url = detail.get("image_url")
    if not image_url:
        raise ValueError(f"No image found for Finn.no listing {ad_id}")
    response = client.session.get(image_url, timeout=15)
    response.raise_for_status()
    image_format = (
        "jpeg"
        if str(image_url).lower().split("?", 1)[0].endswith((".jpg", ".jpeg"))
        else "png"
    )
    return Image(data=response.content, format=image_format)
