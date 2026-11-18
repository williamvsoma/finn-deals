"""Compact views of research evidence for the MCP text channel."""

from typing import Any, Literal

View = Literal["compact", "full"]
DESCRIPTION_LIMIT = 400
DEFAULT_SEARCH_LIMIT = 10


def compact_listing(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: row[key]
        for key in (
            "ad_id",
            "title",
            "url",
            "price_amount",
            "price_currency",
            "location",
            "timestamp",
            "availability",
            "shipping_payment_info",
            "num_images",
            "detail_status",
            "detail_error",
            "review_signals",
        )
        if row.get(key) is not None
    }
    if row.get("promoted"):
        result["promoted"] = True
    extras = {
        k: v for k, v in (row.get("extras") or {}).items() if not k.endswith("_id")
    }
    if extras:
        result["extras"] = extras
    if description := row.get("description"):
        result["description"] = description[:DESCRIPTION_LIMIT]
        if len(description) > DESCRIPTION_LIMIT:
            result["description_truncated"] = True
    return result


def search_window(
    search: dict[str, Any], view: View, limit: int | None, offset: int
) -> dict[str, Any]:
    listings = search["listings"]
    if view == "compact" and limit is None:
        limit = DEFAULT_SEARCH_LIMIT
    stop = offset + limit if limit is not None else None
    window = listings[offset:stop]
    end = offset + len(window)
    result = {
        **search,
        "listings": window,
        "returned_count": len(window),
        "next_offset": end if end < len(listings) else None,
    }
    if view == "compact":
        result.pop("coverage_note")
        result["listings"] = [compact_listing(row) for row in window]
    return result


def compact_shortlist(shortlist: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            k: v
            for k, v in shortlist.items()
            if k not in ("candidates", "coverage_note", "next_steps", "search_sample")
        },
        "candidates": [compact_listing(row) for row in shortlist["candidates"]],
    }


def compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    evidence_fields = (
        "peer_ad_ids",
        "peer_count",
        "peer_median_asking_price_nok",
        "percent_below_peer_median",
        "price_evidence_eligible",
        "comparison_status",
    )
    return {
        **{k: v for k, v in comparison.items() if k not in ("listings", "limitations")},
        "listings": [
            {**compact_listing(row), **{key: row[key] for key in evidence_fields}}
            for row in comparison["listings"]
        ],
    }
