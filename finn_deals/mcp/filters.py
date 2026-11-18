"""Discover FINN's own Torget filter values and verify their use in searches."""

from collections.abc import Iterator
from typing import Any, Literal, get_args

from finn_deals.scraping.finn import SearchResult

FilterGroup = Literal[
    "location", "category", "condition", "shipping_types", "brand", "dealer_segment"
]
FilterKey = Literal[
    "location",
    "category",
    "sub_category",
    "product_category",
    "condition",
    "shipping_types",
    "brand",
    "dealer_segment",
]
FinnFilters = dict[FilterKey, str]


def validate_filters(filters: FinnFilters) -> None:
    for key, value in filters.items():
        if key not in get_args(FilterKey) or not value.strip():
            raise ValueError(
                f"Invalid FINN filter {key!r}. Use finn_discover_filters to find valid options."
            )


def verify_selected(result: SearchResult, filters: FinnFilters) -> None:
    """Do not silently report nationwide or cross-category results for an ignored filter."""
    if not filters:
        return
    selected = {
        (p.get("parameter_name"), str(p.get("parameter_value")))
        for group in result.metadata.get("selected_filters") or []
        for p in group.get("parameters") or []
    }
    for key, value in filters.items():
        if (key, value) not in selected:
            raise ValueError(
                f"FINN did not confirm {key}={value}. Use finn_discover_filters for current options."
            )


def _walk_options(
    items: list[dict[str, Any]], parents: tuple[str, ...] = ()
) -> Iterator[dict[str, Any]]:
    for item in items:
        label = item.get("display_name")
        if not label:
            continue
        path = (*parents, label)
        key, value, hits = item.get("name"), item.get("value"), item.get("hits")
        if (
            key in get_args(FilterKey)
            and value is not None
            and isinstance(hits, int)
            and hits >= 0
        ):
            yield {
                "label": label,
                "path": list(path),
                "hits": hits,
                "finn_filters": {key: str(value)},
            }
        yield from _walk_options(item.get("filter_items") or [], path)


def discover_options(
    result: SearchResult, group: FilterGroup, text: str, limit: int
) -> dict[str, Any]:
    facets = result.raw.get("filters")
    if not isinstance(facets, list):
        raise ValueError("FINN did not return filter metadata for this search")
    group_data = next((f for f in facets if f.get("name") == group), {})
    term = text.strip().casefold()
    options = [
        option
        for option in _walk_options(group_data.get("filter_items") or [])
        if (term and any(term in label.casefold() for label in option["path"]))
        or (not term and option["hits"] > 0)
    ]
    options.sort(key=lambda o: (o["label"].casefold() != term, -o["hits"], o["path"]))
    return {
        "group": group,
        "total_matches": result.total_matches,
        "matching_options": len(options),
        "options": options[:limit],
        "truncated": len(options) > limit,
    }
