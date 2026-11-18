import asyncio
import json

import pytest

from finn_deals.mcp import tools
from finn_deals.scraping.finn import FinnAPI, SearchResult


def facet(key, value, label, hits, *children):
    return dict(
        name=key, value=value, display_name=label, hits=hits, filter_items=children
    )


class FacetClient(FinnAPI):
    def __init__(self):
        self.calls = []

    def search(self, query, page=1, **params):
        self.calls.append((query, params))
        category = facet(
            "category",
            "0.93",
            "Elektronikk",
            7,
            facet("product_category", "2.93.3906.53", "Høyttalere", 5),
            facet("product_category", "2.93.3906.61", "Radio", 0),
        )
        selected = [
            {
                "parameters": [
                    {
                        "parameter_name": "product_category",
                        "parameter_value": "2.93.3906.53",
                    }
                ]
            }
        ]
        listing = self._build_listing(
            {"id": 1, "heading": "KEF", "price": {"amount": 5000}}
        )
        return SearchResult(
            [listing],
            1,
            page,
            1,
            True,
            {"selected_filters": selected},
            {"filters": [{"name": "category", "filter_items": [category]}]},
        )

    def get_item(self, ad_id):
        return {"ad_id": ad_id, "title": "KEF"}


@pytest.mark.parametrize("tool", [tools.finn_search, tools.finn_find_deals])
def test_discovered_nested_category_scopes_keywordless_search(monkeypatch, tool):
    client = FacetClient()
    monkeypatch.setattr(tools, "_client", client)
    result = json.loads(
        asyncio.run(tools.finn_discover_filters("category", text="HØYTTALERE"))
    )
    (option,) = result["options"]
    assert option == {
        "path": ["Elektronikk", "Høyttalere"],
        "label": "Høyttalere",
        "hits": 5,
        "finn_filters": {"product_category": "2.93.3906.53"},
    }
    payload = json.loads(
        asyncio.run(tool(max_pages=1, finn_filters=option["finn_filters"]))
    )
    query, params = client.calls[1]
    assert (
        query == ""
        and params["product_category"] == "2.93.3906.53"
        and params["trade_type"] == 1
    )
    assert payload["count"] == 1 and len(client.calls) == 2


def test_discovery_handles_parent_matches_zero_hits_and_limit(monkeypatch):
    monkeypatch.setattr(tools, "_client", FacetClient())
    result = json.loads(
        asyncio.run(
            tools.finn_discover_filters("category", text="Elektronikk", limit=1)
        )
    )
    assert result["matching_options"] == 3 and result["truncated"]
    assert [o["label"] for o in result["options"]] == ["Elektronikk"]
    result = json.loads(asyncio.run(tools.finn_discover_filters("category")))
    assert result["matching_options"] == 2 and not result["truncated"]


@pytest.mark.parametrize(
    "tool", [tools.finn_search, tools.finn_find_deals, tools.finn_discover_filters]
)
def test_ignored_native_filter_never_returns_broad_results(monkeypatch, tool):
    client = FacetClient()
    monkeypatch.setattr(tools, "_client", client)
    kwargs = {"group": "category"} if tool == tools.finn_discover_filters else {}
    with pytest.raises(ValueError, match="did not confirm location"):
        asyncio.run(tool(query="KEF", finn_filters={"location": "0.99999"}, **kwargs))


def test_filter_parameters_cannot_override_page_or_trade_type(monkeypatch):
    client = FacetClient()
    monkeypatch.setattr(tools, "_client", client)
    with pytest.raises(ValueError, match="Invalid FINN filter"):
        asyncio.run(tools.finn_search("KEF", finn_filters={"trade_type": "2"}))
    assert not client.calls
