import asyncio
import json

import pytest

from finn_deals.mcp import tools
from finn_deals.mcp.deals import compare_listings, review_signals
from finn_deals.scraping.finn import FinnAPI, SearchResult


def listing(ad_id, price, **kwargs):
    return {
        "ad_id": str(ad_id),
        "title": "Example model",
        "price_amount": price,
        "price_currency": "NOK",
        "availability": "active",
        **kwargs,
    }


def test_comparison_excludes_self_and_requires_three_peers():
    result = compare_listings(
        [listing(1, 500), listing(2, 900), listing(3, 1000), listing(4, 1100)]
    )
    candidate = result["listings"][0]
    assert candidate["peer_ad_ids"] == ["2", "3", "4"]
    assert candidate["peer_median_asking_price_nok"] == 1000
    assert candidate["percent_below_peer_median"] == 50
    assert candidate["comparison_status"] == "unverified_comparability"
    sparse = compare_listings([listing(1, 500), listing(2, 900), listing(3, 1000)])
    assert sparse["listings"][0]["percent_below_peer_median"] is None


def test_comparison_excludes_duplicates_inactive_free_missing_and_wrong_currency():
    result = compare_listings(
        [
            listing(1, 500),
            listing(1, 500),
            listing(2, 1000),
            listing(3, 2000, availability="disposed"),
            listing(4, 0),
            listing(5, None),
            listing(6, 1500, price_currency="EUR"),
            listing(7, float("nan")),
            listing(8, 1000, availability="unknown"),
        ]
    )
    assert result["eligible_price_count"] == 2
    assert result["listings"][0]["peer_count"] == 1
    assert result["listings"][0]["percent_below_peer_median"] is None


def test_review_signals_quote_evidence_instead_of_asserting_defects():
    signals = review_signals(listing(1, 500, description="Utestet og mangler lader"))
    assert any("seller text:" in signal and "Utestet" in signal for signal in signals)
    assert any("included parts" in signal for signal in signals)


def test_placeholder_and_split_sale_prices_are_marked_for_review():
    signals = review_signals(
        listing(1, 1, description="Spill selges separat, se beskrivelse")
    )
    assert any("placeholder" in signal for signal in signals)
    assert any(
        "actual price" in signal and "seller text" in signal for signal in signals
    )


def test_condition_signals_include_context_for_ai_interpretation():
    signals = review_signals(
        listing(
            1, 250, description="Armlenene ble kastet. Har visuell skade på ryggen."
        )
    )
    assert any(
        "condition and completeness" in signal and "kastet" in signal
        for signal in signals
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " "},
        {"max_pages": 0},
        {"max_pages": 6},
        {"price_from": -1},
        {"price_from": 100, "price_to": 50},
        {"price_to": float("inf")},
        {"sort": "invalid"},
    ],
)
def test_bad_search_input_rejected_without_network(kwargs):
    with pytest.raises(ValueError):
        asyncio.run(tools.finn_search(**{"query": "item", **kwargs}))


class SearchClient(FinnAPI):
    def __init__(self):
        super().__init__()
        self.calls = []

    def search(self, query, page=1, **kwargs):
        self.calls.append((page, kwargs))
        docs = [
            {"id": 1, "heading": "iPhone 13", "price": {"amount": 900}},
            {"id": 2, "heading": "iPhone 13 Pro", "price": {"amount": 700}},
            {"id": 3, "heading": "iPhone 130", "price": {"amount": 500}},
            {"id": 4, "heading": "iPhone 13", "price": {"amount": 400}},
        ]
        return SearchResult(
            [self._build_listing(d) for d in docs], 5000, page, 50, False, {}, {}
        )

    def get_item(self, ad_id):
        if ad_id == "4":
            raise ValueError("Listing removed")
        return listing(
            ad_id,
            900,
            description="Works",
            image_urls=["https://images.finncdn.no/photo"],
        )


def test_broad_search_is_bounded_filters_variants_sorts_and_reports_coverage(
    monkeypatch,
):
    client = SearchClient()
    monkeypatch.setattr(tools, "_client", client)
    result = json.loads(
        asyncio.run(
            tools.finn_search(
                "iPhone",
                max_pages=1,
                required_terms=["13"],
                exclude_terms=["pro"],
                sort="PRICE_ASC",
            )
        )
    )
    assert len(client.calls) == 1
    assert client.calls[0][1]["trade_type"] == 1
    assert [row["ad_id"] for row in result["listings"]] == ["4", "1"]
    assert result["truncated"] is True
    assert result["pages_fetched"] == 1
    assert result["total_matches"] == 5000


def test_shortlist_preserves_partial_detail_errors(monkeypatch):
    monkeypatch.setattr(tools, "_client", SearchClient())
    result = json.loads(
        asyncio.run(
            tools.finn_find_deals(
                "iPhone 13",
                max_pages=1,
                limit=2,
                required_terms=["13"],
                exclude_terms=["Pro"],
            )
        )
    )
    assert [row["detail_status"] for row in result["candidates"]] == ["error", "ok"]
    assert result["candidate_count"] == 2
    assert "not confirmed deals" in result["assessment"]


def test_compare_tool_deduplicates_and_reports_failed_fetches(monkeypatch):
    monkeypatch.setattr(tools, "_client", SearchClient())
    result = json.loads(asyncio.run(tools.finn_compare_listings(["1", "1", "4"])))
    assert len(result["listings"]) == 1
    assert result["fetch_errors"] == [{"ad_id": "4", "error": "Listing removed"}]


def test_image_signature_overrides_extensionless_url(monkeypatch):
    client = SearchClient()

    class Response:
        content = b"\xff\xd8\xffsample"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(tools, "_client", client)
    image = asyncio.run(tools.finn_get_image("1"))
    assert image.to_image_content().mimeType == "image/jpeg"
    with pytest.raises(ValueError, match="unavailable"):
        asyncio.run(tools.finn_get_image("1", image_index=1))
