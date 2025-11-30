import base64
import json
from datetime import datetime, timezone

import pandas as pd

from finn_deals.scraping.finn import FinnAPI, Listing, SearchResult


def _make_payload_html(doc: dict, metadata: dict) -> str:
    payload = {"queries": [{"state": {"data": {"docs": [doc], "metadata": metadata}}}]}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"<html><head></head><body><script>{encoded}</script></body></html>"


def test_parse_search_response_parses_listing_fields():
    doc = {
        "id": 123,
        "heading": "Test Item",
        "canonical_url": "/item/123",
        "location": "Oslo",
        "price": {"amount": 100, "currency_code": "NOK", "price_unit": None},
        "image_urls": ["https://img.test/1"],
        "timestamp": 1_700_000_000_000,
    }
    metadata = {"result_size": {"match_count": 1}, "paging": {"current": 1, "last": 1}, "is_end_of_paging": True}
    html = _make_payload_html(doc, metadata)

    api = FinnAPI()
    result = api._parse_search_response(html)

    assert result.total_matches == 1
    assert result.is_end is True
    assert result.page == 1
    listing = result.listings[0]
    assert listing.id == "123"
    assert listing.title == "Test Item"
    assert listing.url.endswith("/item/123")
    assert listing.location == "Oslo"
    assert listing.price_display == "100 NOK"
    assert listing.image_url == "https://img.test/1"
    assert listing.timestamp == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_iter_search_respects_is_end(monkeypatch):
    calls = []

    def fake_search(query, page, **_):
        calls.append(page)
        listings = [{"id": page}] if page == 1 else []
        return SearchResult(
            listings=listings,
            total_matches=0,
            page=page,
            last_page=2,
            is_end=page >= 2,
            metadata={},
            raw={},
        )

    api = FinnAPI()
    monkeypatch.setattr(api, "search", fake_search)

    pages = list(api.iter_search("anything", max_pages=3))

    assert calls == [1, 2]
    assert len(pages) == 2
    assert pages[-1].is_end is True


def test_search_dataframe_concatenates_results(monkeypatch):
    api = FinnAPI()

    class DummyResult:
        def __init__(self, n):
            self.n = n

        def to_dataframe(self, include_raw: bool = False):
            return pd.DataFrame({"id": [self.n], "value": [self.n * 10]})

    monkeypatch.setattr(api, "iter_search", lambda *args, **kwargs: [DummyResult(1), DummyResult(2)])

    df = api.search_dataframe("q", max_pages=2)

    assert len(df) == 2
    assert set(df["id"]) == {1, 2}


def test_listing_price_display_handles_missing_price():
    listing = Listing(
        id="x",
        title="No price",
        url="https://example.com",
        location=None,
        price_amount=None,
        price_currency=None,
        price_unit=None,
        image_url=None,
        timestamp=None,
        raw={},
    )

    assert listing.price_display == "N/A"


def test_absolute_url_handles_relative_and_full():
    api = FinnAPI()
    assert api._absolute_url("/path/to/item") == "https://www.finn.no/path/to/item"
    full = "https://cdn.finn.no/img.jpg"
    assert api._absolute_url(full) == full
