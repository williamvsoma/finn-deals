import asyncio
import json
from typing import Any

import pandas as pd
import pytest
from mcp.server.fastmcp.utilities.types import Image

from finn_deals.mcp import tools
from finn_deals.mcp.tools import finn_get_image, finn_get_listing, finn_search


class FakeResponse:
    def __init__(self, content: bytes = b"fakejpegdata") -> None:
        self.content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.calls: list = []

    def get(
        self, url: str, params: Any = None, timeout: float | None = None
    ) -> FakeResponse:
        self.calls.append((url, params, timeout))
        return FakeResponse(b"fakejpegdata")


class FakeClient:
    """Minimal stand-in for FinnListingClient used by the MCP tools."""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.df = df if df is not None else pd.DataFrame()
        self.detail = detail if detail is not None else {}
        self.session = FakeSession()

    def search_dataframe_sharded(
        self,
        query: str,
        price_min: float,
        price_max: float,
        max_pages: int = 2,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.df

    def get_item(self, ad_id: str) -> dict[str, Any]:
        return self.detail


def _listing_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "1",
                "title": "Gitar til salgs",
                "url": "https://www.finn.no/item/1",
                "location": "Oslo",
                "price_amount": 100.0,
                "price_currency": "NOK",
                "price_unit": None,
                "price_display": "100 NOK",
                "image_url": "https://cdn.finn.no/img/guitar.jpg",
                "timestamp": None,
                "promoted": True,
            },
            {
                "id": "2",
                "title": "Gitar til salgs",
                "url": "https://www.finn.no/item/2",
                "location": "Oslo",
                "price_amount": 200.0,
                "price_currency": "NOK",
                "price_unit": None,
                "price_display": "200 NOK",
                "image_url": "https://cdn.finn.no/img/guitar2.jpg",
                "timestamp": None,
                "promoted": False,
            },
            {
                "id": "3",
                "title": "Gitar til salgs",
                "url": "https://www.finn.no/item/3",
                "location": "Bergen",
                "price_amount": 150.0,
                "price_currency": "NOK",
                "price_unit": None,
                "price_display": "150 NOK",
                "image_url": "https://cdn.finn.no/img/guitar3.jpg",
                "timestamp": None,
                "promoted": False,
            },
        ]
    )


def test_finn_search_returns_filtered_listings_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "_client", FakeClient(df=_listing_df()))

    result = asyncio.run(
        finn_search(query="gitar", price_from=0.0, price_to=1000.0, location="oslo")
    )

    payload = json.loads(result)
    assert payload["query"] == "gitar"
    assert payload["count"] == 2
    assert {listing["ad_id"] for listing in payload["listings"]} == {"1", "2"}


def test_finn_search_returns_empty_listings_when_nothing_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools, "_client", FakeClient(df=_listing_df()))

    result = asyncio.run(finn_search(query="gitar", price_from=5000.0, price_to=6000.0))

    payload = json.loads(result)
    assert payload["count"] == 0
    assert payload["listings"] == []


def test_finn_get_listing_returns_full_detail_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail = {
        "ad_id": "451778168",
        "title": "Gitar til salgs",
        "url": "https://www.finn.no/item/451778168",
        "price_amount": 1500,
        "price_currency": "NOK",
        "price_unit": None,
        "price_display": "1500 NOK",
        "description": "God gammeldags gitar",
        "image_url": "https://cdn.finn.no/img/guitar.jpg",
        "image_urls": ["https://cdn.finn.no/img/guitar.jpg"],
        "num_images": 1,
        "location": "Oslo",
        "postal_code": "0180",
        "postal_name": "Oslo",
        "country_code": "NO",
        "latitude": 59.91,
        "longitude": 10.75,
        "coord_accuracy": 4.0,
        "timestamp": None,
        "category": "Musikk",
        "category_id": 4400,
        "sub_category": "Gitar",
        "sub_category_id": 4410,
        "top_category": "Musikk",
        "top_category_id": 4000,
        "extras": {"condition": "Bra"},
        "transactable": {"buyNow": True},
        "meta": {},
        "follower_count": None,
        "seller": None,
    }
    monkeypatch.setattr(tools, "_client", FakeClient(detail=detail))

    result = asyncio.run(finn_get_listing(ad_id="451778168"))

    parsed = json.loads(result)
    assert parsed["ad_id"] == "451778168"
    assert parsed["description"] == "God gammeldags gitar"
    assert parsed["image_urls"] == ["https://cdn.finn.no/img/guitar.jpg"]
    assert parsed["latitude"] == 59.91
    assert parsed["longitude"] == 10.75
    assert parsed["postal_code"] == "0180"


def test_finn_get_image_returns_image_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tools,
        "_client",
        FakeClient(detail={"image_url": "https://cdn.finn.no/img/guitar.jpg"}),
    )

    image = asyncio.run(finn_get_image(ad_id="451778168"))

    assert isinstance(image, Image)
    assert image.data == b"fakejpegdata"
    image_content = image.to_image_content()
    assert image_content.mimeType == "image/jpeg"
    assert image_content.data == "ZmFrZWpwZWdkYXRh"


def test_get_finn_client_returns_shared_client_instance() -> None:
    assert tools.get_finn_client() is tools.get_finn_client()
