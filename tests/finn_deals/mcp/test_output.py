import asyncio
import json

import pytest

from finn_deals.mcp import tools
from tests.finn_deals.mcp.test_tools import FakeClient, _listing_df


@pytest.fixture
def client(monkeypatch):
    df = _listing_df().iloc[[0] * 12].copy()
    df["id"] = [str(i) for i in range(12)]
    detail = {
        "description": "Pent brukt. " * 100,
        "image_urls": ["photo"] * 30,
        "availability": "active",
        "extras": {"condition": "Brukt", "condition_id": 3},
        "price_amount": 100,
        "price_currency": "NOK",
        "num_images": 30,
    }

    class Client(FakeClient):
        def get_item(self, ad_id):
            return {**detail, "ad_id": ad_id}

    fake = Client(df)
    monkeypatch.setattr(tools, "_client", fake)
    return fake


def read(tool, **args):
    return json.loads(asyncio.run(tool(**args)))


def test_compact_search_window_preserves_coverage_and_order(client):
    first = read(tools.finn_search, query="gitar")
    second = read(tools.finn_search, query="gitar", offset=first["next_offset"])
    full = read(tools.finn_search, query="gitar", view="full")
    assert first["count"] == 12 and first["returned_count"] == 10
    assert not first["truncated"] and first["next_offset"] == 10
    assert second["returned_count"] == 2 and second["next_offset"] is None
    assert [r["ad_id"] for r in first["listings"] + second["listings"]] == [
        r["ad_id"] for r in full["listings"]
    ]
    assert (
        "image_url" not in first["listings"][0] and "image_url" in full["listings"][0]
    )
    assert read(tools.finn_search, query="gitar", limit=1)["returned_count"] == 1


@pytest.mark.parametrize("args", [{"limit": 0}, {"offset": -1}])
def test_invalid_window_rejected_before_search(args):
    with pytest.raises(ValueError, match="offset"):
        read(tools.finn_search, query="gitar", **args)


def test_compact_listing_and_shortlist_allow_targeted_expansion(client):
    short = read(tools.finn_find_deals, query="gitar", limit=1)
    (candidate,) = short["candidates"]
    assert "search_sample" not in short and "next_steps" not in short
    assert candidate["detail_status"] == "ok" and candidate["review_signals"]
    ad_id = candidate["ad_id"]
    detail = read(tools.finn_get_listing, ad_id=ad_id)
    for row in [detail, candidate]:
        assert len(row["description"]) == 400 and row["description_truncated"]
        assert row["extras"] == {"condition": "Brukt"}
        assert "image_urls" not in row and row["num_images"] == 30
    full = read(tools.finn_get_listing, ad_id=ad_id, view="full")
    assert full == client.get_item(ad_id)
    assert (
        len(
            read(tools.finn_find_deals, query="gitar", limit=1, view="full")[
                "search_sample"
            ]
        )
        == 12
    )


def test_comparison_views_have_identical_price_evidence(client):
    args = {"ad_ids": ["1", "2", "3", "4"]}
    compact = read(tools.finn_compare_listings, **args)
    full = read(tools.finn_compare_listings, **args, view="full")
    for summary, detailed in zip(compact["listings"], full["listings"], strict=True):
        for key in [
            "ad_id",
            "peer_ad_ids",
            "price_evidence_eligible",
            "comparison_status",
            "peer_median_asking_price_nok",
            "percent_below_peer_median",
            "review_signals",
        ]:
            assert summary[key] == detailed[key]
        assert "image_urls" not in summary and detailed["image_urls"]
