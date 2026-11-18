import base64
import json

import pytest

from finn_deals.mcp.client import FinnListingClient, ListingDetails


def item_section():
    # Representative current FINN shape, stripped of personal data.
    return {
        "itemData": {
            "title": 'Stol "Modell 2" – blå',
            "price": 1200,
            "description": 'Pent brukt.\nInkluderer pute, ikke bord. "Som den er".',
            "category": {
                "id": 1,
                "value": "Stoler",
                "parent": {"id": 2, "value": "Møbler"},
            },
            "location": {
                "postalCode": "4006",
                "postalName": "Stavanger",
                "position": {"lat": 58.97, "lng": 5.73},
            },
            "images": [
                {"uri": "https://images.finncdn.no/dynamic/default/item/123/abc"}
            ],
            "extras": [{"id": "condition", "value": "Pent brukt", "valueId": 3}],
            "meta": {"isInactive": False, "edited": "2026-10-24T18:43:56+02:00"},
            "disposed": False,
        },
        "jsonLd": {"offers": {"priceCurrency": "NOK"}},
        "meta": {"canonical": "https://www.finn.no/recommerce/forsale/item/123"},
        "transactableData": {"buyNow": True},
        "transactableUiData": {
            "sections": {
                "sidebar": {
                    "optedIn": {
                        "shippingPrice": {
                            "text": "Frakt fra 79 kr + Trygg betaling 42 kr"
                        }
                    }
                }
            }
        },
    }


def test_current_hydration_parsing_handles_escaped_json_and_later_branches():
    payload = {"loaderData": {"root": {}, "item": item_section()}}
    encoded = json.dumps(json.dumps(payload, ensure_ascii=False))
    html = f'<script>window.__staticRouterHydrationData = JSON.parse({encoded}); window.other = "ignore";</script>'
    raw = FinnListingClient()._parse_item_page(html)
    detail = ListingDetails.from_raw("123", raw).to_dict()
    assert detail["title"] == 'Stol "Modell 2" – blå'
    assert detail["price_amount"] == 1200
    assert detail["price_currency"] == "NOK"
    assert detail["price_display"] == "1200 NOK"
    assert detail["location"] == "Stavanger"
    assert detail["latitude"] == 58.97
    assert detail["extras"]["condition"] == "Pent brukt"
    assert detail["num_images"] == 1
    assert detail["seller"] is None
    assert detail["availability"] == "active"
    assert detail["timestamp"].startswith("2026-10-24T18:43:56")
    assert detail["shipping_payment_info"].startswith("Frakt fra 79")


def test_legacy_payload_and_price_dict_remain_supported():
    section = item_section()
    section["itemData"]["price"] = {"amount": 1500, "currency_code": "NOK"}
    section["itemData"]["heading"] = section["itemData"].pop("title")
    payload = {"queries": [{"state": {"data": section}}]}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    raw = FinnListingClient()._parse_item_page(f"<script>{encoded}</script>")
    assert ListingDetails.from_raw("123", raw).price_amount == 1500


def test_missing_item_payload_raises_instead_of_empty_success():
    encoded = json.dumps(json.dumps({"loaderData": {"root": {}}}))
    with pytest.raises(ValueError, match="listing data"):
        FinnListingClient()._parse_item_page(
            f"window.__staticRouterHydrationData = JSON.parse({encoded})"
        )


@pytest.mark.parametrize(
    "ad_id", ["../search", "https://example.org", "", "123?test=1"]
)
def test_invalid_ad_id_rejected_before_network(ad_id):
    with pytest.raises(ValueError, match="numeric"):
        FinnListingClient().get_item(ad_id)


def test_status_survives_parsing():
    section = item_section()
    section["itemData"]["disposed"] = True
    assert ListingDetails.from_raw("123", section).availability == "disposed"
    section["itemData"]["disposed"] = False
    section["itemData"]["meta"]["isInactive"] = True
    assert ListingDetails.from_raw("123", section).availability == "inactive"
