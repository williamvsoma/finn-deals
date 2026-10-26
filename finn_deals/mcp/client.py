"""Finn.no listing client with full item-detail support.

Wraps :class:`finn_deals.scraping.finn.FinnAPI` and adds full listing-detail
fetching through FINN's item pages so that MCP tools can expose everything
about a listing: description, all images, precise location, condition/extras,
category hierarchy and seller information.
"""

import json
import logging
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

from finn_deals.scraping.finn import FinnAPI

logger = logging.getLogger(__name__)


@dataclass
class SellerProfile:
    """Seller / company information for retailer listings."""

    org_name: str | None = None
    org_id: str | None = None
    logo_url: str | None = None
    homepage_url: str | None = None
    more_ads_url: str | None = None
    contact_name: str | None = None
    contact_phone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the seller profile."""
        return {
            "org_name": self.org_name,
            "org_id": self.org_id,
            "logo_url": self.logo_url,
            "homepage_url": self.homepage_url,
            "more_ads_url": self.more_ads_url,
            "contact_name": self.contact_name,
            "contact_phone": self.contact_phone,
        }


@dataclass
class ListingDetails:
    """Everything we know about a single Finn.no listing."""

    ad_id: str = ""
    title: str | None = None
    url: str | None = None
    price_amount: int | None = None
    price_currency: str | None = None
    price_unit: str | None = None
    price_display: str | None = None
    description: str | None = None
    image_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    num_images: int = 0
    location: str | None = None
    postal_code: str | None = None
    postal_name: str | None = None
    country_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    coord_accuracy: float | None = None
    timestamp: datetime | None = None
    category: str | None = None
    category_id: int | None = None
    sub_category: str | None = None
    sub_category_id: int | None = None
    top_category: str | None = None
    top_category_id: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    transactable: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    follower_count: int | None = None
    seller: SellerProfile | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls, ad_id: str, raw: dict[str, Any], base_url: str = "https://www.finn.no"
    ) -> "ListingDetails":
        """Flatten a raw item-page payload into a ListingDetails."""
        item = dict(raw.get("itemData") or {})
        transactable = dict(raw.get("transactableData") or {})
        meta_page = dict(raw.get("meta") or {})

        price = dict(item.get("price") or {})
        price_amount = price.get("amount")
        price_unit = price.get("price_unit") or price.get("currency_code")

        extras: dict[str, Any] = {}
        for extra in item.pop("extras", []) or []:
            if not isinstance(extra, dict):
                continue
            extra_id = extra.get("id") or ""
            if extra_id:
                extras[extra_id] = extra.get("value")
                if extra.get("valueId") is not None:
                    extras[f"{extra_id}_id"] = extra.get("valueId")

        category = dict(item.pop("category") or {})
        parent = dict(category.get("parent") or {})
        grandparent = dict(parent.get("parent") or {})

        image_urls: list[str] = []
        for image in item.pop("images", []) or []:
            if isinstance(image, dict) and image.get("uri"):
                image_urls.append(image["uri"])

        location = dict(item.pop("location") or {})
        position = dict(location.get("position") or {})

        meta: dict[str, Any] = dict(item.pop("meta") or {})

        txn = dict(transactable)

        follower_count = raw.get("follower_count")
        if not isinstance(follower_count, int):
            follower_count = None
        company_raw = raw.get("companyProfile")
        company = company_raw if isinstance(company_raw, dict) else {}
        seller = SellerProfile(
            org_name=company.get("orgName"),
            org_id=company.get("orgId"),
            logo_url=company.get("logo"),
            homepage_url=company.get("homepageUrl"),
            more_ads_url=company.get("moreAdsFromThisCompanyUrl"),
        )
        contacts = company.get("contacts")
        if isinstance(contacts, list) and contacts:
            contact = contacts[0]
            seller.contact_name = (
                contact.get("name") if isinstance(contact, dict) else None
            )
            seller.contact_phone = (
                contact.get("phone") if isinstance(contact, dict) else None
            )

        url = meta_page.get("canonical") or f"/item/{ad_id}"
        url_str = str(url)
        if not url_str.startswith("http"):
            url_str = f"{base_url.rstrip('/')}/{url_str.lstrip('/')}"

        return cls(
            ad_id=str(ad_id),
            title=item.get("heading"),
            url=url_str,
            price_amount=price_amount,
            price_currency=price.get("currency_code"),
            price_unit=price_unit,
            price_display=(
                f"{price_amount} {price_unit}".strip()
                if price_amount is not None
                else None
            ),
            description=item.get("description"),
            image_url=image_urls[0] if image_urls else None,
            image_urls=image_urls,
            num_images=len(image_urls),
            location=location.get("location"),
            postal_code=location.get("postalCode"),
            postal_name=location.get("postalName"),
            country_code=location.get("countryCode"),
            latitude=position.get("lat"),
            longitude=position.get("lng"),
            coord_accuracy=position.get("accuracy"),
            timestamp=cls._parse_timestamp(item.get("timestamp")),
            category=category.get("value"),
            category_id=category.get("id"),
            sub_category=parent.get("value"),
            sub_category_id=parent.get("id"),
            top_category=grandparent.get("value"),
            top_category_id=grandparent.get("id"),
            extras=extras,
            transactable=txn,
            meta=meta,
            follower_count=follower_count,
            seller=seller,
            raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the listing details."""
        return {
            "ad_id": self.ad_id,
            "title": self.title,
            "url": self.url,
            "price_amount": self.price_amount,
            "price_currency": self.price_currency,
            "price_unit": self.price_unit,
            "price_display": self.price_display,
            "description": self.description,
            "image_url": self.image_url,
            "image_urls": self.image_urls,
            "num_images": self.num_images,
            "location": self.location,
            "postal_code": self.postal_code,
            "postal_name": self.postal_name,
            "country_code": self.country_code,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coord_accuracy": self.coord_accuracy,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "category": self.category,
            "category_id": self.category_id,
            "sub_category": self.sub_category,
            "sub_category_id": self.sub_category_id,
            "top_category": self.top_category,
            "top_category_id": self.top_category_id,
            "extras": self.extras,
            "transactable": self.transactable,
            "meta": self.meta,
            "follower_count": self.follower_count,
            "seller": self.seller.to_dict() if self.seller is not None else None,
        }

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
        except (TypeError, ValueError, OSError):
            return None


class FinnListingClient(FinnAPI):
    """Finn.no listing search + full item-detail client."""

    item_path = "/recommerce/forsale/item"

    _HYDRATION_RE = re.compile(
        r'window\.__staticRouterHydrationData\s*=\s*JSON\.parse\("(.*)"\)',
        re.DOTALL,
    )

    _FOLLOWER_RE = re.compile(r"(\d+)\s+følger")

    _COMPANY_PROFILE_RE = re.compile(
        r"<companyProfilePodlet-isolated[^>]*>(.*?)</companyProfilePodlet-isolated>",
        re.DOTALL,
    )

    def __init__(self, session: requests.Session | None = None) -> None:
        super().__init__(session=session)
        self.session.headers.setdefault("User-Agent", "finn-mcp-server/0.1")

    # ------------------------------------------------------------------
    # Item-page fetching
    # ------------------------------------------------------------------

    def _get_with_retries(
        self, url: str, params: dict[str, Any], attempts: int = 3
    ) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Failed to fetch {url} after {attempts} attempts"
        ) from last_exc

    def _item_page(self, ad_id: str) -> dict[str, Any]:
        response = self._get_with_retries(
            f"{self.base_url}{self.item_path}/{ad_id}",
            params={},
        )
        return self._parse_item_page(response.text)

    def get_item(self, ad_id: str) -> dict[str, Any]:
        """Fetch a single listing by ad-id and return a full detail dict."""
        raw = self._item_page(str(ad_id))
        details = ListingDetails.from_raw(str(ad_id), raw, base_url=self.base_url)
        return details.to_dict()

    def get_items(self, ad_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Fetch details for many listings concurrently, keyed by ad_id."""
        ad_ids = [str(aid) for aid in ad_ids]
        details: dict[str, dict[str, Any]] = {}
        if not ad_ids:
            return details
        workers = max(1, min(8, len(ad_ids)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.get_item, aid): aid for aid in ad_ids}
            for future in as_completed(futures):
                ad_id = futures[future]
                try:
                    details[ad_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to fetch item %s: %s", ad_id, exc)
        return details

    def get_image(self, ad_id: str) -> bytes:
        """Fetch the raw bytes of a listing's main image."""
        detail = self.get_item(ad_id)
        image_url = detail.get("image_url")
        if not image_url:
            raise ValueError(f"No image found for Finn.no listing {ad_id}")
        resp = self.session.get(image_url, timeout=15)
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Item-page parsing helpers
    # ------------------------------------------------------------------

    def _parse_item_page(self, html: str) -> dict[str, Any]:
        payload = self._extract_dehydrated_payload(html)
        section = self._find_item_section(payload)
        result: dict[str, Any] = {
            "itemData": section.get("itemData") or {},
            "transactableData": section.get("transactableData") or {},
            "profileData": section.get("profileData") or {},
            "meta": section.get("meta") or {},
        }
        fm = self._FOLLOWER_RE.search(html)
        if fm:
            result["follower_count"] = int(fm.group(1))
        result["companyProfile"] = self._extract_company_profile(html)
        return result

    def _find_item_section(self, node: Any) -> dict[str, Any]:
        """Recursively locate the item section inside a dehydrated payload."""
        if isinstance(node, dict):
            if isinstance(node.get("itemData"), dict):
                return node
            for value in node.values():
                found = self._find_item_section(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = self._find_item_section(value)
                if found is not None:
                    return found
        return {}

    def _extract_company_profile(self, html: str) -> dict[str, Any] | None:
        m = self._COMPANY_PROFILE_RE.search(html)
        if not m:
            return None
        inner = m.group(1)
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", inner, re.DOTALL)
        for script_body in scripts:
            script_body = script_body.strip()
            if not script_body.startswith("{"):
                continue
            try:
                profile_json = json.loads(script_body)
            except json.JSONDecodeError:
                continue
            profile = profile_json.get("extendedProfileRecommerce")
            if isinstance(profile, dict):
                return profile
        return None
