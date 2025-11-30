import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests


@dataclass
class Listing:
    id: str
    title: str
    url: str
    location: Optional[str]
    price_amount: Optional[int]
    price_currency: Optional[str]
    price_unit: Optional[str]
    image_url: Optional[str]
    timestamp: Optional[datetime]
    raw: Dict[str, Any]

    @property
    def price_display(self) -> str:
        if self.price_amount is None:
            return "N/A"
        unit = self.price_unit or self.price_currency
        return f"{self.price_amount} {unit}".strip() if unit else str(self.price_amount)


@dataclass
class SearchResult:
    listings: List[Listing]
    total_matches: int
    page: int
    last_page: Optional[int]
    is_end: bool
    metadata: Dict[str, Any]
    raw: Dict[str, Any]

    def to_dataframe(self, include_raw: bool = False):
        """
        Convert listings to a pandas DataFrame.
        Set include_raw=True to also include the raw listing payload.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Install pandas to use to_dataframe(), e.g. pip install pandas"
            ) from exc

        rows = []
        for listing in self.listings:
            row = {
                "id": listing.id,
                "title": listing.title,
                "url": listing.url,
                "location": listing.location,
                "price_amount": listing.price_amount,
                "price_currency": listing.price_currency,
                "price_unit": listing.price_unit,
                "price_display": listing.price_display,
                "image_url": listing.image_url,
                "timestamp": listing.timestamp.isoformat() if listing.timestamp else None,
            }
            if include_raw:
                row["raw"] = listing.raw
            rows.append(row)

        return pd.DataFrame(rows)


class FinnAPI:
    """
    Scraper wrapper around FINN Torget search.
    """

    base_url = "https://www.finn.no"
    search_path = "/recommerce/forsale/search"

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "finn-scraper/0.1")

    def search(self, query: str, page: int = 1, **params: Any) -> SearchResult:
        """
        Run a search against FINN Torget and return parsed listings.
        Accepts the same params as the web UI (e.g. page, q, price_from, price_to).
        """
        query_params = {"q": query, "page": page}
        query_params.update({k: v for k, v in params.items() if v is not None})

        resp = self.session.get(
            f"{self.base_url}{self.search_path}", params=query_params, timeout=15
        )
        resp.raise_for_status()
        return self._parse_search_response(resp.text)

    def iter_search(
        self, query: str, max_pages: Optional[int] = None, **params: Any
    ) -> Iterable[SearchResult]:
        """
        Generator that walks through result pages until FINN reports the end
        or max_pages is reached.
        """
        page = 1
        while True:
            result = self.search(query=query, page=page, **params)
            yield result

            should_stop = result.is_end or (max_pages and page >= max_pages)
            if not result.listings:
                should_stop = True

            if should_stop:
                break
            page += 1

    def search_dataframe(
        self, query: str, max_pages: Optional[int] = None, include_raw: bool = False, **params: Any
    ):
        """
        Fetch multiple pages and return a single pandas DataFrame.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Install pandas to use search_dataframe(), e.g. pip install pandas"
            ) from exc

        frames = []
        for result in self.iter_search(query=query, max_pages=max_pages, **params):
            df = result.to_dataframe(include_raw=include_raw)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _parse_search_response(self, html: str) -> SearchResult:
        payload = self._extract_dehydrated_payload(html)
        search_data = self._extract_search_data(payload)

        metadata = search_data.get("metadata") or {}
        docs = search_data.get("docs") or []

        listings = [self._build_listing(doc) for doc in docs if isinstance(doc, dict)]

        result_size = metadata.get("result_size") or {}
        paging = metadata.get("paging") or {}

        total_matches = result_size.get("match_count", len(listings))
        page = paging.get("current", 1)
        last_page = paging.get("last")
        is_end = bool(
            metadata.get("is_end_of_paging")
            or (last_page is not None and page >= last_page)
        )

        return SearchResult(
            listings=listings,
            total_matches=total_matches,
            page=page,
            last_page=last_page,
            is_end=is_end,
            metadata=metadata,
            raw=search_data,
        )

    def _extract_dehydrated_payload(self, html: str) -> Dict[str, Any]:
        for encoded in self._iter_base64_blobs(html):
            decoded_bytes = self._safe_b64decode(encoded)
            if decoded_bytes is None:
                continue
            try:
                data = json.loads(decoded_bytes)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "queries" in data:
                return data
        raise ValueError("Unable to locate FINN search payload in page")

    def _extract_search_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for query in payload.get("queries", []):
            state = query.get("state") or {}
            data = state.get("data")
            if isinstance(data, dict) and "docs" in data:
                return data
        raise ValueError("Unable to locate search results in payload")

    def _iter_base64_blobs(self, html: str) -> Iterable[str]:
        pattern = re.compile(r"<script[^>]*>(eyJ[^<]+)</script>", re.DOTALL)
        for match in pattern.finditer(html):
            yield match.group(1).strip()

    def _safe_b64decode(self, encoded: str) -> Optional[bytes]:
        cleaned = encoded.replace("\n", "").replace("\r", "").strip()
        padding = len(cleaned) % 4
        if padding:
            cleaned += "=" * (4 - padding)
        try:
            return base64.b64decode(cleaned)
        except (base64.binascii.Error, ValueError):
            return None

    def _build_listing(self, doc: Dict[str, Any]) -> Listing:
        price = doc.get("price") or {}
        return Listing(
            id=str(doc.get("id") or doc.get("ad_id") or ""),
            title=doc.get("heading") or "",
            url=self._absolute_url(doc.get("canonical_url")),
            location=doc.get("location"),
            price_amount=price.get("amount"),
            price_currency=price.get("currency_code"),
            price_unit=price.get("price_unit"),
            image_url=self._extract_image_url(doc),
            timestamp=self._parse_timestamp(doc.get("timestamp")),
            raw=doc,
        )

    def _extract_image_url(self, doc: Dict[str, Any]) -> Optional[str]:
        image = doc.get("image")
        if isinstance(image, dict) and image.get("url"):
            return image["url"]
        urls = doc.get("image_urls")
        if isinstance(urls, list) and urls:
            return urls[0]
        return None

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None

    def _absolute_url(self, url: Optional[str]) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"


def main() -> None:
    api = FinnAPI()
    result = api.search(query="gitar")

    page_info = f"{result.page}/{result.last_page}" if result.last_page else str(result.page)
    print(f"Found {result.total_matches} matches (page {page_info}). Showing first 5:")

    for listing in result.listings[:5]:
        print(
            f"- {listing.title} | {listing.location or 'Unknown'} | "
            f"{listing.price_display} | {listing.url}"
        )

    # Full DataFrame across all pages (capped to first 3 pages as an example)
    try:
        df = api.search_dataframe("gitar", max_pages=3)
        print(f"\nFetched {len(df)} rows across up to 3 pages.")
    except ImportError:
        print("\nInstall pandas to export results to a DataFrame.")


if __name__ == "__main__":
    main()
