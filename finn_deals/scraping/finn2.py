import re
import requests
import base64
import json
from dataclasses import dataclass
from typing import List, Any, Dict, Iterable, Optional
import logging

logger = logging.getLogger(__name__)

class FinnAPI:
    """
    Scraper wrapper around FINN Torget search.
    """

    base_url = "https://www.finn.no"
    search_path = "/recommerce/forsale/search"

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.setdefault("Usser-Agent", "finn-scraper/0.2")
        self.max_public_results = 2000

    def search(self, query:str, page:int = 1, price_from: Optional[int] = None, price_to: Optional[int] = None, **params: Any) -> Any:
        params = {"q": query, "page": page}

        if price_from is not None:
            params["price_from"] = price_from
        if price_to is not None:
            params["price_to"] = price_to

        response = self.session.get(
            f"{self.base_url}{self.search_path}", params=params,
            timeout=15
        )
        response.raise_for_status()
        return self._parse_search_response(response.text)

    def _parse_search_response(self, html: str) -> Any:
        payload = self._extract_dehydrated_payload(html)
        search_data = self._extract_search_data(payload)
        return search_data
    
    def _extract_search_data(self, payload: Dict[str, Any]) -> Any:
        for query in payload.get("queries", []):
            state = query.get("state", {})
            data = state.get("data", {})
            if isinstance(data, dict) and "docs" in data:
                return data
        raise ValueError("Unable to locate search results in payload")

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


    def _iter_base64_blobs(self, html: str) -> Iterable[str]:
        pattern = re.compile(r"<script[^>]*>(eyJ[^<]+)</script>", re.DOTALL)
        for match in pattern.finditer(html):
            yield match.group(1).strip()
    
    def _safe_b64decode(self, encoded:str) -> Optional[bytes]:
        cleaned = encoded.replace("\n", "").replace("\r", "").strip()
        padding = len(cleaned) % 4
        if padding:
            cleaned += "=" * (4 - padding)
        try:
            return base64.b64decode(cleaned)
        except (base64.binascii.Error, ValueError):
            return None

def main():
    api = FinnAPI()
    result = api.search("gitar")
    print(result)

if __name__ == "__main__":
    main()