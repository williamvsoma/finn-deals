from typing import Any, Generator, Iterable
from tenacity import retry, stop_after_attempt, wait_fixed
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import json
import base64
import binascii
import requests
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FinnAPI:
    """
    Scraper wrapper around FINN Torget search.
    """

    base_url = "https://www.finn.no"
    search_path = "/recommerce/forsale/search"
    item_path = "/recommerce/forsale/item"
    max_pages = 50

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "finn-scraper/0.3")

    def search(self, query: str, max_workers: int = 4) -> pd.DataFrame:
        """
        Search FINN for the given query and return a DataFrame with one row per listing.

        Concurrently fetches item-page details (description, all image URLs,
        condition, location, seller info, etc.) for each listing.
        """
        df = self._iter_price_ranges(query)
        if not df.empty and "ad_id" in df.columns:
            df = self._enrich_with_item_details(df, max_workers=max_workers)
        return df

    def _item_page(self, ad_id: str) -> dict[str, Any]:
        """
        Fetch a single item page and return the structured item data
        extracted from the page's hydration payload.

        The returned dict contains all item-level detail that FINN embeds
        in the page (itemData, transactableData, profileData, etc.).
        """
        response = self._request_with_retries(
            f"{self.base_url}{self.item_path}/{ad_id}",
            params={},
        )
        return self._parse_item_page(response.text)

    def get_item(self, ad_id: str) -> pd.DataFrame:
        """
        Fetch a single FINN listing by ad-id and return a one-row
        DataFrame with every available field flattened into columns.
        """
        raw = self._item_page(str(ad_id))
        return self._extract_item(raw)

    def get_items(self, ad_ids: Iterable[str]) -> pd.DataFrame:
        """
        Fetch multiple FINN listings and return a DataFrame with one
        row per listing.
        """
        frames = []
        for ad_id in ad_ids:
            try:
                frames.append(self.get_item(str(ad_id)))
            except Exception as exc:
                logger.warning("Failed to fetch item %s: %s", ad_id, exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _get_item_details(self, ad_ids: Iterable[str], max_workers: int = 8) -> dict[str, dict[str, Any]]:
        """
        Fetch full item-page details for multiple ad IDs concurrently.

        Returns a mapping of ad_id → dict with keys useful for price
        prediction: description, all image URLs, condition, location,
        seller type, category details, etc.
        """
        ad_ids = [str(aid) for aid in ad_ids]

        def _fetch_one(ad_id: str) -> tuple[str, dict[str, Any]]:
            raw = self._item_page(ad_id)
            item_data = raw.get("itemData") or {}
            transactable = raw.get("transactableData") or {}

            # All image URLs (full list from item page)
            images = item_data.get("images") or []
            all_image_urls = [img.get("uri") for img in images if img.get("uri")]

            # Location details
            location = item_data.get("location") or {}
            position = location.get("position") or {}

            # Extras (condition, brand, model, etc.)
            extras = {}
            for extra in item_data.get("extras") or []:
                eid = extra.get("id", "")
                if eid:
                    extras[f"extra_{eid}"] = extra.get("value")

            # Category hierarchy
            cat = item_data.get("category") or {}
            parent_cat = cat.get("parent") or {}
            grandparent_cat = parent_cat.get("parent") or {}

            detail: dict[str, Any] = {
                "description": item_data.get("description", ""),
                "all_image_urls": all_image_urls,
                "num_images_detail": len(all_image_urls),
                "postal_code": location.get("postalCode"),
                "postal_name": location.get("postalName"),
                "latitude": position.get("lat"),
                "longitude": position.get("lng"),
                "category": cat.get("value"),
                "category_id": cat.get("id"),
                "sub_category": parent_cat.get("value"),
                "sub_category_id": parent_cat.get("id"),
                "top_category": grandparent_cat.get("value"),
                "top_category_id": grandparent_cat.get("id"),
                "txn_buyNow": transactable.get("buyNow"),
                "txn_shipping": transactable.get("eligibleForShipping"),
                **extras,
            }

            # Seller / retailer info
            if "follower_count" in raw:
                detail["seller_follower_count"] = raw["follower_count"]
            cp = raw.get("companyProfile")
            if cp and isinstance(cp, dict):
                detail["seller_org_name"] = cp.get("org_name")
                detail["seller_org_id"] = cp.get("org_id")

            return ad_id, detail

        details: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, aid): aid for aid in ad_ids}
            for future in as_completed(futures):
                ad_id = futures[future]
                try:
                    _, detail = future.result()
                    details[ad_id] = detail
                except Exception as exc:
                    logger.warning("Failed to fetch details for %s: %s", ad_id, exc)
        return details

    def _enrich_with_item_details(self, df: pd.DataFrame, max_workers: int = 8, ad_id_col: str = "ad_id") -> pd.DataFrame:
        """
        Enrich a search-results DataFrame with full item-page details:
        description, all image URLs, condition, location, seller info, etc.
        """
        ad_ids = df[ad_id_col].dropna().unique().astype(str).tolist()
        logger.info("Fetching item details for %d listings...", len(ad_ids))
        details = self._get_item_details(ad_ids, max_workers=max_workers)

        # Build a DataFrame from the details dict and merge on ad_id
        if not details:
            return df

        details_df = pd.DataFrame.from_dict(details, orient="index")
        details_df.index.name = ad_id_col
        details_df = details_df.reset_index()
        details_df[ad_id_col] = details_df[ad_id_col].astype(str)

        result = df.copy()
        result[ad_id_col] = result[ad_id_col].astype(str)

        # Only merge columns that don't already exist (avoid clobbering search data)
        existing_cols = set(result.columns)
        new_cols = [c for c in details_df.columns if c not in existing_cols or c == ad_id_col]
        result = result.merge(details_df[new_cols], on=ad_id_col, how="left")

        return result

    # ------------------------------------------------------------------
    # Item-page parsing helpers
    # ------------------------------------------------------------------

    _HYDRATION_RE = re.compile(
        r'window\.__staticRouterHydrationData\s*=\s*JSON\.parse\("(.*)"\)',
        re.DOTALL,
    )

    _FOLLOWER_RE = re.compile(r"(\d+)\s+følger")

    _COMPANY_PROFILE_RE = re.compile(
        r"<companyProfilePodlet-isolated[^>]*>(.*?)</companyProfilePodlet-isolated>",
        re.DOTALL,
    )

    def _parse_item_page(self, html: str) -> dict[str, Any]:
        """
        Extract the hydration JSON from an item page and return a dict
        with keys ``itemData``, ``transactableData``, ``profileData``,
        and ``meta``.
        """
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)

        for script in scripts:
            script = script.strip()
            m = self._HYDRATION_RE.search(script)
            if not m:
                continue

            # The JSON is double-escaped inside a JS string literal.
            # decode("unicode_escape") treats each byte as latin-1, so
            # raw UTF-8 sequences (æøå) end up as individual latin-1
            # chars that need a round-trip to recover. However, \uXXXX
            # escapes for chars > 0xFF (e.g. U+2028) are decoded directly
            # and must be kept as-is (they can't survive latin-1 encoding).
            decoded = m.group(1).encode("utf-8").decode("unicode_escape")
            parts = re.split(r"([^\x00-\xff]+)", decoded)
            unescaped = "".join(
                part.encode("latin-1").decode("utf-8") if i % 2 == 0 else part
                for i, part in enumerate(parts)
            )
            data = json.loads(unescaped)

            # The item loader lives under either "item-recommerce" (private)
            # or "item-bap" (retailer / webstore).
            loader = data.get("loaderData", {})
            for key in loader:
                if key.startswith("item"):
                    section = loader[key]
                    result = {
                        "itemData": section.get("itemData", {}),
                        "transactableData": section.get("transactableData", {}),
                        "profileData": section.get("profileData", {}),
                        "meta": section.get("meta", {}),
                    }

                    # --- follower count (retailer ads only) ---------------
                    fm = self._FOLLOWER_RE.search(html)
                    if fm:
                        result["follower_count"] = int(fm.group(1))

                    # --- company profile (retailer ads only) --------------
                    result["companyProfile"] = self._extract_company_profile(html)

                    return result

        raise ValueError("Unable to locate item hydration data in page")

    def _extract_company_profile(self, html: str) -> dict[str, Any] | None:
        """
        Extract company/seller profile data from the
        ``<companyProfilePodlet-isolated>`` shadow-DOM element, if present.

        Returns a dict with org name, org id, contacts, logo, homepage
        url, etc. – or ``None`` for private-seller ads.
        """
        m = self._COMPANY_PROFILE_RE.search(html)
        if not m:
            return None

        inner = m.group(1)
        # The profile data sits inside a <script> tag as JSON
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", inner, re.DOTALL)
        for script_body in scripts:
            script_body = script_body.strip()
            if not script_body.startswith("{"):
                continue
            try:
                profile_json = json.loads(script_body)
            except json.JSONDecodeError:
                continue

            # The relevant data is nested under extendedProfileRecommerce
            profile = profile_json.get("extendedProfileRecommerce")
            if profile and isinstance(profile, dict):
                return {
                    "org_name": profile.get("orgName"),
                    "org_id": profile.get("orgId"),
                    "logo_url": profile.get("logo"),
                    "homepage_url": profile.get("homepageUrl"),
                    "more_ads_url": profile.get("moreAdsFromThisCompanyUrl"),
                    "contacts": profile.get("contacts"),  # list of {name, phone}
                }

        return None

    def _extract_item(self, raw: dict[str, Any]) -> pd.DataFrame:
        """
        Flatten the parsed item-page dict into a single-row DataFrame
        using ``pd.json_normalize``.
        """
        item = dict(raw.get("itemData") or {})
        transactable = raw.get("transactableData") or {}
        meta_page = raw.get("meta") or {}

        # ---- pre-process nested structures that json_normalize
        #      would turn into unhelpful column names ----

        # extras → promote each {id, value} pair to a top-level key
        for extra in item.pop("extras", []):
            eid = extra.get("id", "")
            if eid:
                item[f"extra_{eid}"] = extra.get("value")
                if "valueId" in extra:
                    item[f"extra_{eid}_id"] = extra.get("valueId")

        # category hierarchy → flatten to leaf / parent / grandparent
        cat = item.pop("category", None) or {}
        item["category"] = cat.get("value")
        item["category_id"] = cat.get("id")
        parent = cat.get("parent") or {}
        item["sub_category"] = parent.get("value")
        item["sub_category_id"] = parent.get("id")
        grandparent = parent.get("parent") or {}
        item["top_category"] = grandparent.get("value")
        item["top_category_id"] = grandparent.get("id")

        # images → count + first image url + all urls as list
        images = item.pop("images", []) or []
        item["num_images"] = len(images)
        item["image_url"] = images[0].get("uri") if images else None
        item["image_urls"] = [img.get("uri") for img in images]

        # location → flatten the interesting bits, drop map links
        loc = item.pop("location", None) or {}
        pos = loc.get("position") or {}
        item["latitude"] = pos.get("lat")
        item["longitude"] = pos.get("lng")
        item["coord_accuracy"] = pos.get("accuracy")
        item["postal_code"] = loc.get("postalCode")
        item["postal_name"] = loc.get("postalName")
        item["country_code"] = loc.get("countryCode")

        # meta (ad-level) → prefix with meta_
        ad_meta = item.pop("meta", None) or {}
        for k, v in ad_meta.items():
            item[f"meta_{k}"] = v

        # transactableData → prefix with txn_
        for k, v in transactable.items():
            item[f"txn_{k}"] = v

        # page meta (title, canonical, etc.)
        if meta_page.get("canonical"):
            item["canonical_url"] = meta_page["canonical"]

        # follower count (retailer ads only)
        if "follower_count" in raw:
            item["follower_count"] = raw["follower_count"]

        # company / seller profile (retailer ads only)
        cp = raw.get("companyProfile")
        if cp and isinstance(cp, dict):
            item["seller_org_name"] = cp.get("org_name")
            item["seller_org_id"] = cp.get("org_id")
            item["seller_logo_url"] = cp.get("logo_url")
            item["seller_homepage_url"] = cp.get("homepage_url")
            item["seller_more_ads_url"] = cp.get("more_ads_url")
            contacts = cp.get("contacts") or []
            if contacts:
                item["seller_contact_name"] = contacts[0].get("name")
                item["seller_contact_phone"] = contacts[0].get("phone")

        # ---- normalize whatever remains --------------------------------
        df = pd.json_normalize([item], sep="_")

        return df

    def _iter_price_ranges(self, query: str) -> pd.DataFrame:
        max_price = 0
        results: list[pd.DataFrame] = []
        while True:
            price_range_result: list[pd.DataFrame] = []
            for df in self._iter_pages(query, price_from=max_price):
                price_range_result.append(df)

            results.extend(price_range_result)

            df = price_range_result[-1]
            max_price = df.loc[~df["is_promoted"], "price_amount"].max()

            if np.isnan(max_price):
                break
            max_price = int(max_price)

        df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        if not df.empty and "ad_id" in df.columns:
            df = df.drop_duplicates(subset="ad_id", keep="first").reset_index(drop=True)
        return df

    def _iter_pages(self, query: str, price_from: int) -> Generator[pd.DataFrame, None, None]:
        for page in range(1, self.max_pages + 1):
            df = self._query(query, page, price_from)
            if df.empty:
                break
            yield df

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3), reraise=True)
    def _request_with_retries(self, url: str, params: dict[str, str]) -> requests.Response:
        logger.debug("Searching FINN with params: %s", params)
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp

    def _query(self, query: str, page: int, price_from: int) -> pd.DataFrame:
        params: dict[str, str] = {
            "q": query,
            "page": str(page),
            "price_from": str(price_from),
            "sort": "PRICE_ASC",
        }
        response = self._request_with_retries(
            f"{self.base_url}{self.search_path}",
            params=params,
        )
        return self._parse_query_response(response.text)

    def _parse_query_response(self, html: str) -> pd.DataFrame:
        payload = self._extract_dehydrated_payload(html)
        df = self._extract_listings(payload)
        return df

    # ------------------------------------------------------------------
    # Payload extraction helpers
    # ------------------------------------------------------------------

    def _extract_dehydrated_payload(self, html: str) -> dict[str, Any]:
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

    def _safe_b64decode(self, encoded: str) -> bytes | None:
        cleaned = encoded.replace("\n", "").replace("\r", "").strip()
        padding = len(cleaned) % 4
        if padding:
            cleaned += "=" * (4 - padding)
        try:
            return base64.b64decode(cleaned)
        except (binascii.Error, ValueError):
            return None

    # ------------------------------------------------------------------
    # Search-data extraction
    # ------------------------------------------------------------------

    def _extract_search_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the query entry whose ``data`` dict contains ``docs``."""
        for query in payload.get("queries", []):
            state = query.get("state") or {}
            data = state.get("data")
            if isinstance(data, dict) and "docs" in data:
                return data
        raise ValueError("Unable to locate search results (docs) in payload")

    def _extract_promoted_entry(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return the promoted / sponsored listing (``searchEntry``), if any."""
        for query in payload.get("queries", []):
            state = query.get("state") or {}
            data = state.get("data")
            if isinstance(data, dict) and "result" in data:
                result = data["result"]
                if isinstance(result, dict) and "searchEntry" in result:
                    return result["searchEntry"]
        return None

    # ------------------------------------------------------------------
    # Public: extract listings as a DataFrame
    # ------------------------------------------------------------------

    def _extract_listings(self, payload: dict[str, Any]) -> pd.DataFrame:
        """
        Given a raw dehydrated payload (as returned by ``_query``),
        extract **all** listings (promoted + organic) and return a
        ``pandas.DataFrame`` with one row per listing and every useful
        field flattened into its own column.

        Uses ``pd.json_normalize`` to automatically flatten nested
        dicts (price, coordinates, image) and then derives a handful
        of convenience columns on top.
        """
        records: list[dict[str, Any]] = []

        # 1. promoted / sponsored listing (may not exist)
        promoted = self._extract_promoted_entry(payload)
        if promoted:
            promoted["_promoted"] = True
            records.append(promoted)

        # 2. organic docs
        search_data = self._extract_search_data(payload)
        for doc in search_data.get("docs") or []:
            if isinstance(doc, dict):
                doc["_promoted"] = False
                records.append(doc)

        if not records:
            return pd.DataFrame()

        # --- flatten with json_normalize ----------------------------------
        df = pd.json_normalize(
            records,
            sep="_",  # price.amount → price_amount
        )

        # --- unpack the ``extras`` list into proper columns ---------------
        #  extras is a list[{id, label, values}] – pivot each entry into
        #  its own column named "extra_<id>" with values joined as strings.
        if "extras" in df.columns:
            for idx, extras in df["extras"].items():
                if not isinstance(extras, list):
                    continue
                for extra in extras:
                    key = extra.get("id", "")
                    vals = extra.get("values") or []
                    if key and vals:
                        df.loc[idx, f"extra_{key}"] = ", ".join(str(v) for v in vals)
            df.drop(columns=["extras"], inplace=True)

        # --- unpack labels list into label_ids / label_texts columns ------
        if "labels" in df.columns:
            df["label_ids"] = df["labels"].apply(
                lambda lbls: ", ".join(l.get("id", "") for l in lbls)
                if isinstance(lbls, list) else None
            )
            df["label_texts"] = df["labels"].apply(
                lambda lbls: ", ".join(l.get("text", "") for l in lbls)
                if isinstance(lbls, list) else None
            )
            df.drop(columns=["labels"], inplace=True)

        # --- convenience boolean flags from the flags list ----------------
        if "flags" in df.columns:
            flag_col = df["flags"].apply(lambda f: f if isinstance(f, list) else [])
            df["is_private"] = flag_col.apply(lambda f: "private" in f)
            df["is_retailer"] = flag_col.apply(lambda f: "retailer" in f)
            df["has_shipping"] = flag_col.apply(lambda f: "shipping_exists" in f)
            df["has_buy_now"] = flag_col.apply(lambda f: "buy_now" in f)
            # keep flags as a comma-separated string
            df["flags"] = flag_col.apply(lambda f: ", ".join(f) if f else None)

        # --- num_images ---------------------------------------------------
        if "image_urls" in df.columns:
            df["num_images"] = df["image_urls"].apply(
                lambda u: len(u) if isinstance(u, list) else 0
            )

        # --- readable timestamp from epoch ms -----------------------------
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")

        # --- rename _promoted sentinel ------------------------------------
        if "_promoted" in df.columns:
            df.rename(columns={"_promoted": "is_promoted"}, inplace=True)

        # --- attach pagination / result-size metadata ---------------------
        metadata = search_data.get("metadata") or {}
        paging = metadata.get("paging") or {}
        result_size = metadata.get("result_size") or {}

        df["page_current"] = paging.get("current")
        df["page_last"] = paging.get("last")
        df["total_match_count"] = result_size.get("match_count")
        df["total_group_count"] = result_size.get("group_count")
        df["search_query"] = (metadata.get("params") or {}).get("q", [None])[0]

        return df


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)

    api = FinnAPI()

    # --- single item detail ---
    print("\n=== Item Detail ===")
    df = api.get_item("451778168")
    print(f"Columns ({len(df.columns)}):", list(df.columns))
    print()
    cols = [c for c in [
        "meta_adId", "title", "price", "postal_name",
        "extra_condition", "extra_phone_brand", "extra_mobile_model",
        "extra_mobile_memory_size", "category", "sub_category",
        "description", "num_images",
        "txn_transactable", "txn_buyNow", "txn_eligibleForShipping",
        "follower_count", "seller_org_name", "seller_org_id",
        "seller_contact_name", "seller_contact_phone",
    ] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()