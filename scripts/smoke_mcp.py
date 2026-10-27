#!/usr/bin/env python3
"""Exercise the actual stdio MCP server against live FINN, without writing data."""

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(query: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "finn_deals.mcp"],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    async with (
        stdio_client(params) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        await session.initialize()
        names = {tool.name for tool in (await session.list_tools()).tools}
        assert names == {
            "finn_search",
            "finn_get_listing",
            "finn_get_image",
            "finn_find_deals",
            "finn_compare_listings",
        }, names
        print("PASS: server startup and all five tool schemas", flush=True)

        async def call(name, arguments):
            result = await session.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(f"{name}: {result.content}")
            return result

        def payload(result):
            return json.loads(next(c.text for c in result.content if c.type == "text"))

        search = payload(
            await call(
                "finn_search",
                {
                    "query": query,
                    "max_pages": 1,
                    "sort": "PRICE_ASC",
                    "price_from": 1,
                },
            )
        )
        assert search["pages_fetched"] == 1
        assert search["count"] > 0, "No priced listings; choose another query"
        print(
            f"PASS: search {query!r}: {search['count']} sampled / {search['total_matches']} matches; truncated={search['truncated']}",
            flush=True,
        )
        # Pick an active ad with a photo; a removed ad should not make the
        # smoke check depend on a single stale search result.
        detail = None
        for row in search["listings"][:5]:
            result = await session.call_tool(
                "finn_get_listing", {"ad_id": row["ad_id"]}
            )
            if not result.isError:
                candidate = payload(result)
                if candidate.get("image_urls") and candidate.get("title"):
                    detail = candidate
                    break
        assert detail is not None, "No inspectable listing among first five results"
        assert detail["price_currency"] == "NOK"
        assert isinstance(detail["price_amount"], (int, float))
        print(
            f"PASS: listing {detail['ad_id']}: {detail['title']} ({detail['price_amount']} NOK), {detail['num_images']} photos",
            flush=True,
        )

        image_result = await call(
            "finn_get_image", {"ad_id": detail["ad_id"], "image_index": 0}
        )
        photo = next(c for c in image_result.content if c.type == "image")
        data = base64.b64decode(photo.data, validate=True)
        assert len(data) > 100
        # Independent decode when Pillow is installed (optional).
        try:
            from io import BytesIO

            from PIL import Image
        except ImportError:
            pass
        else:
            with Image.open(BytesIO(data)) as decoded:
                assert Image.MIME[decoded.format] == photo.mimeType
                decoded.verify()
        print(f"PASS: image {photo.mimeType}, {len(data)} bytes", flush=True)

        shortlist = payload(
            await call("finn_find_deals", {"query": query, "max_pages": 1, "limit": 2})
        )
        assert shortlist["candidate_count"] > 0
        assert any(row["detail_status"] == "ok" for row in shortlist["candidates"])
        print(
            f"PASS: deal shortlist, {shortlist['candidate_count']} candidates with detail/error evidence",
            flush=True,
        )

        ids = list(dict.fromkeys(row["ad_id"] for row in search["listings"]))[:4]
        if len(ids) >= 2:
            comparison = payload(await call("finn_compare_listings", {"ad_ids": ids}))
            assert comparison["listings"]
            print(
                f"PASS: comparison fetch/math, {len(comparison['listings'])} ads, {len(comparison['fetch_errors'])} fetch errors (product equivalence not asserted)",
                flush=True,
            )
        else:
            raise AssertionError(
                "Need at least two ads to exercise comparison; choose a broader query"
            )

        invalid = await session.call_tool(
            "finn_search", {"query": query, "max_pages": 0}
        )
        assert invalid.isError
        print(
            "PASS: invalid input returns a tool error; server remains responsive",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="Nintendo Switch",
        help="A query with several current priced listings",
    )
    asyncio.run(run(parser.parse_args().query))
