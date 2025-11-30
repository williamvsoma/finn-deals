import argparse
import pandas as pd

from finn import FinnAPI
from train_price_model import prepare_dataframe


def collect_data(query: str, pages: int, output: str, prepared_output: str | None):
    api = FinnAPI()
    df = api.search_dataframe(query, max_pages=pages, include_raw=False)
    if df.empty:
        raise SystemExit("No data fetched; adjust query or page count.")

    df.to_csv(output, index=False)
    print(f"Saved raw scrape ({len(df)} rows) to {output}")

    if prepared_output:
        prepped = prepare_dataframe(df)
        prepped.to_csv(prepared_output, index=False)
        print(f"Saved prepared dataset ({len(prepped)} rows) to {prepared_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FINN listings into CSV for offline training."
    )
    parser.add_argument("--query", default="gitar", help="Search query string.")
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Number of pages to fetch (approx. 50 items per page, capped by FINN).",
    )
    parser.add_argument(
        "--output",
        default="listings_raw.csv",
        help="Path to save raw scraped data.",
    )
    parser.add_argument(
        "--prepared-output",
        default=None,
        help="Optional path to also save the prepared dataset (with engineered features).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_data(
        query=args.query,
        pages=args.pages,
        output=args.output,
        prepared_output=args.prepared_output,
    )


if __name__ == "__main__":
    main()
