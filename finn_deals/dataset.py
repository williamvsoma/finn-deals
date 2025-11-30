import argparse
from pathlib import Path
import pandas as pd

from finn_deals.features import prepare_dataframe
from finn_deals.scraping.finn import FinnAPI


def collect_data(query: str, pages: int, output: str, prepared_output: str | None):
    api = FinnAPI()
    df = api.search_dataframe(query, max_pages=pages, include_raw=False)
    if df.empty:
        raise SystemExit("No data fetched; adjust query or page count.")

    raw_path = Path(output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw scrape ({len(df)} rows) to {raw_path}")

    if prepared_output:
        prepped = prepare_dataframe(df)
        processed_path = Path(prepared_output)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        prepped.to_csv(processed_path, index=False)
        print(f"Saved prepared dataset ({len(prepped)} rows) to {processed_path}")


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
        default="data/raw/listings_raw.csv",
        help="Path to save raw scraped data.",
    )
    parser.add_argument(
        "--prepared-output",
        default="data/processed/listings_prepared.csv",
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
