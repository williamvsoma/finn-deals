"""Scrape FINN listings and save to data/raw/."""

import argparse
from pathlib import Path

from finn_deals.scraping.finn import FinnAPI


def main():
    parser = argparse.ArgumentParser(description="Scrape FINN listings to CSV.")
    parser.add_argument("--query", required=True, help="Search query string.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/raw/<query>.csv).",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(f"data/raw/{args.query}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    api = FinnAPI()
    df = api.search(args.query)

    if df.empty:
        raise SystemExit("No results found.")

    df.to_csv(output, index=False)
    print(f"Saved {len(df)} listings to {output}")


if __name__ == "__main__":
    main()
