# Finn Deals
Pipeline for scraping Finn.no listings, engineering features, and training price-prediction models. Structured after the Cookiecutter Data Science layout (`data/`, `notebooks/`, `finn_deals/`, `models/`, `tests/`).

## Requirements
- Python 3.13+
- pip (or uv/pipx) for installing dependencies
- Network access to Finn.no for scraping

## Project layout
- `finn_deals/` — package code
  - `scraping/` — Finn API wrapper and HTML parsing
  - `mcp/` — Finn.no MCP server for agents
  - `features.py` — feature engineering helpers
  - `dataset.py` — CLI for scraping to CSV
  - `modeling/` — training/eval/inference utilities
- `data/` — place raw/processed datasets (tracked with `.gitkeep`)
- `models/` — trained artifacts (tracked with `.gitkeep`)
- `notebooks/` — exploratory work
- `tests/` — fast unit tests (pytest)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # dev installs add pytest, ruff, black, mypy, jupyter
```

## Quickstart workflow
1) **Scrape data to CSV** (raw and prepared; defaults write to `data/raw/listings_raw.csv` and `data/processed/listings_prepared.csv`):
```bash
python -m finn_deals.dataset \
  --query "gitar" \
  --pages 5 \
  --output data/raw/listings_raw.csv \
  --prepared-output data/processed/listings_prepared.csv
```

2) **Train a model** (uses scraped or provided CSV):
```bash
python -m finn_deals.modeling.train \
  --query "gitar" \
  --pages 5 \
  --test-size 0.2 \
  --output models/price_model.joblib \
  --model-type ridge
# Or train from an existing CSV: add --data-csv data/processed/listings_prepared.csv
```

3) **Evaluate a model** (fresh scrape or dataset file):
```bash
python -m finn_deals.modeling.eval \
  --model models/price_model.joblib \
  --query "gitar" \
  --pages 3 \
  --samples 5
# Evaluate on a saved dataset: add --dataset-csv data/processed/listings_prepared.csv
```

> Note: Scraping respects Finn.no’s public interface; avoid excessive paging and consider adding delays if you scale up.

## MCP server (agent-facing Finn.no search)

The package ships an MCP server (`finn_deals.mcp`) that lets coding agents search
[Finn.no](https://www.finn.no) listings on the user's behalf and inspect the results.

### Tools

- `finn_search` — run a Finn.no search and filter listings by price range and location.
  Returns a JSON list of matching listings (`ad_id`, `title`, `url`, `location`, `price_*`,
  `image_url`, `timestamp`, `promoted`). Price-range sharding is handled internally when the
  result set is large.
- `finn_get_listing` — fetch full details for a single listing by ad id: description, all
  image urls, price, location (postal code + coordinates), condition/extras, category
  hierarchy, transaction flags and seller info.
- `finn_get_image` — fetch a listing image and return it as image content the model can view.

### Run the server
```bash
python -m finn_deals.mcp              # stdio (default, for Claude Desktop / MCP clients)
python -m finn_deals.mcp --transport streamable-http   # streamable http
```

### Wire it into an MCP client

```json
{
  "mcpServers": {
    "finn-deals": {
      "command": "python",
      "args": ["-m", "finn_deals.mcp"],
      "env": {}
    }
  }
}
```

## Development
- Run tests: `pytest`
- Lint: `ruff check .`
- Format: `black .`
- Type-check: `mypy .`

Keep new data/artifacts inside `data/` and `models/`; large files should stay out of version control unless intentionally committed.
