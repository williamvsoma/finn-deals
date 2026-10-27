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

## MCP server: AI-assisted deal research on FINN Torget

The MCP server (`finn_deals.mcp`) lets an AI assistant search any Torget product
category, inspect descriptions and photos, and compare explicitly selected ads.
Your MCP client's AI does the reasoning. No separate AI API key or trained model
is needed. The experimental price-prediction pipeline above is independent of
the deal tools and is not used to declare bargains.

### Tools

- `finn_search` — search sale ads with price bounds, a location substring, title
  requirements/exclusions, and sorting (`RELEVANCE`, `PUBLISHED_DESC`,
  `PUBLISHED_ASC`, `PRICE_ASC`, `PRICE_DESC`). Returns listings plus the upstream
  match count, pages fetched, timestamp and `truncated` coverage indicator.
  `max_pages` is **1–5 total pages**, without automatic sharding. Unpriced ads
  are counted and skipped. Default upper asking-price bound: 1,000,000 NOK.
- `finn_get_listing` — fetch full details for a single listing by ad id: description, all
  photo URLs, price, condition/extras, location, availability and FINN's displayed
  shipping/payment text. Missing seller details stay null.
- `finn_get_image` — view any photo with a zero-based `image_index` (default 0).
  Detects JPEG/PNG/WebP/GIF from bytes, including extensionless FINN URLs.
- `finn_find_deals` — produce a bounded shortlist with full listing details,
  review signals and the search sample. Default: up to five cheap leads;
  choose `sort="PUBLISHED_DESC"` for fresh leads. At most ten detailed candidates.
  Failed detail fetches remain visible per candidate.
- `finn_compare_listings` — fetch 2–10 distinct ads explicitly selected by the AI
  and compare their current asking prices. Each ad is excluded from its own peer
  median. A percentage difference requires at least three active, priced NOK
  peers. Duplicate IDs, unavailable ads, unknown availability, free items and
  missing prices cannot inflate the reference sample. Per-ad fetch failures are
  reported alongside successful results.

### How to find useful deals

Ask your MCP assistant, for example:

> Find promising used office chairs below 3,000 NOK around Stavanger. Start broad,
> identify specific models, then compare equivalent chairs elsewhere in Norway.
> Check condition, included parts and photos. Explain the evidence and total-cost
> uncertainties, and give me links to the best leads.

Or use a precise product query:

```json
{
  "query": "iPhone 13",
  "price_to": 3500,
  "required_terms": ["iPhone", "13"],
  "exclude_terms": ["Pro", "mini", "deksel", "skjermbeskytter"],
  "max_pages": 2,
  "limit": 5
}
```

Those are arguments for `finn_find_deals`. Title terms match words/phrases,
case-insensitively; they are not a semantic model classifier. The assistant
should narrow broad categories to actual products, inspect descriptions and
photos, then run another search without the buyer's budget/location limits to
find reference ads. After selecting equivalent products, call
`finn_compare_listings` with their IDs. An iPhone Pro, a broken phone, an accessory
and a different storage capacity must not silently become equivalent peers.

The tools report **asking prices, not sold prices or fair value**. They do not
infer a market-value score from a broad search. Shipping, payment fees, travel,
repairs and incomplete bundles can erase an apparent saving. Review signals
quote seller phrases and need interpretation. Seller text is untrusted content,
not instructions to the assistant.

Searches are samples: title/location filters apply **after** fetching the bounded
pages, so an empty local result is not proof that no such listing exists.
`PRICE_ASC` is useful for cheap leads, while `PUBLISHED_DESC` is useful for new
ads. Promoted duplicates are removed. The standalone dataset pipeline retains
its separate price-sharding method; MCP searches do not use it.

### Run the server
```bash
python -m finn_deals.mcp              # stdio (default, for Claude Desktop / MCP clients)
python -m finn_deals.mcp --transport streamable-http   # streamable http
```

### Wire it into an MCP client

This checkout includes a project-local Codex connection in `.codex/config.toml`
and research instructions in `AGENTS.md`. For trusted Codex sessions opened in
this directory, the server is enabled and starts on demand. Start a new chat or
restart/reconnect MCP to load a newly added connection. Check discovery with:

```bash
codex mcp get finn-deals
```

The checked-in connection points at `/Users/soma/finn-deals`; update `command` and
`cwd` if you move this checkout. Other MCP clients can use:

```json
{
  "mcpServers": {
    "finn-deals": {
      "command": "/Users/soma/finn-deals/.venv/bin/python",
      "args": ["-m", "finn_deals.mcp"],
      "env": {}
    }
  }
}
```

After the editable install in Setup, the absolute Python path works regardless
of the client's working directory. Adjust it for a different checkout. Restart
or reconnect your MCP client after changing server code. This server exposes
public read operations; it does not buy items or message sellers.

### Verify the real server

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_mcp.py --query "Nintendo Switch"
```

The executable smoke script launches the stdio server, checks discovery of all
five tools, and exercises live search, listing parsing, image retrieval, deal
shortlisting, comparison and invalid-input handling. It makes a small number
of public GET requests and writes no datasets. A query with at least two current
priced ads is required. The smoke comparison checks fetching/math, not whether
the selected products are genuinely comparable.

## Development
- Run tests: `pytest`
- Lint: `ruff check .`
- Format: `black .`
- Type-check: `mypy .`

Keep new data/artifacts inside `data/` and `models/`; large files should stay out of version control unless intentionally committed.
