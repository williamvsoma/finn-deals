# Finn Deals
Data pipeline and toolkit for scraping Finn.no adverts, cleaning the data, and training models to predict listing prices.

## Requirements
- Python 3.13+

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development instead do:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests
Run the suite with:
```bash
pytest
```