import pandas as pd
import pytest

from finn_deals.modeling import train


def _sample_data():
    return pd.DataFrame(
        [
            {"title": "Gitar", "location": "Oslo", "price_amount": 1000, "timestamp": "2024-01-01T00:00:00Z"},
            {"title": "Bass", "location": "Bergen", "price_amount": 1500, "timestamp": "2024-01-02T00:00:00Z"},
            {"title": "Piano", "location": "Trondheim", "price_amount": 2000, "timestamp": "2024-01-03T00:00:00Z"},
            {"title": "Trommer", "location": "Oslo", "price_amount": 1200, "timestamp": "2024-01-04T00:00:00Z"},
            {"title": "Synth", "location": "Stavanger", "price_amount": 1800, "timestamp": "2024-01-05T00:00:00Z"},
        ]
    )


def test_build_pipeline_rejects_bad_model_type():
    with pytest.raises(ValueError):
        train.build_pipeline(model_type="invalid")


def test_fetch_listings_uses_finn_api(monkeypatch):
    captured = {}

    class DummyFinn:
        def search_dataframe(self, query, max_pages=None, include_raw=None):
            captured["query"] = query
            captured["max_pages"] = max_pages
            captured["include_raw"] = include_raw
            return _sample_data()

    monkeypatch.setattr(train, "FinnAPI", lambda: DummyFinn())

    df = train.fetch_listings("gitar", pages=2)

    assert captured == {"query": "gitar", "max_pages": 2, "include_raw": False}
    assert len(df) == len(_sample_data())


def test_train_price_model_creates_artifacts(tmp_path):
    csv_path = tmp_path / "data.csv"
    _sample_data().to_csv(csv_path, index=False)

    model_path = tmp_path / "models" / "price_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train.train_price_model(
        query="gitar",
        pages=1,
        test_size=0.4,
        output=str(model_path),
        model_type="ridge",
        data_csv=str(csv_path),
    )

    assert model_path.exists()
    holdout = model_path.with_suffix(".joblib.holdout.csv")
    assert holdout.exists()
