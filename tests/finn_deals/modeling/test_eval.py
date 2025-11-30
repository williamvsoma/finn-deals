import pandas as pd

from finn_deals.modeling import eval as eval_mod
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


def test_evaluate_model_runs_and_prints_metrics(tmp_path, capsys):
    data_path = tmp_path / "data.csv"
    _sample_data().to_csv(data_path, index=False)

    model_path = tmp_path / "price_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train.train_price_model(
        query="gitar",
        pages=1,
        test_size=0.4,
        output=str(model_path),
        model_type="ridge",
        data_csv=str(data_path),
    )

    eval_mod.evaluate_model(
        model_path=str(model_path),
        query="gitar",
        pages=1,
        samples=2,
        dataset_csv=str(data_path),
    )

    out = capsys.readouterr().out
    assert "MAE" in out
    assert "Sample predictions" in out
    assert "Top positive tokens" in out or "Model does not expose" in out
