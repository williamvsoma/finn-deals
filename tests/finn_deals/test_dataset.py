import pandas as pd
import pytest

from finn_deals import dataset


class DummyAPI:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def search_dataframe(self, *args, **kwargs):
        return self.df


def test_collect_data_writes_raw_and_processed(tmp_path, monkeypatch):
    sample = pd.DataFrame(
        [
            {"title": "Item A", "location": "Oslo", "price_amount": 100, "timestamp": "2024-01-01T00:00:00Z"},
            {"title": "Item B", "location": "Bergen", "price_amount": 200, "timestamp": "2024-01-02T00:00:00Z"},
        ]
    )

    def fake_prepare(df):
        return df.assign(prepared=True)

    monkeypatch.setattr(dataset, "FinnAPI", lambda: DummyAPI(sample))
    monkeypatch.setattr(dataset, "prepare_dataframe", fake_prepare)

    raw_path = tmp_path / "data" / "raw" / "raw.csv"
    processed_path = tmp_path / "data" / "processed" / "prepared.csv"

    dataset.collect_data(
        query="anything",
        pages=1,
        output=str(raw_path),
        prepared_output=str(processed_path),
    )

    raw_df = pd.read_csv(raw_path)
    processed_df = pd.read_csv(processed_path)

    assert len(raw_df) == len(sample)
    assert len(processed_df) == len(sample)
    assert "prepared" in processed_df.columns


def test_collect_data_handles_no_prepared_output(tmp_path, monkeypatch):
    sample = pd.DataFrame([{"title": "Only raw", "location": "Oslo", "price_amount": 100, "timestamp": "2024-01-01"}])
    monkeypatch.setattr(dataset, "FinnAPI", lambda: DummyAPI(sample))

    raw_path = tmp_path / "raw.csv"
    dataset.collect_data(
        query="anything",
        pages=1,
        output=str(raw_path),
        prepared_output=None,
    )

    raw_df = pd.read_csv(raw_path)
    assert len(raw_df) == 1


def test_collect_data_raises_on_empty_result(tmp_path, monkeypatch):
    empty_df = pd.DataFrame(columns=["title", "location", "price_amount", "timestamp"])
    monkeypatch.setattr(dataset, "FinnAPI", lambda: DummyAPI(empty_df))

    with pytest.raises(SystemExit):
        dataset.collect_data(
            query="anything",
            pages=1,
            output=str(tmp_path / "raw.csv"),
            prepared_output=None,
        )
