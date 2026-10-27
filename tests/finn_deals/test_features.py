import numpy as np
import pandas as pd

from finn_deals.features import DataPipeline, prepare_dataframe
from finn_deals.modeling.utils import EmbeddingEncoder, WordPieceTokenizer


def test_prepare_dataframe_filters_and_engineers_columns():
    df = pd.DataFrame(
        [
            {
                "title": "Gitar til salgs",
                "location": "Oslo",
                "price_amount": 1500,
                "timestamp": "2024-01-01T12:00:00Z",
            },
            {
                "title": "Forsterker",
                "location": "Bergen",
                "price_amount": 2500,
                "timestamp": None,
            },
            {
                "title": "Gratis sak",
                "location": "Trondheim",
                "price_amount": None,
                "timestamp": "2024-02-01T00:00:00Z",
            },
        ]
    )

    prepped = prepare_dataframe(df)

    # Drops rows without price_amount
    assert len(prepped) == 2
    assert prepped["price_amount"].isna().sum() == 0

    # Builds combined text feature
    assert (
        prepped.loc[prepped["title"] == "Gitar til salgs", "text"].item()
        == "Gitar til salgs Oslo"
    )
    assert (
        prepped.loc[prepped["title"] == "Forsterker", "text"].item()
        == "Forsterker Bergen"
    )

    # Encodes timestamp numeric and fills missing with median
    ts_val = prepped.loc[prepped["title"] == "Gitar til salgs", "timestamp_val"].item()
    filled_ts = prepped.loc[prepped["title"] == "Forsterker", "timestamp_val"].item()
    assert np.isclose(ts_val, filled_ts)


def test_tokenizer_normalizes_fit_and_transform_consistently():
    tokenizer = WordPieceTokenizer().fit(["Gitar-ÆØÅ", "Rock_band"])

    unk_id = tokenizer.unk_token_id
    encoded = tokenizer(["gitar æøå", "rock band", "gitar, æøå!"])

    assert all(unk_id not in row for row in encoded)


def test_embedding_encoder_reserves_zero_for_unknowns():
    encoder = EmbeddingEncoder().fit(["Oslo", "Bergen"])

    known_idx = encoder(["Bergen"])[0]
    unknown_idx = encoder(["Trondheim"])[0]

    assert known_idx != 0
    assert unknown_idx == 0


def test_data_pipeline_fits_on_train_and_transforms_unseen_values():
    plan = {
        "target": ["price_amount"],
        "text": ["heading"],
        "numeric": ["num_images"],
        "numeric_log": [],
        "temporal": ["timestamp"],
        "binary": ["is_private"],
        "categorical_low": ["trade_type", "coordinates_accuracy"],
        "categorical_high": ["location"],
    }
    train = pd.DataFrame(
        [
            {
                "price_amount": 100,
                "heading": "Gitar",
                "num_images": 1,
                "timestamp": "2024-01-01T00:00:00Z",
                "is_private": True,
                "trade_type": "Til salgs",
                "coordinates_accuracy": 5.0,
                "location": "Oslo",
            },
            {
                "price_amount": 200,
                "heading": "Bass",
                "num_images": 3,
                "timestamp": "2024-01-02T00:00:00Z",
                "is_private": False,
                "trade_type": "Til salgs",
                "coordinates_accuracy": 6.0,
                "location": "Bergen",
            },
        ]
    )
    holdout = pd.DataFrame(
        [
            {
                "price_amount": 1000,
                "heading": "Ukjent synth",
                "num_images": 2,
                "timestamp": "2024-01-03T00:00:00Z",
                "is_private": True,
                "trade_type": "Gis bort",
                "coordinates_accuracy": 9.0,
                "location": "Trondheim",
            }
        ]
    )

    pipe = DataPipeline(
        plan,
        reference_timestamp=pd.Timestamp("2024-01-03T00:00:00Z"),
    )
    train_transformed = pipe.fit_transform(train)
    holdout_transformed = pipe.transform(holdout)

    assert train_transformed.columns.tolist() == holdout_transformed.columns.tolist()
    assert pipe.target_scalers["price_amount"].max == np.log1p(200)
    assert holdout_transformed["price_amount"].item() > 1.0
    assert holdout_transformed["location"].item() == 0
    assert "trade_type_Gis bort" not in pipe.feature_cols["categorical_low"]
    assert holdout_transformed["trade_type_Til salgs"].item() == 0
    assert "coordinates_accuracy_9.0" not in pipe.feature_cols["categorical_low"]
