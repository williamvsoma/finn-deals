import numpy as np
import pandas as pd

from finn_deals.features import prepare_dataframe


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
    assert prepped.loc[prepped["title"] == "Gitar til salgs", "text"].item() == "Gitar til salgs Oslo"
    assert prepped.loc[prepped["title"] == "Forsterker", "text"].item() == "Forsterker Bergen"

    # Encodes timestamp numeric and fills missing with median
    ts_val = prepped.loc[prepped["title"] == "Gitar til salgs", "timestamp_val"].item()
    filled_ts = prepped.loc[prepped["title"] == "Forsterker", "timestamp_val"].item()
    assert np.isclose(ts_val, filled_ts)
