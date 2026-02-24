import numpy as np
import pandas as pd


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw FINN listings and engineer model-friendly columns.
    Keeps only priced rows, builds combined text, and encodes timestamp numerically.
    """
    clean = df.copy()
    clean = clean[clean["price_amount"].notna()]

    clean["text"] = (
        clean["title"].fillna("") + " " + clean["location"].fillna("")
    ).str.strip()

    ts = pd.to_datetime(clean["timestamp"], errors="coerce")
    ts_seconds = ts.astype("int64", copy=False) / 1e9
    ts_seconds = ts_seconds.where(~ts.isna(), np.nan)
    median_ts = ts_seconds.dropna().median()
    clean["timestamp_val"] = ts_seconds.fillna(median_ts)

    return clean

