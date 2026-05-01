import numpy as np
import pandas as pd

from finn_deals.modeling.utils import (
    EmbeddingEncoder,
    Log1pMinMaxScaler,
    WordPieceTokenizer,
)
from sklearn.preprocessing import MinMaxScaler
from typing import Dict


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw FINN listings and engineer model-friendly columns.
    Keeps only priced rows, builds combined text, and encodes timestamp numerically.
    """
    clean = df.copy()
    clean = clean[clean["price_amount"].notna()]

    title_col = "heading" if "heading" in clean.columns else "title"
    timestamp_col = "timestamp_dt" if "timestamp_dt" in clean.columns else "timestamp"

    clean["text"] = (
        clean[title_col].fillna("") + " " + clean["location"].fillna("")
    ).str.strip()

    ts = pd.to_datetime(clean[timestamp_col], errors="coerce", utc=True)
    ts_seconds = ts.astype("int64") / 1e9
    ts_seconds = ts_seconds.where(~ts.isna(), np.nan)
    median_ts = ts_seconds.dropna().median()
    clean["timestamp_val"] = ts_seconds.fillna(
        0.0 if np.isnan(median_ts) else median_ts
    )

    return clean


class DataPipeline:
    def __init__(
        self,
        df: pd.DataFrame | dict | None = None,
        encoding_plan: dict | None = None,
        reference_timestamp: pd.Timestamp | None = None,
    ):
        if encoding_plan is None and isinstance(df, dict):
            encoding_plan = df
            df = None
        if encoding_plan is None:
            raise ValueError("encoding_plan is required")

        self.input_df = df
        self.encoding_plan = encoding_plan
        self.reference_timestamp = reference_timestamp
        self.df = pd.DataFrame()
        self.tokenizers: Dict[str, WordPieceTokenizer] = {}
        self.encoders: Dict[str, EmbeddingEncoder] = {}
        self.feature_cols: Dict[str, list[str]] = {}
        self.target_scalers: Dict[str, Log1pMinMaxScaler] = {}
        self.log_scalers: Dict[str, Log1pMinMaxScaler] = {}
        self.temporal_scalers: Dict[str, Log1pMinMaxScaler] = {}
        self.reference_timestamps: Dict[str, pd.Timestamp] = {}
        self.numeric_scaler = MinMaxScaler()
        self.numeric_medians = pd.Series(dtype=float)
        self.categorical_low_columns: list[str] = []
        self.fitted = False

    def _dataframe(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        data = self.input_df if df is None else df
        if data is None:
            raise ValueError("No dataframe supplied")
        return data

    def fit(self, df: pd.DataFrame | None = None) -> "DataPipeline":
        data = self._dataframe(df)
        self.input_df = data
        self.feature_cols = {}

        self._fit_targets(data)
        self._fit_text(data)
        self._fit_numeric(data)
        self._fit_numeric_log(data)
        self._fit_temporal(data)
        self._fit_binary()
        self._fit_categorical_low(data)
        self._fit_categorical_high(data)

        self.fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        data = self._dataframe(df)
        self.fit(data)
        return self.transform(data)

    def transform(
        self,
        df: pd.DataFrame | None = None,
        include_target: bool = True,
    ) -> pd.DataFrame:
        data = self._dataframe(df)
        if not self.fitted:
            raise RuntimeError("DataPipeline must be fit before calling transform")

        transformed = pd.DataFrame(index=data.index)
        if include_target:
            self._transform_targets(data, transformed)
        self._transform_text(data, transformed)
        self._transform_numeric(data, transformed)
        self._transform_numeric_log(data, transformed)
        self._transform_temporal(data, transformed)
        self._transform_binary(data, transformed)
        self._transform_categorical_low(data, transformed)
        self._transform_categorical_high(data, transformed)

        self.df = transformed
        return transformed

    def _fit_targets(self, data: pd.DataFrame) -> None:
        cols = self.encoding_plan.get("target", [])
        self.target_scalers = {}
        for col in cols:
            self.target_scalers[col] = Log1pMinMaxScaler()
            self.target_scalers[col].fit(data[col].dropna().values)
        self.feature_cols["target"] = cols

    def _transform_targets(self, data: pd.DataFrame, transformed: pd.DataFrame) -> None:
        for col in self.feature_cols.get("target", []):
            if col not in data:
                continue
            series = data[col]
            nan_mask = series.isna()
            scaled = pd.Series(
                self.target_scalers[col].transform(series.fillna(0).values),
                index=data.index,
            )
            scaled[nan_mask] = 0.0
            transformed[col] = scaled

    def _build_text_series(self, data: pd.DataFrame, col: str) -> pd.Series:
        """Concatenate the primary text column with any text_concat columns."""
        concat_cols = self.encoding_plan.get("text_concat", [])
        combined = data[col].fillna("")
        for extra_col in concat_cols:
            if extra_col in data.columns:
                combined = combined + " " + data[extra_col].fillna("")
        return combined.str.strip()

    def _fit_text(self, data: pd.DataFrame) -> None:
        for col in self.encoding_plan["text"]:
            self.tokenizers[col] = WordPieceTokenizer()
            self.tokenizers[col].fit(self._build_text_series(data, col))
        self.feature_cols["text"] = self.encoding_plan["text"]

    def _transform_text(self, data: pd.DataFrame, transformed: pd.DataFrame) -> None:
        for col in self.feature_cols.get("text", []):
            transformed[col] = self.tokenizers[col](self._build_text_series(data, col))

    def _fit_numeric(self, data: pd.DataFrame) -> None:
        cols = self.encoding_plan["numeric"]
        if cols:
            numeric = data[cols]
            self.numeric_medians = numeric.median()
            self.numeric_scaler = MinMaxScaler()
            self.numeric_scaler.fit(numeric.fillna(self.numeric_medians))
        else:
            self.numeric_medians = pd.Series(dtype=float)
        self.feature_cols["numeric"] = cols

    def _transform_numeric(self, data: pd.DataFrame, transformed: pd.DataFrame) -> None:
        cols = self.feature_cols.get("numeric", [])
        if not cols:
            return
        numeric = data[cols]
        nan_mask = numeric.isna()
        scaled = pd.DataFrame(
            self.numeric_scaler.transform(numeric.fillna(self.numeric_medians)),
            columns=cols,
            index=data.index,
        )
        scaled[nan_mask] = 0.0
        transformed[cols] = scaled

    def _fit_numeric_log(self, data: pd.DataFrame) -> None:
        cols = self.encoding_plan.get("numeric_log", [])
        self.log_scalers = {}
        for col in cols:
            self.log_scalers[col] = Log1pMinMaxScaler()
            self.log_scalers[col].fit(data[col].dropna().values)
        self.feature_cols["numeric_log"] = cols

    def _transform_numeric_log(
        self,
        data: pd.DataFrame,
        transformed: pd.DataFrame,
    ) -> None:
        for col in self.feature_cols.get("numeric_log", []):
            series = data[col]
            nan_mask = series.isna()
            scaled = pd.Series(
                self.log_scalers[col].transform(series.fillna(0).values),
                index=data.index,
            )
            scaled[nan_mask] = 0.0
            transformed[col] = scaled

    @staticmethod
    def _sine_encoder(x: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
        angle = 2 * np.pi * x / period
        return (np.sin(angle) + 1) / 2, (np.cos(angle) + 1) / 2

    @staticmethod
    def _parse_timestamp(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", utc=True)

    def _fit_temporal(self, data: pd.DataFrame) -> None:
        temporal_cols: list[str] = []
        self.temporal_scalers = {}
        self.reference_timestamps = {}

        for col in self.encoding_plan["temporal"]:
            ts = self._parse_timestamp(data[col])
            reference = (
                pd.Timestamp(self.reference_timestamp).tz_localize("UTC")
                if self.reference_timestamp is not None
                and pd.Timestamp(self.reference_timestamp).tzinfo is None
                else pd.Timestamp(self.reference_timestamp)
                if self.reference_timestamp is not None
                else ts.max()
            )
            if pd.isna(reference):
                reference = pd.Timestamp("1970-01-01", tz="UTC")
            reference = pd.Timestamp(reference).tz_convert("UTC")
            self.reference_timestamps[col] = reference

            days_ago = (reference - ts).dt.total_seconds() / 86400
            days_ago = days_ago.clip(lower=0)
            self.temporal_scalers[col] = Log1pMinMaxScaler()
            self.temporal_scalers[col].fit(days_ago.dropna().values)
            temporal_cols.append(col + "_days_ago")
            temporal_cols += [col + "_hour_sin", col + "_hour_cos"]
            temporal_cols += [col + "_dow_sin", col + "_dow_cos"]
            temporal_cols += [col + "_dom_sin", col + "_dom_cos"]
            temporal_cols += [col + "_month_sin", col + "_month_cos"]
            temporal_cols += [col + "_week_sin", col + "_week_cos"]
            temporal_cols += [col + "_doy_sin", col + "_doy_cos"]
        self.feature_cols["temporal"] = temporal_cols

    def _transform_temporal(
        self, data: pd.DataFrame, transformed: pd.DataFrame
    ) -> None:
        for col in self.encoding_plan["temporal"]:
            ts = self._parse_timestamp(data[col])
            nan_mask = ts.isna()

            days_ago = (self.reference_timestamps[col] - ts).dt.total_seconds() / 86400
            days_ago = days_ago.clip(lower=0)
            scaled_days = pd.Series(
                self.temporal_scalers[col].transform(days_ago.fillna(0).values),
                index=data.index,
            )
            scaled_days[nan_mask] = 0.0
            transformed[col + "_days_ago"] = scaled_days

            hour = ts.dt.hour.to_numpy(dtype=float)
            sin_h, cos_h = self._sine_encoder(hour, 24)
            transformed[col + "_hour_sin"] = np.where(nan_mask, 0.0, sin_h)
            transformed[col + "_hour_cos"] = np.where(nan_mask, 0.0, cos_h)

            dow = ts.dt.dayofweek.to_numpy(dtype=float)
            sin_d, cos_d = self._sine_encoder(dow, 7)
            transformed[col + "_dow_sin"] = np.where(nan_mask, 0.0, sin_d)
            transformed[col + "_dow_cos"] = np.where(nan_mask, 0.0, cos_d)

            dom = ts.dt.day.to_numpy(dtype=float)
            days_in_month = ts.dt.days_in_month.to_numpy(dtype=float)
            angle_dm = 2 * np.pi * dom / days_in_month
            transformed[col + "_dom_sin"] = np.where(
                nan_mask,
                0.0,
                (np.sin(angle_dm) + 1) / 2,
            )
            transformed[col + "_dom_cos"] = np.where(
                nan_mask,
                0.0,
                (np.cos(angle_dm) + 1) / 2,
            )

            month = ts.dt.month.to_numpy(dtype=float)
            sin_m, cos_m = self._sine_encoder(month, 12)
            transformed[col + "_month_sin"] = np.where(nan_mask, 0.0, sin_m)
            transformed[col + "_month_cos"] = np.where(nan_mask, 0.0, cos_m)

            week = ts.dt.isocalendar().week.to_numpy(dtype=float)
            sin_w, cos_w = self._sine_encoder(week, 52)
            transformed[col + "_week_sin"] = np.where(nan_mask, 0.0, sin_w)
            transformed[col + "_week_cos"] = np.where(nan_mask, 0.0, cos_w)

            doy = ts.dt.dayofyear.to_numpy(dtype=float)
            sin_dy, cos_dy = self._sine_encoder(doy, 365)
            transformed[col + "_doy_sin"] = np.where(nan_mask, 0.0, sin_dy)
            transformed[col + "_doy_cos"] = np.where(nan_mask, 0.0, cos_dy)

    def _fit_binary(self) -> None:
        cols = self.encoding_plan["binary"]
        self.feature_cols["binary"] = cols

    def _transform_binary(self, data: pd.DataFrame, transformed: pd.DataFrame) -> None:
        cols = self.feature_cols.get("binary", [])
        if cols:
            transformed[cols] = data[cols].fillna(0).astype(int)

    def _fit_categorical_low(self, data: pd.DataFrame) -> None:
        cols = self.encoding_plan["categorical_low"]
        onehot_encoded = self._onehot_low(data, cols)
        self.categorical_low_columns = onehot_encoded.columns.tolist()
        self.feature_cols["categorical_low"] = self.categorical_low_columns

    @staticmethod
    def _onehot_low(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=data.index)
        categorical = data[cols].astype("string").fillna("<MISSING>")
        return pd.get_dummies(categorical, columns=cols, dtype=int)

    def _transform_categorical_low(
        self,
        data: pd.DataFrame,
        transformed: pd.DataFrame,
    ) -> None:
        cols = self.encoding_plan["categorical_low"]
        onehot_encoded = self._onehot_low(data, cols)
        onehot_encoded = onehot_encoded.reindex(
            columns=self.categorical_low_columns,
            fill_value=0,
        )
        transformed[self.categorical_low_columns] = onehot_encoded

    def _fit_categorical_high(self, data: pd.DataFrame) -> None:
        cols = self.encoding_plan["categorical_high"]
        for col in self.encoding_plan["categorical_high"]:
            self.encoders[col] = EmbeddingEncoder()
            self.encoders[col].fit(data[col])
        self.feature_cols["categorical_high"] = cols

    def _transform_categorical_high(
        self,
        data: pd.DataFrame,
        transformed: pd.DataFrame,
    ) -> None:
        for col in self.feature_cols.get("categorical_high", []):
            transformed[col] = self.encoders[col](data[col])
