import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finn import FinnAPI


def fetch_listings(query: str, pages: int) -> pd.DataFrame:
    api = FinnAPI()
    df = api.search_dataframe(query, max_pages=pages, include_raw=False)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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


def build_pipeline(model_type: str = "ridge") -> Pipeline:
    text_vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    time_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "text"),
            ("time", time_pipeline, ["timestamp_val"]),
        ]
    )

    if model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "Install xgboost to use model_type='xgb' (pip install xgboost)"
            ) from exc

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_price_model(
    query: str,
    pages: int,
    test_size: float,
    output: str,
    model_type: str,
    data_csv: str | None = None,
) -> None:
    if data_csv:
        raw_df = pd.read_csv(data_csv)
        source = f"loaded from {data_csv}"
    else:
        raw_df = fetch_listings(query, pages)
        source = f"scraped query='{query}', pages={pages}"

    if raw_df.empty:
        raise SystemExit("No data provided; adjust query/page count or dataset.")

    df = prepare_dataframe(raw_df)
    if df.empty:
        raise SystemExit("No priced listings available for training.")

    X = df[["text", "timestamp_val"]]
    y = df["price_amount"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipeline = build_pipeline(model_type=model_type)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    joblib.dump(pipeline, output)

    print(
        f"Source: {source}\n"
        f"Model: {model_type} | Trained on {len(X_train)} samples, evaluated on {len(X_test)} samples.\n"
        f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f}\n"
        f"Model saved to {output}"
    )

    # Save holdout set for later, to ensure evaluations stay out-of-sample.
    holdout = df.loc[X_test.index].copy()
    holdout["prediction"] = preds
    holdout_path = f"{output}.holdout.csv"
    holdout.to_csv(holdout_path, index=False)
    print(f"Saved holdout with predictions to {holdout_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a price prediction model from FINN listings."
    )
    parser.add_argument("--query", default="gitar", help="Search query string.")
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Number of pages to fetch for training (approx. 50 items per page).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation.",
    )
    parser.add_argument(
        "--output",
        default="price_model.joblib",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Optional CSV to train on (skips scraping). If it lacks engineered columns, they will be created.",
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "xgb"],
        default="ridge",
        help="Choose estimator: linear ridge or xgboost.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_price_model(
        query=args.query,
        pages=args.pages,
        test_size=args.test_size,
        output=args.output,
        model_type=args.model_type,
        data_csv=args.data_csv,
    )


if __name__ == "__main__":
    main()
