import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from train_price_model import fetch_listings, prepare_dataframe


def load_model(path: str):
    return joblib.load(path)


def evaluate_model(
    model_path: str,
    query: str,
    pages: int,
    samples: int = 5,
    dataset_csv: str | None = None,
):
    model = load_model(model_path)

    if dataset_csv:
        df_loaded = pd.read_csv(dataset_csv)
        if {"text", "timestamp_val", "price_amount"}.issubset(df_loaded.columns):
            df = df_loaded
        else:
            df = prepare_dataframe(df_loaded)
        source = f"from {dataset_csv}"
    else:
        df_raw = fetch_listings(query, pages)
        df = prepare_dataframe(df_raw)
        source = f"fresh scrape: query='{query}', pages={pages}"

    if df.empty:
        raise SystemExit("No data available for evaluation.")

    X = df[["text", "timestamp_val"]]
    y = df["price_amount"].astype(float)

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    print(
        f"Eval on {len(df)} samples ({source}) | MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R^2: {r2:,.3f}"
    )

    # Show a few sample predictions
    sample_df = df.copy()
    sample_df["prediction"] = preds
    print("\nSample predictions:")
    cols = ["price_amount", "prediction", "title", "location"]
    print(sample_df[cols].head(samples).to_string(index=False))

    # Feature importance (top tokens)
    pre = model.named_steps["preprocess"]
    feature_names = pre.get_feature_names_out()
    model_step = model.named_steps["model"]

    if hasattr(model_step, "coef_"):
        importances = model_step.coef_
        label = "coefficient"
    elif hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        label = "feature importance"
    else:
        print("\nModel does not expose coefficients or feature importances.")
        return

    # Extract time weight
    time_idx = np.where(feature_names == "time__timestamp_val")[0]
    if len(time_idx):
        time_weight = importances[time_idx[0]]
        print(f"\nTimestamp {label} (scaled): {time_weight:.4f}")

    # Text feature importance
    text_mask = np.char.startswith(feature_names.astype(str), "text__")
    text_features = feature_names[text_mask]
    text_weights = importances[text_mask]

    top_pos_idx = np.argsort(text_weights)[-10:][::-1]
    top_neg_idx = np.argsort(text_weights)[:10]

    print("\nTop positive tokens:")
    for idx in top_pos_idx:
        token = text_features[idx].replace("text__", "")
        print(f"  {token:20s} {text_weights[idx]:.4f}")

    print("\nTop negative tokens:")
    for idx in top_neg_idx:
        token = text_features[idx].replace("text__", "")
        print(f"  {token:20s} {text_weights[idx]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained price model and inspect feature weights."
    )
    parser.add_argument(
        "--model",
        default="price_model.joblib",
        help="Path to trained model file.",
    )
    parser.add_argument("--query", default="gitar", help="Search query to evaluate on.")
    parser.add_argument(
        "--pages",
        type=int,
        default=10,
        help="Pages to fetch for evaluation (approx. 50 items per page).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="How many sample predictions to print.",
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional CSV to evaluate (e.g. generated holdout); skips scraping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_path=args.model,
        query=args.query,
        pages=args.pages,
        samples=args.samples,
        dataset_csv=args.dataset_csv,
    )


if __name__ == "__main__":
    main()
