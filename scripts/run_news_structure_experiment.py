from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from run_modelling_baselines import (
    PRICE_ONLY_FEATURES,
    build_paths,
    build_random_forest_pipeline,
    evaluate_predictions,
    load_dataset,
)


SENTIMENT_BUNDLES = {
    "structure_only": [
        "news_count",
        "log_news_count",
        "finbert_positive_std",
        "finbert_negative_std",
        "finbert_neutral_std",
        "finbert_sentiment_score_std",
        "finbert_positive_max",
        "finbert_negative_max",
        "finbert_neutral_max",
        "finbert_sentiment_score_max",
        "finbert_sentiment_score_min",
        "finbert_sentiment_score_surprise",
        "finbert_positive_mean_surprise",
        "finbert_negative_mean_surprise",
        "news_count_rolling5",
    ],
    "levels_and_lags": [
        "finbert_positive_mean",
        "finbert_negative_mean",
        "finbert_neutral_mean",
        "finbert_sentiment_score_mean",
        "finbert_positive_mean_lag1",
        "finbert_negative_mean_lag1",
        "finbert_neutral_mean_lag1",
        "finbert_sentiment_score_lag1",
        "finbert_positive_mean_rolling5",
        "finbert_negative_mean_rolling5",
        "finbert_neutral_mean_rolling5",
        "finbert_sentiment_score_rolling5",
    ],
}


def build_tsla_news_days_slice(df: pd.DataFrame) -> pd.DataFrame:
    tsla = df[(df["ticker"] == "TSLA") & (df["has_news"] == 1)].copy()
    if tsla.empty:
        raise ValueError("No TSLA news-day rows found.")
    return tsla.sort_values("trading_date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_idx]
    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train_df, test_df, cutoff_date


def run_one(train_df: pd.DataFrame, test_df: pd.DataFrame, bundle_name: str, feature_cols: list[str]) -> dict:
    pipeline = build_random_forest_pipeline(feature_cols)
    pipeline.fit(train_df[feature_cols], train_df["target_next_day_up"])
    pred = pipeline.predict(test_df[feature_cols])
    proba = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    metrics = evaluate_predictions(test_df["target_next_day_up"], pred, proba)
    return {
        "bundle_name": bundle_name,
        "feature_count": len(feature_cols),
        "always_up_accuracy": float(test_df["target_next_day_up"].mean()),
        **metrics,
    }


def run_pipeline(project_root: str | None = None, dataset_name: str = "model_dataset_finbert_complete.csv", test_fraction: float = 0.2):
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    tsla = build_tsla_news_days_slice(df)
    train_df, test_df, cutoff_date = time_split(tsla, test_fraction)

    configs = {
        "price_only": PRICE_ONLY_FEATURES,
        "price_plus_structure_only": PRICE_ONLY_FEATURES + SENTIMENT_BUNDLES["structure_only"],
        "price_plus_levels_and_lags": PRICE_ONLY_FEATURES + SENTIMENT_BUNDLES["levels_and_lags"],
        "price_plus_all_sentiment": PRICE_ONLY_FEATURES + SENTIMENT_BUNDLES["structure_only"] + SENTIMENT_BUNDLES["levels_and_lags"],
    }

    rows = [run_one(train_df, test_df, name, cols) for name, cols in configs.items()]
    results_df = pd.DataFrame(rows).sort_values(["roc_auc", "f1", "accuracy"], ascending=[False, False, False])
    metadata_df = pd.DataFrame(
        [{
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cutoff_date": cutoff_date.date().isoformat(),
            "rows": len(tsla),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "unique_dates": tsla["trading_date"].nunique(),
            "test_fraction": test_fraction,
        }]
    )
    results_df.to_csv(paths.tables_dir / "news_structure_results.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "news_structure_metadata.csv", index=False)
    return {"results_df": results_df, "metadata_df": metadata_df}


def parse_args():
    parser = argparse.ArgumentParser(description="Run news-structure feature comparisons for TSLA news days.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.test_fraction)
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
