from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from run_modelling_baselines import (
    FEATURE_SETS,
    build_paths,
    build_random_forest_pipeline,
    evaluate_predictions,
    load_dataset,
)


TARGET_THRESHOLDS = [0.0, 0.005, 0.01]


def build_tsla_news_days_with_return(df: pd.DataFrame) -> pd.DataFrame:
    tsla = df[df["ticker"] == "TSLA"].copy().sort_values("trading_date")
    tsla["next_day_return"] = tsla.groupby("ticker")["return_1d"].shift(-1)
    tsla = tsla[(tsla["has_news"] == 1) & tsla["next_day_return"].notna()].copy()
    if tsla.empty:
        raise ValueError("No TSLA news-day rows with next-day return available.")
    return tsla.reset_index(drop=True)


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


def run_one(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_set_name: str, threshold: float) -> dict:
    feature_cols = FEATURE_SETS[feature_set_name]
    train = train_df.copy()
    test = test_df.copy()
    train["target_custom"] = (train["next_day_return"] > threshold).astype(int)
    test["target_custom"] = (test["next_day_return"] > threshold).astype(int)

    pipeline = build_random_forest_pipeline(feature_cols)
    pipeline.fit(train[feature_cols], train["target_custom"])
    pred = pipeline.predict(test[feature_cols])
    proba = pipeline.predict_proba(test[feature_cols])[:, 1]
    metrics = evaluate_predictions(test["target_custom"], pred, proba)

    return {
        "target_threshold": threshold,
        "feature_set": feature_set_name,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positive_rate": float(test["target_custom"].mean()),
        "always_positive_accuracy": float(test["target_custom"].mean()),
        **metrics,
    }


def run_pipeline(project_root: str | None = None, dataset_name: str = "model_dataset_finbert_complete.csv", test_fraction: float = 0.2):
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    tsla = build_tsla_news_days_with_return(df)
    train_df, test_df, cutoff_date = time_split(tsla, test_fraction)

    rows = []
    for threshold in TARGET_THRESHOLDS:
        for feature_set_name in FEATURE_SETS:
            rows.append(run_one(train_df, test_df, feature_set_name, threshold))

    results_df = pd.DataFrame(rows).sort_values(["target_threshold", "roc_auc"], ascending=[True, False])
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
    results_df.to_csv(paths.tables_dir / "target_redesign_results.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "target_redesign_metadata.csv", index=False)
    return {"results_df": results_df, "metadata_df": metadata_df}


def parse_args():
    parser = argparse.ArgumentParser(description="Run target redesign experiments for TSLA news days.")
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
