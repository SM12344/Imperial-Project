from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_modelling_baselines import (
    FEATURE_SETS,
    build_paths,
    build_random_forest_pipeline,
    evaluate_predictions,
    load_dataset,
)


DEFAULT_TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "NVDA"]


def time_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for a stable time-based split.")

    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_idx]

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train_df, test_df, cutoff_date


def make_experiment_slices(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    slices: dict[str, pd.DataFrame] = {
        "pooled_all_days": df.copy(),
        "pooled_news_days": df[df["has_news"] == 1].copy(),
    }
    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].copy()
        slices[f"{ticker}_all_days"] = ticker_df
        slices[f"{ticker}_news_days"] = ticker_df[ticker_df["has_news"] == 1].copy()
    return slices


def run_random_forest_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set_name]
    pipeline = build_random_forest_pipeline(feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_next_day_up"]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)

    result_row = {
        "feature_set": feature_set_name,
        "model_name": "random_forest",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["trading_date"].min().date().isoformat(),
        "train_end": train_df["trading_date"].max().date().isoformat(),
        "test_start": test_df["trading_date"].min().date().isoformat(),
        "test_end": test_df["trading_date"].max().date().isoformat(),
        **metrics,
    }

    predictions = test_df[["ticker", "trading_date", "target_next_day_up", "has_news"]].copy()
    predictions["feature_set"] = feature_set_name
    predictions["model_name"] = "random_forest"
    predictions["predicted_up"] = pred
    predictions["predicted_probability_up"] = proba
    predictions["correct_prediction"] = (
        predictions["predicted_up"] == predictions["target_next_day_up"]
    ).astype(int)
    return result_row, predictions


def run_slice(
    slice_name: str,
    df: pd.DataFrame,
    test_fraction: float,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], dict[str, Any] | None]:
    if df.empty or df["trading_date"].nunique() < 10:
        return [], [], None

    train_df, test_df, cutoff_date = time_split(df, test_fraction=test_fraction)
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for feature_set_name in FEATURE_SETS:
        row, preds = run_random_forest_experiment(train_df, test_df, feature_set_name)
        row["slice_name"] = slice_name
        row["tickers_in_slice"] = ", ".join(sorted(df["ticker"].drop_duplicates()))
        row["news_only_slice"] = int((df["has_news"] == 1).all())
        rows.append(row)

        preds["slice_name"] = slice_name
        prediction_frames.append(preds)

    metadata = {
        "slice_name": slice_name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff_date.date().isoformat(),
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "unique_dates": df["trading_date"].nunique(),
        "train_unique_dates": train_df["trading_date"].nunique(),
        "test_unique_dates": test_df["trading_date"].nunique(),
        "has_news_rows": int(df["has_news"].sum()),
        "up_rate": float(df["target_next_day_up"].mean()),
    }
    return rows, prediction_frames, metadata


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    tickers: list[str] | None = None,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    selected_tickers = tickers or DEFAULT_TICKERS
    slices = make_experiment_slices(df, selected_tickers)

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for slice_name, slice_df in slices.items():
        rows, pred_frames, metadata = run_slice(slice_name, slice_df, test_fraction)
        if metadata is None:
            skipped_rows.append(
                {
                    "slice_name": slice_name,
                    "rows": len(slice_df),
                    "unique_dates": int(slice_df["trading_date"].nunique()),
                    "reason": "insufficient_unique_dates_or_empty",
                }
            )
            continue
        results.extend(rows)
        predictions.extend(pred_frames)
        split_rows.append(metadata)

    results_df = pd.DataFrame(results).sort_values(
        ["slice_name", "roc_auc", "f1", "accuracy"], ascending=[True, False, False, False]
    )
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    split_df = pd.DataFrame(split_rows).sort_values("slice_name")
    skipped_df = pd.DataFrame(skipped_rows).sort_values("slice_name") if skipped_rows else pd.DataFrame()

    results_df.to_csv(paths.tables_dir / "ticker_specific_results.csv", index=False)
    predictions_df.to_csv(paths.processed_dir / "ticker_specific_predictions.csv", index=False)
    split_df.to_csv(paths.tables_dir / "ticker_specific_split_metadata.csv", index=False)
    skipped_df.to_csv(paths.tables_dir / "ticker_specific_skipped_slices.csv", index=False)

    return {
        "dataset": df,
        "results_df": results_df,
        "predictions_df": predictions_df,
        "split_df": split_df,
        "skipped_df": skipped_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ticker-specific and news-days-only random-forest comparisons."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project folder containing data/processed and outputs/tables. Defaults to the current directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default="model_dataset_finbert_complete.csv",
        help="Input dataset filename under data/processed.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers to evaluate individually.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique trading dates reserved for the time-ordered test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        tickers=args.tickers,
        test_fraction=args.test_fraction,
    )
    print(outputs["split_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))
    if not outputs["skipped_df"].empty:
        print()
        print(outputs["skipped_df"].to_string(index=False))


if __name__ == "__main__":
    main()
