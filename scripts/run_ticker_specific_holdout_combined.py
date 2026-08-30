from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    build_ablation_feature_sets,
    build_logistic_model,
    split_holdout,
    tune_model,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
TARGETS = ["target_5d_excess_gt_0", "target_3d_excess_gt_0"]
FEATURE_SET_NAMES = [
    "price_only",
    "market_context_only",
    "news_all_only",
    "price_quality",
    "price_news_events",
    "price_news_quality",
]


def safe_metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    row["roc_auc"] = roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan
    return row


def fit_selected_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]):
    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[target].astype(int)
    if selected["selected_model"] == "logistic_balanced":
        model = build_logistic_model()
    else:
        model = build_boosting_model(selected["selected_model"], selected["selected_params"], float(y_train_full.mean()))
    model.fit(train_full[feature_cols], y_train_full)
    return model, train_full, y_train_full


def run_one_ticker(
    df: pd.DataFrame,
    ticker: str,
    target: str,
    feature_set_name: str,
    feature_cols: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scoped = df[(df["ticker"] == ticker) & (df["all_days"] == 1)].copy()
    train, val, test = split_holdout(scoped)
    y_test = test[target].astype(int)

    if len(train) < 80 or len(val) < 25 or len(test) < 25 or y_test.nunique() < 2:
        return [], []

    selected = tune_model(train, val, feature_cols, target)
    if selected is None:
        return [], []

    model, train_full, y_train_full = fit_selected_model(train, val, feature_cols, target, selected)
    test_proba = model.predict_proba(test[feature_cols])[:, 1]

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for threshold_mode, threshold in [("validation_tuned", selected["selected_threshold"]), ("default_0.5", 0.5)]:
        test_pred = (test_proba >= threshold).astype(int)
        metrics = safe_metric_row(y_test, test_pred, test_proba)
        result_rows.append(
            {
                "ticker": ticker,
                "target": target,
                "feature_set": feature_set_name,
                "feature_count": len(feature_cols),
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "train_rows": len(train),
                "validation_rows": len(val),
                "train_plus_validation_rows": len(train_full),
                "test_rows": len(test),
                "train_positive_rate": float(train[target].mean()),
                "validation_positive_rate": float(val[target].mean()),
                "test_positive_rate": float(y_test.mean()),
                "test_majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                **selected,
                **metrics,
            }
        )
        prediction_rows.extend(
            {
                "ticker": ticker,
                "trading_date": trading_date,
                "target": target,
                "feature_set": feature_set_name,
                "threshold_mode": threshold_mode,
                "y_true": int(actual),
                "proba": float(proba),
                "pred": int(pred),
                "selected_model": selected["selected_model"],
                "selected_threshold": float(threshold),
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
            }
            for trading_date, actual, proba, pred in zip(test["trading_date"], y_test, test_proba, test_pred)
        )

    return result_rows, prediction_rows


def build_combined_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["target", "feature_set", "threshold_mode"]
    for keys, group in predictions.groupby(group_cols):
        target, feature_set, threshold_mode = keys
        y_true = group["y_true"].astype(int)
        proba = group["proba"].astype(float).to_numpy()
        pred = group["pred"].astype(int).to_numpy()
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "threshold_mode": threshold_mode,
                "ticker_count": group["ticker"].nunique(),
                "test_rows": len(group),
                "test_positive_rate": float(y_true.mean()),
                "test_majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
                **safe_metric_row(y_true, pred, proba),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "threshold_mode", "roc_auc", "balanced_accuracy"], ascending=[True, True, False, False])


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    all_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for target in TARGETS:
        for ticker in TICKERS:
            for feature_set_name in FEATURE_SET_NAMES:
                features = [col for col in feature_sets.get(feature_set_name, []) if col in df.columns]
                if not features:
                    continue
                result_rows, prediction_rows = run_one_ticker(df, ticker, target, feature_set_name, features)
                all_results.extend(result_rows)
                all_predictions.extend(prediction_rows)

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)
    combined = build_combined_metrics(predictions) if len(predictions) else pd.DataFrame()

    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "tickers": ", ".join(TICKERS),
                "targets": ", ".join(TARGETS),
                "feature_sets": ", ".join(FEATURE_SET_NAMES),
                "modelling_design": "Train one model per ticker on 2020-2022, tune on 2023, refit on 2020-2023, pool 2024 predictions across tickers.",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"ticker_specific_holdout_results{suffix}.csv", index=False)
    combined.to_csv(paths.tables_dir / f"ticker_specific_holdout_combined{suffix}.csv", index=False)
    predictions.to_csv(paths.tables_dir / f"ticker_specific_holdout_predictions{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"ticker_specific_holdout_metadata{suffix}.csv", index=False)
    return {"results": results, "combined": combined, "predictions": predictions, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ticker-specific strict holdout models and pool 2024 predictions.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nCombined all-ticker pooled metrics")
    print(outputs["combined"].to_string(index=False))
    print("\nBest per-ticker rows by ROC AUC")
    best = (
        outputs["results"]
        .sort_values(["ticker", "target", "threshold_mode", "roc_auc", "balanced_accuracy"], ascending=[True, True, True, False, False])
        .groupby(["ticker", "target", "threshold_mode"], as_index=False)
        .head(3)
    )
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
