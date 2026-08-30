from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from run_2024_holdout_ablation import (
    add_holdout_filter_columns,
    best_threshold,
    build_ablation_feature_sets,
    split_holdout,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model


TARGET = "target_5d_excess_gt_0"
FEATURE_SET = "price_quality"
MODEL_NAME = "lightgbm"
BEST_AMZN_PARAMS = {
    "n_estimators": 80,
    "max_depth": 1,
    "num_leaves": 3,
    "learning_rate": 0.03,
    "min_child_samples": 25,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
}


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def add_ticker_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    tickers = sorted(out["ticker"].dropna().unique())
    cols = []
    for ticker in tickers:
        col = f"ticker_is_{ticker.lower()}"
        out[col] = (out["ticker"] == ticker).astype(int)
        cols.append(col)
    return out, cols


def evaluate_variant(df: pd.DataFrame, feature_cols: list[str], variant: str) -> tuple[dict[str, Any], pd.DataFrame]:
    train, val, test = split_holdout(df)
    y_train = train[TARGET].astype(int)
    y_val = val[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    model = build_model(MODEL_NAME, BEST_AMZN_PARAMS, float(y_train.mean()))
    model.fit(train[feature_cols], y_train)
    val_proba = model.predict_proba(val[feature_cols])[:, 1]
    threshold, validation_balanced_accuracy = best_threshold(y_val, val_proba)

    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[TARGET].astype(int)
    final_model = build_model(MODEL_NAME, BEST_AMZN_PARAMS, float(y_train_full.mean()))
    final_model.fit(train_full[feature_cols], y_train_full)
    test_proba = final_model.predict_proba(test[feature_cols])[:, 1]

    summary = {
        "variant": variant,
        "model": MODEL_NAME,
        "params": BEST_AMZN_PARAMS,
        "target": TARGET,
        "feature_set": FEATURE_SET,
        "feature_count": len(feature_cols),
        "threshold": threshold,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        "validation_roc_auc": roc_auc_score(y_val, val_proba),
        "validation_balanced_accuracy_at_threshold": validation_balanced_accuracy,
        **metrics(y_test, test_proba, threshold),
    }

    ticker_rows = []
    pred = (test_proba >= threshold).astype(int)
    test_eval = test[["ticker"]].copy()
    test_eval["y_true"] = y_test.to_numpy()
    test_eval["proba"] = test_proba
    test_eval["pred"] = pred
    for ticker, part in test_eval.groupby("ticker"):
        if part["y_true"].nunique() < 2:
            continue
        ticker_rows.append(
            {
                "variant": variant,
                "ticker": ticker,
                "rows": len(part),
                "positive_rate": float(part["y_true"].mean()),
                "majority_baseline_accuracy": float(max(part["y_true"].mean(), 1 - part["y_true"].mean())),
                "accuracy": accuracy_score(part["y_true"], part["pred"]),
                "balanced_accuracy": balanced_accuracy_score(part["y_true"], part["pred"]),
                "roc_auc": roc_auc_score(part["y_true"], part["proba"]),
                "precision": precision_score(part["y_true"], part["pred"], zero_division=0),
                "recall": recall_score(part["y_true"], part["pred"], zero_division=0),
                "f1": f1_score(part["y_true"], part["pred"], zero_division=0),
            }
        )
    return summary, pd.DataFrame(ticker_rows)


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_cols = [col for col in build_ablation_feature_sets(df)[FEATURE_SET] if col in df.columns]

    rows = []
    ticker_parts = []
    summary, ticker_df = evaluate_variant(df, feature_cols, "pooled_without_ticker_flags")
    rows.append(summary)
    ticker_parts.append(ticker_df)

    ticker_df_input, ticker_cols = add_ticker_flags(df)
    summary_flags, ticker_df_flags = evaluate_variant(ticker_df_input, feature_cols + ticker_cols, "pooled_with_ticker_flags")
    rows.append(summary_flags)
    ticker_parts.append(ticker_df_flags)

    summary_df = pd.DataFrame(rows)
    ticker_summary = pd.concat(ticker_parts, ignore_index=True)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": "2020-01-01 to 2022-12-31",
                "validation_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                "model_source": "AMZN best holdout model class and parameters",
                "test_policy": "2024 is untouched; threshold selected on 2023 validation only",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    summary_df.to_csv(paths.tables_dir / f"best_model_all_tickers_summary{suffix}.csv", index=False)
    ticker_summary.to_csv(paths.tables_dir / f"best_model_all_tickers_by_ticker{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"best_model_all_tickers_metadata{suffix}.csv", index=False)
    return {"summary": summary_df, "by_ticker": ticker_summary, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best AMZN model setup on pooled all-ticker data.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print("Summary")
    print(outputs["summary"].to_string(index=False))
    print("\nBy ticker")
    print(outputs["by_ticker"].to_string(index=False))


if __name__ == "__main__":
    main()
