from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_event_target_experiments import (
    CORE_NEWS_FEATURES,
    CORE_PRICE_FEATURES,
    ENGINEERED_NEWS_FEATURES,
    EVENT_PATTERNS,
    build_event_daily_features,
    category_feature_names,
    load_scored_news,
)
from run_high_signal_event_experiments import QUALITY_NEWS_FEATURES, add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import CATBOOST_GRID, LIGHTGBM_GRID, XGBOOST_GRID, build_model as build_boosting_model


TRAIN_END = "2023-01-01"
VALIDATION_START = "2023-01-01"
TEST_START = "2024-01-01"
TEST_END = "2025-01-01"

SCENARIOS = [
    {
        "scenario": "pooled_all_3d_abs_move",
        "scope": "pooled_all_tickers",
        "filter": "all_days",
        "target": "target_3d_abs_excess_gt_1pct",
    },
    {
        "scenario": "pooled_high_news_3d_direction",
        "scope": "pooled_all_tickers",
        "filter": "top_10pct_news_volume",
        "target": "target_3d_excess_gt_0",
    },
    {
        "scenario": "pooled_strong_sentiment_5d_abs_move",
        "scope": "pooled_all_tickers",
        "filter": "strong_abs_sentiment",
        "target": "target_5d_abs_excess_gt_1pct",
    },
    {
        "scenario": "amzn_all_5d_direction",
        "scope": "AMZN",
        "filter": "all_days",
        "target": "target_5d_excess_gt_0",
    },
    {
        "scenario": "nvda_earnings_3d_direction",
        "scope": "NVDA",
        "filter": "earnings_high_news",
        "target": "target_3d_excess_gt_0",
    },
    {
        "scenario": "nvda_earnings_5d_direction",
        "scope": "NVDA",
        "filter": "earnings_high_news",
        "target": "target_5d_excess_gt_0",
    },
    {
        "scenario": "nvda_all_3d_direction",
        "scope": "NVDA",
        "filter": "all_days",
        "target": "target_3d_excess_gt_0",
    },
    {
        "scenario": "nvda_all_5d_direction",
        "scope": "NVDA",
        "filter": "all_days",
        "target": "target_5d_excess_gt_0",
    },
    {
        "scenario": "tsla_analyst_earnings_5d_abs_move",
        "scope": "TSLA",
        "filter": "analyst_or_earnings",
        "target": "target_5d_abs_excess_gt_1pct",
    },
]


STOCK_FINBERT_FEATURES = [
    "news_count",
    "log_news_count",
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score_mean",
    "finbert_sentiment_score_lag1",
    "finbert_sentiment_score_rolling5",
    "finbert_sentiment_score_surprise",
]

MARKET_CONTEXT_FEATURES = [
    "market_news_count",
    "market_log_news_count",
    "market_finbert_positive_mean",
    "market_finbert_negative_mean",
    "market_finbert_neutral_mean",
    "market_finbert_sentiment_score_mean",
    "market_finbert_sentiment_score_lag1",
    "market_finbert_sentiment_score_rolling5",
    "market_finbert_sentiment_score_surprise",
    "market_abs_sentiment_score",
]

INTERACTION_FEATURES = [
    "sentiment_intensity",
    "abs_sentiment_score",
    "sentiment_weighted_news_count",
    "negative_weighted_news_count",
    "positive_weighted_news_count",
    "stock_minus_market_sentiment",
    "stock_sentiment_x_market_sentiment",
]


def build_ablation_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    event_cols = [col for col in category_feature_names() if col in df.columns]
    quality_cols = [col for col in QUALITY_NEWS_FEATURES if col in df.columns]
    stock_cols = [col for col in STOCK_FINBERT_FEATURES if col in df.columns]
    market_cols = [col for col in MARKET_CONTEXT_FEATURES if col in df.columns]
    interaction_cols = [col for col in INTERACTION_FEATURES if col in df.columns]
    price_cols = [col for col in CORE_PRICE_FEATURES if col in df.columns]

    return {
        "price_only": price_cols,
        "stock_finbert_only": stock_cols,
        "market_context_only": market_cols,
        "events_only": event_cols,
        "quality_only": quality_cols,
        "news_all_only": stock_cols + market_cols + interaction_cols + event_cols + quality_cols,
        "price_stock_finbert": price_cols + stock_cols,
        "price_market_context": price_cols + market_cols,
        "price_events": price_cols + event_cols,
        "price_quality": price_cols + quality_cols,
        "price_news_events": price_cols + stock_cols + market_cols + interaction_cols + event_cols,
        "price_news_quality": price_cols + stock_cols + market_cols + interaction_cols + event_cols + quality_cols,
    }


def add_holdout_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reference = out[out["trading_date"] < pd.Timestamp(TEST_START)]
    q = reference[["news_count", "market_news_count", "abs_sentiment_score", "sentiment_intensity"]].quantile([0.75, 0.90, 0.95])
    out["all_days"] = 1
    out["top_10pct_news_volume"] = (out["news_count"] >= q.loc[0.90, "news_count"]).astype(int)
    out["top_5pct_news_volume"] = (out["news_count"] >= q.loc[0.95, "news_count"]).astype(int)
    out["top_10pct_market_news"] = (out["market_news_count"] >= q.loc[0.90, "market_news_count"]).astype(int)
    out["strong_abs_sentiment"] = (out["abs_sentiment_score"] >= q.loc[0.90, "abs_sentiment_score"]).astype(int)
    out["high_sentiment_intensity"] = (out["sentiment_intensity"] >= q.loc[0.90, "sentiment_intensity"]).astype(int)
    out["earnings_high_news"] = ((out["earnings_count"] > 0) & (out["news_count"] >= q.loc[0.75, "news_count"])).astype(int)
    out["analyst_or_earnings"] = ((out["analyst_count"] > 0) | (out["earnings_count"] > 0)).astype(int)
    out["macro_high_market"] = ((out["macro_count"] > 0) | (out["market_news_count"] >= q.loc[0.90, "market_news_count"])).astype(int)
    return out


def scenario_frame(df: pd.DataFrame, scenario: dict[str, str]) -> pd.DataFrame:
    scoped = df.copy() if scenario["scope"] == "pooled_all_tickers" else df[df["ticker"] == scenario["scope"]].copy()
    return scoped[scoped[scenario["filter"]] == 1].copy()


def split_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["trading_date"] < pd.Timestamp(TRAIN_END)].copy()
    val = df[(df["trading_date"] >= pd.Timestamp(VALIDATION_START)) & (df["trading_date"] < pd.Timestamp(TEST_START))].copy()
    test = df[(df["trading_date"] >= pd.Timestamp(TEST_START)) & (df["trading_date"] < pd.Timestamp(TEST_END))].copy()
    return train, val, test


def metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_balanced = -np.inf
    for threshold in np.linspace(0.2, 0.8, 25):
        score = balanced_accuracy_score(y_true, (proba >= threshold).astype(int))
        if score > best_balanced:
            best_balanced = score
            best_t = float(threshold)
    return best_t, float(best_balanced)


def build_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )


def candidate_models(positive_rate: float) -> list[tuple[str, dict[str, Any], Pipeline]]:
    models: list[tuple[str, dict[str, Any], Pipeline]] = [("logistic_balanced", {}, build_logistic_model())]
    for params in LIGHTGBM_GRID:
        models.append(("lightgbm", params, build_boosting_model("lightgbm", params, positive_rate)))
    for params in XGBOOST_GRID:
        models.append(("xgboost", params, build_boosting_model("xgboost", params, positive_rate)))
    for params in CATBOOST_GRID:
        models.append(("catboost", params, build_boosting_model("catboost", params, positive_rate)))
    return models


def tune_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str) -> dict[str, Any] | None:
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    if len(train) < 80 or len(val) < 25 or y_train.nunique() < 2 or y_val.nunique() < 2:
        return None

    best: dict[str, Any] | None = None
    for model_name, params, model in candidate_models(float(y_train.mean())):
        model.fit(train[feature_cols], y_train)
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        threshold, val_balanced_at_threshold = best_threshold(y_val, val_proba)
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "selected_threshold": threshold,
            "validation_roc_auc": float(val_auc),
            "validation_balanced_accuracy_at_threshold": val_balanced_at_threshold,
            "validation_accuracy_at_threshold": accuracy_score(y_val, (val_proba >= threshold).astype(int)),
        }
        if best is None or (row["validation_roc_auc"], row["validation_balanced_accuracy_at_threshold"]) > (
            best["validation_roc_auc"],
            best["validation_balanced_accuracy_at_threshold"],
        ):
            best = row
    return best


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        sdf = scenario_frame(df, scenario)
        train, val, test = split_holdout(sdf)
        coverage_rows.append(
            {
                **scenario,
                "total_rows": len(sdf),
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "total_unique_dates": sdf["trading_date"].nunique(),
                "train_unique_dates": train["trading_date"].nunique(),
                "validation_unique_dates": val["trading_date"].nunique(),
                "test_unique_dates": test["trading_date"].nunique(),
                "train_positive_rate": train[scenario["target"]].mean() if len(train) else np.nan,
                "validation_positive_rate": val[scenario["target"]].mean() if len(val) else np.nan,
                "test_positive_rate": test[scenario["target"]].mean() if len(test) else np.nan,
                "test_majority_baseline_accuracy": max(test[scenario["target"]].mean(), 1 - test[scenario["target"]].mean()) if len(test) else np.nan,
            }
        )
        y_test = test[scenario["target"]].astype(int)
        if len(test) < 25 or y_test.nunique() < 2:
            continue
        for feature_set_name, features in feature_sets.items():
            feature_cols = [col for col in features if col in train.columns]
            if not feature_cols:
                continue
            selected = tune_model(train, val, feature_cols, scenario["target"])
            if selected is None:
                continue
            y_train_full = pd.concat([train[scenario["target"]], val[scenario["target"]]]).astype(int)
            train_full = pd.concat([train, val], ignore_index=True)
            model = build_logistic_model() if selected["selected_model"] == "logistic_balanced" else build_boosting_model(
                selected["selected_model"],
                selected["selected_params"],
                float(y_train_full.mean()),
            )
            model.fit(train_full[feature_cols], y_train_full)
            test_proba = model.predict_proba(test[feature_cols])[:, 1]
            for threshold_mode, threshold in [("validation_tuned", selected["selected_threshold"]), ("default_0.5", 0.5)]:
                test_pred = (test_proba >= threshold).astype(int)
                rows.append(
                    {
                        **scenario,
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "threshold_mode": threshold_mode,
                        "threshold": threshold,
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "train_plus_validation_rows": len(train_full),
                        "test_rows": len(test),
                        "test_positive_rate": float(y_test.mean()),
                        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                        **selected,
                        **metric_row(y_test, test_pred, test_proba),
                    }
                )

    results = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    best = results.sort_values(["scenario", "roc_auc", "balanced_accuracy"], ascending=[True, False, False]).groupby("scenario", as_index=False).head(8)
    price = (
        results[results["feature_set"] == "price_only"]
        .sort_values(["scenario", "threshold_mode", "roc_auc"], ascending=[True, True, False])
        .groupby(["scenario", "threshold_mode"], as_index=False)
        .head(1)
    )
    news = (
        results[results["feature_set"] != "price_only"]
        .sort_values(["scenario", "threshold_mode", "roc_auc"], ascending=[True, True, False])
        .groupby(["scenario", "threshold_mode"], as_index=False)
        .head(1)
    )
    comparison = price.merge(news, on=["scenario", "threshold_mode"], suffixes=("_price", "_news"))
    comparison["roc_auc_delta_news_minus_price"] = comparison["roc_auc_news"] - comparison["roc_auc_price"]
    comparison["balanced_accuracy_delta_news_minus_price"] = comparison["balanced_accuracy_news"] - comparison["balanced_accuracy_price"]
    comparison["accuracy_delta_news_minus_price"] = comparison["accuracy_news"] - comparison["accuracy_price"]

    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": "2020-01-01 to 2022-12-31",
                "validation_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                "models_tuned": "logistic_balanced, lightgbm, xgboost, catboost",
                "feature_sets": ", ".join(feature_sets),
                "filter_threshold_reference": "2020-2023 only",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"holdout_2024_ablation_results{suffix}.csv", index=False)
    best.to_csv(paths.tables_dir / f"holdout_2024_ablation_best{suffix}.csv", index=False)
    comparison.to_csv(paths.tables_dir / f"holdout_2024_ablation_news_vs_price{suffix}.csv", index=False)
    coverage.to_csv(paths.tables_dir / f"holdout_2024_ablation_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"holdout_2024_ablation_metadata{suffix}.csv", index=False)
    return {"results": results, "best": best, "comparison": comparison, "coverage": coverage, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2024 holdout feature ablation with validation-only tuning.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nCoverage")
    print(outputs["coverage"].to_string(index=False))
    print("\nBest rows")
    print(outputs["best"].head(60).to_string(index=False))
    print("\nNews vs price")
    print(outputs["comparison"].sort_values("roc_auc_delta_news_minus_price", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
