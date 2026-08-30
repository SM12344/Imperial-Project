from __future__ import annotations

import argparse
import re
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
from run_modelling_baselines import build_paths, load_dataset

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


FEATURE_SETS = {
    "price_only": CORE_PRICE_FEATURES,
    "news_only": CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names(),
    "price_news": CORE_PRICE_FEATURES + CORE_NEWS_FEATURES,
    "price_news_events": CORE_PRICE_FEATURES + CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names(),
}

QUALITY_NEWS_FEATURES = [
    "quality_unique_story_count",
    "quality_duplicate_article_ratio",
    "quality_source_weight_mean",
    "quality_source_weighted_sentiment_mean",
    "quality_source_weighted_positive_mean",
    "quality_source_weighted_negative_mean",
    "quality_high_conf_positive_count",
    "quality_high_conf_negative_count",
    "quality_high_conf_neutral_count",
    "quality_high_conf_net_count",
    "quality_high_conf_total_count",
    "quality_high_conf_positive_share",
    "quality_high_conf_negative_share",
    "quality_positive_article_count",
    "quality_negative_article_count",
    "quality_neutral_article_count",
    "quality_positive_negative_ratio",
    "quality_max_positive_confidence",
    "quality_max_negative_confidence",
    "quality_sentiment_disagreement",
    "quality_dedup_sentiment_mean",
    "quality_dedup_positive_mean",
    "quality_dedup_negative_mean",
    "quality_source_weighted_sentiment_mean_lag1",
    "quality_source_weighted_sentiment_mean_rolling5",
    "quality_high_conf_net_count_lag1",
    "quality_high_conf_net_count_rolling5",
    "quality_high_conf_total_count_lag1",
    "quality_high_conf_total_count_rolling5",
    "quality_sentiment_disagreement_lag1",
    "quality_sentiment_disagreement_rolling5",
    "quality_unique_story_count_lag1",
    "quality_unique_story_count_rolling5",
]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    quality_cols = [col for col in QUALITY_NEWS_FEATURES if col in df.columns]
    feature_sets = dict(FEATURE_SETS)
    if quality_cols:
        feature_sets["quality_news_only"] = CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names() + quality_cols
        feature_sets["price_news_quality"] = CORE_PRICE_FEATURES + CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names() + quality_cols
    return feature_sets

TARGETS = [
    "target_3d_excess_gt_0",
    "target_3d_excess_gt_1pct",
    "target_3d_abs_excess_gt_1pct",
    "target_5d_excess_gt_0",
    "target_5d_excess_gt_1pct",
    "target_5d_excess_gt_2pct",
    "target_5d_abs_excess_gt_1pct",
]


def add_features_and_targets(base: pd.DataFrame, event_daily: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    existing_event_cols = [col for col in category_feature_names() if col in df.columns]
    if existing_event_cols:
        df = df.drop(columns=existing_event_cols)
    event_daily = event_daily.copy()
    event_daily["trading_date"] = pd.to_datetime(event_daily["trading_date"])
    df = df.merge(event_daily, on=["ticker", "trading_date"], how="left")

    for col in category_feature_names():
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["ticker", "trading_date"]).copy()
    g = df.groupby("ticker", group_keys=False)
    for horizon in [3, 5]:
        df[f"fwd_{horizon}d_return"] = g["adj_close"].shift(-horizon) / df["adj_close"] - 1
        df[f"fwd_{horizon}d_spy_return"] = g["spy_return_1d"].transform(
            lambda s: (1 + s.shift(-1)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1
        )
        df[f"fwd_{horizon}d_excess_return"] = df[f"fwd_{horizon}d_return"] - df[f"fwd_{horizon}d_spy_return"]
        df[f"target_{horizon}d_excess_gt_0"] = (df[f"fwd_{horizon}d_excess_return"] > 0).astype(int)
        df[f"target_{horizon}d_excess_gt_1pct"] = (df[f"fwd_{horizon}d_excess_return"] > 0.01).astype(int)
        df[f"target_{horizon}d_excess_gt_2pct"] = (df[f"fwd_{horizon}d_excess_return"] > 0.02).astype(int)
        df[f"target_{horizon}d_abs_excess_gt_1pct"] = (df[f"fwd_{horizon}d_excess_return"].abs() > 0.01).astype(int)

    df["sentiment_intensity"] = df["finbert_positive_mean"] + df["finbert_negative_mean"]
    df["abs_sentiment_score"] = df["finbert_sentiment_score_mean"].abs()
    df["sentiment_weighted_news_count"] = df["finbert_sentiment_score_mean"] * df["log_news_count"]
    df["negative_weighted_news_count"] = df["finbert_negative_mean"] * df["log_news_count"]
    df["positive_weighted_news_count"] = df["finbert_positive_mean"] * df["log_news_count"]
    df["stock_minus_market_sentiment"] = df["finbert_sentiment_score_mean"] - df["market_finbert_sentiment_score_mean"]
    df["stock_sentiment_x_market_sentiment"] = df["finbert_sentiment_score_mean"] * df["market_finbert_sentiment_score_mean"]
    df["event_count_total"] = df[[f"{name}_count" for name in EVENT_PATTERNS]].sum(axis=1)
    df["market_abs_sentiment_score"] = df["market_finbert_sentiment_score_mean"].abs()

    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["fwd_3d_excess_return", "fwd_5d_excess_return"])


def add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q = out[["news_count", "market_news_count", "abs_sentiment_score", "sentiment_intensity"]].quantile([0.75, 0.90, 0.95])
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


FILTERS = [
    "all_days",
    "top_10pct_news_volume",
    "top_5pct_news_volume",
    "top_10pct_market_news",
    "strong_abs_sentiment",
    "high_sentiment_intensity",
    "earnings_high_news",
    "analyst_or_earnings",
    "macro_high_market",
]


def build_splits(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    dates = sorted(df["trading_date"].drop_duplicates())
    train_end = max(20, int(len(dates) * 0.5))
    test_size = max(5, int(len(dates) * 0.15))
    step = max(5, int(len(dates) * 0.15))
    splits = []
    fold = 1
    while train_end < len(dates) - test_size:
        train_dates = dates[:train_end]
        test_dates = dates[train_end : train_end + test_size]
        splits.append((df[df["trading_date"].isin(train_dates)].copy(), df[df["trading_date"].isin(test_dates)].copy(), fold))
        train_end += step
        fold += 1
    return splits


def build_model(model_name: str) -> Pipeline:
    if model_name == "logistic_balanced":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ]
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMClassifier(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    raise ValueError(f"Unsupported model: {model_name}")


MODELS = ["logistic_balanced", "lightgbm"]


def metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def run_scope(df: pd.DataFrame, scope: str, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for train, test, fold in build_splits(df):
        for filter_name in FILTERS:
            train_f = train[train[filter_name] == 1].copy()
            test_f = test[test[filter_name] == 1].copy()
            if len(train_f) < 80 or len(test_f) < 25:
                continue
            for target in TARGETS:
                y_train = train_f[target].astype(int)
                y_test = test_f[target].astype(int)
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                for feature_set, features in feature_sets.items():
                    feature_cols = [col for col in features if col in train_f.columns]
                    for model_name in MODELS:
                        model = build_model(model_name)
                        model.fit(train_f[feature_cols], y_train)
                        pred = model.predict(test_f[feature_cols])
                        proba = model.predict_proba(test_f[feature_cols])[:, 1]
                        rows.append(
                            {
                                "scope": scope,
                                "fold": fold,
                                "filter": filter_name,
                                "target": target,
                                "feature_set": feature_set,
                                "model_name": model_name,
                                "feature_count": len(feature_cols),
                                "train_rows": len(train_f),
                                "test_rows": len(test_f),
                                "test_positive_rate": float(y_test.mean()),
                                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                                **metric_row(y_test, pred, proba),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "majority_baseline_accuracy", "test_positive_rate", "train_rows", "test_rows"]
    summary = (
        results.groupby(["scope", "filter", "target", "feature_set", "model_name"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metric_cols})
    )
    counts = results.groupby(["scope", "filter", "target", "feature_set", "model_name"], as_index=False)["fold"].count().rename(columns={"fold": "num_folds"})
    return summary.merge(counts, on=["scope", "filter", "target", "feature_set", "model_name"], how="left")


def best_news_vs_price(summary: pd.DataFrame) -> pd.DataFrame:
    idx = ["scope", "filter", "target"]
    best = summary.sort_values(idx + ["mean_roc_auc"], ascending=[True, True, True, False])
    best = best.groupby(idx + ["feature_set"], as_index=False).head(1)
    price = best[best["feature_set"] == "price_only"][idx + ["model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy", "mean_majority_baseline_accuracy", "mean_test_rows"]].rename(
        columns={"model_name": "best_price_model", "mean_roc_auc": "best_price_roc_auc", "mean_accuracy": "best_price_accuracy", "mean_balanced_accuracy": "best_price_balanced_accuracy", "mean_majority_baseline_accuracy": "majority_baseline_accuracy", "mean_test_rows": "mean_test_rows"}
    )
    news = best[best["feature_set"].isin(["price_news", "price_news_events", "price_news_quality"])].sort_values(idx + ["mean_roc_auc"], ascending=[True, True, True, False])
    news = news.groupby(idx, as_index=False).head(1)
    news = news[idx + ["feature_set", "model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy"]].rename(
        columns={"feature_set": "best_news_feature_set", "model_name": "best_news_model", "mean_roc_auc": "best_news_roc_auc", "mean_accuracy": "best_news_accuracy", "mean_balanced_accuracy": "best_news_balanced_accuracy"}
    )
    out = price.merge(news, on=idx, how="inner")
    out["roc_auc_delta_news_minus_price"] = out["best_news_roc_auc"] - out["best_price_roc_auc"]
    out["accuracy_delta_news_minus_price"] = out["best_news_accuracy"] - out["best_price_accuracy"]
    out["balanced_accuracy_delta_news_minus_price"] = out["best_news_balanced_accuracy"] - out["best_price_balanced_accuracy"]
    return out.sort_values(["scope", "filter", "roc_auc_delta_news_minus_price"], ascending=[True, True, False])


def filter_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, scope_df in [("pooled_all_tickers", df), *[(ticker, tdf) for ticker, tdf in df.groupby("ticker")]]:
        for filter_name in FILTERS:
            selected = scope_df[scope_df[filter_name] == 1]
            rows.append(
                {
                    "scope": scope,
                    "filter": filter_name,
                    "rows": len(selected),
                    "unique_dates": selected["trading_date"].nunique(),
                    "mean_news_count": selected["news_count"].mean() if len(selected) else 0,
                    "mean_market_news_count": selected["market_news_count"].mean() if len(selected) else 0,
                    "mean_abs_sentiment_score": selected["abs_sentiment_score"].mean() if len(selected) else 0,
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_feature_sets(df)

    frames = [run_scope(df, "pooled_all_tickers", feature_sets)]
    for ticker, ticker_df in df.groupby("ticker"):
        frames.append(run_scope(ticker_df.copy(), ticker, feature_sets))
    results = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    summary = summarize(results)
    comparison = best_news_vs_price(summary)
    coverage = filter_coverage(df)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "rows": len(df),
                "targets": ", ".join(TARGETS),
                "filters": ", ".join(FILTERS),
                "models": ", ".join(MODELS),
                "feature_sets": ", ".join(feature_sets),
                "quality_feature_count": len([col for col in QUALITY_NEWS_FEATURES if col in df.columns]),
            }
        ]
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"high_signal_event_results{suffix}.csv", index=False)
    summary.to_csv(paths.tables_dir / f"high_signal_event_summary{suffix}.csv", index=False)
    comparison.to_csv(paths.tables_dir / f"high_signal_event_news_vs_price{suffix}.csv", index=False)
    coverage.to_csv(paths.tables_dir / f"high_signal_event_filter_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"high_signal_event_metadata{suffix}.csv", index=False)
    return {"results": results, "summary": summary, "comparison": comparison, "coverage": coverage, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict high-signal event-only experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nFilter coverage")
    print(outputs["coverage"].head(60).to_string(index=False))
    print("\nBest news vs price")
    print(outputs["comparison"].sort_values("roc_auc_delta_news_minus_price", ascending=False).head(40).to_string(index=False))
    print("\nBest rows")
    print(outputs["summary"].sort_values("mean_roc_auc", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
