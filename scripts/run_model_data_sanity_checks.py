from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from run_2024_holdout_ablation import TEST_END, TEST_START, TRAIN_END, VALIDATION_START, build_ablation_feature_sets
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset


TARGETS = [
    "target_next_day_up",
    "target_3d_excess_gt_0",
    "target_3d_excess_gt_1pct",
    "target_3d_abs_excess_gt_1pct",
    "target_5d_excess_gt_0",
    "target_5d_excess_gt_1pct",
    "target_5d_excess_gt_2pct",
    "target_5d_abs_excess_gt_1pct",
]

KEY_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volume_change_1d",
    "ma_5_20_gap",
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_5d",
    "news_count",
    "market_news_count",
    "finbert_sentiment_score_mean",
    "market_finbert_sentiment_score_mean",
    "quality_unique_story_count",
    "quality_high_conf_net_count",
    "quality_high_conf_net_count_lag1",
    "quality_high_conf_net_count_rolling5",
    "quality_sentiment_disagreement",
]

KEYWORD_PATTERNS = {
    "AAPL": "aapl|apple|iphone|ipad|macbook|ios|app store",
    "AMZN": "amzn|amazon|aws|prime video|bezos",
    "MSFT": "msft|microsoft|windows|azure|xbox|copilot|openai",
    "NVDA": "nvda|nvidia|gpu|blackwell|cuda|jensen huang",
    "TSLA": "tsla|tesla|elon|musk|cybertruck|model y|model 3",
}


def split_name(date: pd.Timestamp) -> str:
    if date < pd.Timestamp(TRAIN_END):
        return "train_2020_2022"
    if pd.Timestamp(VALIDATION_START) <= date < pd.Timestamp(TEST_START):
        return "validation_2023"
    if pd.Timestamp(TEST_START) <= date < pd.Timestamp(TEST_END):
        return "test_2024"
    return "outside"


def build_overview(df: pd.DataFrame, dataset_name: str, scored_news_name: str) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    return pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "rows": len(df),
                "columns": df.shape[1],
                "tickers": ", ".join(sorted(df["ticker"].unique())),
                "unique_tickers": df["ticker"].nunique(),
                "unique_trading_dates": df["trading_date"].nunique(),
                "first_trading_date": df["trading_date"].min().date().isoformat(),
                "last_trading_date": df["trading_date"].max().date().isoformat(),
                "duplicate_ticker_date_rows": int(df.duplicated(["ticker", "trading_date"]).sum()),
                "numeric_null_cells": int(numeric.isna().sum().sum()),
                "numeric_infinite_cells": int(np.isinf(numeric.to_numpy()).sum()),
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )


def build_ticker_date_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_dates = set(df["trading_date"].drop_duplicates())
    for ticker, part in df.groupby("ticker"):
        dates = set(part["trading_date"])
        rows.append(
            {
                "ticker": ticker,
                "rows": len(part),
                "unique_dates": len(dates),
                "first_date": part["trading_date"].min().date().isoformat(),
                "last_date": part["trading_date"].max().date().isoformat(),
                "duplicate_dates": int(part.duplicated("trading_date").sum()),
                "missing_dates_vs_union": len(all_dates - dates),
            }
        )
    return pd.DataFrame(rows).sort_values("ticker")


def build_split_target_balance(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["split"] = work["trading_date"].map(split_name)
    rows = []
    for (ticker, split), part in work.groupby(["ticker", "split"], sort=True):
        if split == "outside":
            continue
        row: dict[str, Any] = {"ticker": ticker, "split": split, "rows": len(part), "unique_dates": part["trading_date"].nunique()}
        for target in TARGETS:
            if target in part.columns:
                row[f"{target}_positive_rate"] = float(part[target].mean())
                row[f"{target}_majority_baseline"] = float(max(part[target].mean(), 1 - part[target].mean()))
        rows.append(row)
    pooled = []
    for split, part in work.groupby("split", sort=True):
        if split == "outside":
            continue
        row = {"ticker": "POOLED", "split": split, "rows": len(part), "unique_dates": part["trading_date"].nunique()}
        for target in TARGETS:
            if target in part.columns:
                row[f"{target}_positive_rate"] = float(part[target].mean())
                row[f"{target}_majority_baseline"] = float(max(part[target].mean(), 1 - part[target].mean()))
        pooled.append(row)
    return pd.DataFrame(pooled + rows)


def build_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing = df[col].isna().mean()
        if missing > 0:
            rows.append({"column": col, "missing_rows": int(df[col].isna().sum()), "missing_rate": float(missing)})
    if not rows:
        return pd.DataFrame([{"column": "NONE", "missing_rows": 0, "missing_rate": 0.0}])
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def build_news_coverage(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["split"] = work["trading_date"].map(split_name)
    rows = []
    for (ticker, split), part in work.groupby(["ticker", "split"], sort=True):
        if split == "outside":
            continue
        rows.append(
            {
                "ticker": ticker,
                "split": split,
                "rows": len(part),
                "days_with_stock_news_rate": float((part["news_count"] > 0).mean()),
                "days_with_market_news_rate": float((part["market_news_count"] > 0).mean()),
                "mean_stock_news_count": float(part["news_count"].mean()),
                "median_stock_news_count": float(part["news_count"].median()),
                "mean_market_news_count": float(part["market_news_count"].mean()),
                "median_market_news_count": float(part["market_news_count"].median()),
                "mean_quality_unique_story_count": float(part.get("quality_unique_story_count", pd.Series(0, index=part.index)).mean()),
                "mean_finbert_sentiment": float(part["finbert_sentiment_score_mean"].mean()),
                "mean_abs_finbert_sentiment": float(part["finbert_sentiment_score_mean"].abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_distribution_drift(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["split"] = work["trading_date"].map(split_name)
    rows = []
    for ticker, ticker_part in work.groupby("ticker"):
        train = ticker_part[ticker_part["split"] == "train_2020_2022"]
        test = ticker_part[ticker_part["split"] == "test_2024"]
        if train.empty or test.empty:
            continue
        for col in [c for c in KEY_FEATURES if c in df.columns]:
            train_std = train[col].std()
            pooled_std = ticker_part[col].std()
            rows.append(
                {
                    "ticker": ticker,
                    "feature": col,
                    "train_mean": float(train[col].mean()),
                    "test_mean": float(test[col].mean()),
                    "mean_shift_train_to_test": float(test[col].mean() - train[col].mean()),
                    "shift_in_pooled_std_units": float((test[col].mean() - train[col].mean()) / pooled_std) if pooled_std and not np.isnan(pooled_std) else np.nan,
                    "train_std": float(train_std) if not np.isnan(train_std) else np.nan,
                    "test_std": float(test[col].std()) if not np.isnan(test[col].std()) else np.nan,
                    "train_zero_rate": float((train[col] == 0).mean()),
                    "test_zero_rate": float((test[col] == 0).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("shift_in_pooled_std_units", key=lambda s: s.abs(), ascending=False)


def build_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in KEY_FEATURES if c in df.columns]:
        s = df[col]
        rows.append(
            {
                "feature": col,
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.quantile(0.50)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "zero_rate": float((s == 0).mean()),
                "missing_rate": float(s.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_target_recomputation_checks(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["ticker", "trading_date"]).copy()
    g = work.groupby("ticker", group_keys=False)
    work["recomputed_target_next_day_up"] = (g["adj_close"].shift(-1) > work["adj_close"]).astype(float)
    valid_next = g["adj_close"].shift(-1).notna()
    next_mismatch = (work.loc[valid_next, "recomputed_target_next_day_up"].astype(int) != work.loc[valid_next, "target_next_day_up"].astype(int)).sum()

    rows = [
        {
            "check": "target_next_day_up_matches_adj_close_lead",
            "valid_rows": int(valid_next.sum()),
            "mismatches": int(next_mismatch),
            "mismatch_rate": float(next_mismatch / valid_next.sum()) if valid_next.sum() else np.nan,
        }
    ]
    for horizon in [3, 5]:
        fwd_return = g["adj_close"].shift(-horizon) / work["adj_close"] - 1
        fwd_spy = g["spy_return_1d"].transform(
            lambda s: (1 + s.shift(-1)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1
        )
        excess = fwd_return - fwd_spy
        valid = excess.notna()
        for suffix, recomputed in [
            ("excess_gt_0", excess > 0),
            ("excess_gt_1pct", excess > 0.01),
            ("excess_gt_2pct", excess > 0.02),
            ("abs_excess_gt_1pct", excess.abs() > 0.01),
        ]:
            target = f"target_{horizon}d_{suffix}"
            if target not in work.columns:
                continue
            mismatches = (recomputed.loc[valid].astype(int) != work.loc[valid, target].astype(int)).sum()
            rows.append(
                {
                    "check": f"{target}_matches_recomputed_forward_excess",
                    "valid_rows": int(valid.sum()),
                    "mismatches": int(mismatches),
                    "mismatch_rate": float(mismatches / valid.sum()) if valid.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_leakage_screen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = build_ablation_feature_sets(df)
    model_features = sorted(set().union(*[set(cols) for cols in feature_sets.values()]))
    suspicious_name_rows = []
    for col in model_features:
        lowered = col.lower()
        if any(token in lowered for token in ["target", "fwd", "future", "next"]):
            suspicious_name_rows.append({"feature": col, "reason": "feature name suggests future/target leakage"})
    if not suspicious_name_rows:
        suspicious_name_rows.append({"feature": "NONE", "reason": "No model feature names contain target/fwd/future/next."})

    corr_rows = []
    numeric_features = [col for col in model_features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    for target in [t for t in TARGETS if t in df.columns]:
        y = df[target]
        for col in numeric_features:
            if df[col].nunique(dropna=True) <= 1:
                continue
            corr = df[col].corr(y)
            if pd.notna(corr) and abs(corr) >= 0.15:
                corr_rows.append({"target": target, "feature": col, "pearson_corr": float(corr), "abs_corr": float(abs(corr))})
    if not corr_rows:
        corr_rows.append({"target": "NONE", "feature": "NONE", "pearson_corr": 0.0, "abs_corr": 0.0})
    return pd.DataFrame(suspicious_name_rows), pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)


def add_keyword_relevance(news: pd.DataFrame) -> pd.DataFrame:
    work = news[news["ticker"].isin(KEYWORD_PATTERNS)].copy()
    text = work["title"].fillna("").astype(str) + " " + work["description"].fillna("").astype(str)
    work["keyword_relevant"] = [
        bool(re.search(KEYWORD_PATTERNS[ticker], value.lower())) for ticker, value in zip(work["ticker"], text)
    ]
    work["split"] = work["aligned_trading_date"].map(split_name)
    return work


def build_keyword_relevance_summary(news: pd.DataFrame) -> pd.DataFrame:
    work = add_keyword_relevance(news)
    return (
        work[work["split"] != "outside"]
        .groupby(["ticker", "split"], as_index=False)
        .agg(
            rows=("article_id", "size"),
            unique_articles=("article_id", "nunique"),
            keyword_relevant_rate=("keyword_relevant", "mean"),
            first_date=("aligned_trading_date", "min"),
            last_date=("aligned_trading_date", "max"),
        )
        .sort_values(["ticker", "split"])
    )


def build_low_relevance_daily_news(news: pd.DataFrame) -> pd.DataFrame:
    work = add_keyword_relevance(news)
    daily = (
        work.groupby(["ticker", "aligned_trading_date"], as_index=False)
        .agg(rows=("article_id", "size"), keyword_relevant_rate=("keyword_relevant", "mean"))
    )
    return daily[daily["rows"] >= 30].sort_values(["keyword_relevant_rate", "rows"], ascending=[True, False]).head(50)


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_features_and_targets(base, build_event_daily_features(scored_news))
    df = df.replace([np.inf, -np.inf], np.nan)
    outputs = {
        "overview": build_overview(df, dataset_name, scored_news_name),
        "ticker_date_coverage": build_ticker_date_coverage(df),
        "split_target_balance": build_split_target_balance(df),
        "missingness": build_missingness(df),
        "news_coverage": build_news_coverage(df),
        "distribution_drift": build_distribution_drift(df),
        "extreme_values": build_extreme_values(df),
        "target_recomputation": build_target_recomputation_checks(df),
        "keyword_relevance_summary": build_keyword_relevance_summary(scored_news),
        "low_relevance_daily_news": build_low_relevance_daily_news(scored_news),
    }
    leakage_names, leakage_corr = build_leakage_screen(df)
    outputs["leakage_name_screen"] = leakage_names
    outputs["leakage_correlation_screen"] = leakage_corr

    suffix = f"_{output_suffix}" if output_suffix else ""
    for name, table in outputs.items():
        table.to_csv(paths.tables_dir / f"model_data_sanity_{name}{suffix}.csv", index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all-ticker modelling dataset sanity checks.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    for name, table in outputs.items():
        print(f"\n{name}")
        print(table.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
