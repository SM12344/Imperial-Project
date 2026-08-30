from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from run_event_target_experiments import EVENT_PATTERNS, build_event_daily_features, category_feature_names
from run_modelling_baselines import build_paths, load_dataset


SOURCE_WEIGHTS = {
    "Reuters": 1.35,
    "CNBC": 1.25,
    "The Wall Street Journal": 1.25,
    "Barron's": 1.20,
    "MarketWatch": 1.10,
    "Yahoo Finance": 1.05,
    "Zacks Investment Research": 1.00,
    "Zacks": 1.00,
    "Investing.com": 0.95,
    "Benzinga": 0.90,
    "The Motley Fool": 0.85,
    "GlobeNewswire Inc.": 0.75,
    "PR Newswire": 0.75,
    "Business Wire": 0.75,
}

QUALITY_BASE_FEATURES = [
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
]


def normalize_title(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = [word for word in text.split() if word]
    return " ".join(words[:12])


def load_scored_news(paths, scored_news_name: str) -> pd.DataFrame:
    path = paths.processed_dir / scored_news_name
    if not path.exists():
        raise FileNotFoundError(f"Scored news file not found: {path}")
    news = pd.read_csv(path)
    news = news.dropna(subset=["aligned_trading_date"]).copy()
    news["trading_date"] = pd.to_datetime(news["aligned_trading_date"])
    news["publisher"] = news["publisher"].fillna("Unknown").astype(str)
    news["source_weight"] = news["publisher"].map(SOURCE_WEIGHTS).fillna(0.8)
    news["title_key"] = news["title"].map(normalize_title)
    news["text_for_events"] = (
        news["title"].fillna("").astype(str) + " " + news["description"].fillna("").astype(str)
    ).str.lower()
    return news


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any() or weights[valid].sum() == 0:
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))


def build_daily_quality_features(news: pd.DataFrame) -> pd.DataFrame:
    ticker_news = news[news["ticker"] != "__MARKET__"].copy()
    dedup = (
        ticker_news.sort_values(["ticker", "trading_date", "title_key", "published_utc"])
        .groupby(["ticker", "trading_date", "title_key"], as_index=False)
        .agg(
            article_id=("article_id", "first"),
            finbert_sentiment_score=("finbert_sentiment_score", "mean"),
            finbert_positive=("finbert_positive", "mean"),
            finbert_negative=("finbert_negative", "mean"),
        )
    )

    base_rows = []
    for (ticker, trading_date), group in ticker_news.groupby(["ticker", "trading_date"]):
        weights = group["source_weight"]
        article_count = len(group)
        unique_story_count = group["title_key"].nunique()
        high_pos = (group["finbert_positive"] >= 0.75).sum()
        high_neg = (group["finbert_negative"] >= 0.75).sum()
        high_neu = (group["finbert_neutral"] >= 0.75).sum()
        pos_count = (group["finbert_predicted_label"] == "positive").sum()
        neg_count = (group["finbert_predicted_label"] == "negative").sum()
        neu_count = (group["finbert_predicted_label"] == "neutral").sum()
        dedup_group = dedup[(dedup["ticker"] == ticker) & (dedup["trading_date"] == trading_date)]
        base_rows.append(
            {
                "ticker": ticker,
                "trading_date": trading_date,
                "quality_unique_story_count": unique_story_count,
                "quality_duplicate_article_ratio": 1.0 - (unique_story_count / article_count if article_count else 0.0),
                "quality_source_weight_mean": group["source_weight"].mean(),
                "quality_source_weighted_sentiment_mean": weighted_mean(group["finbert_sentiment_score"], weights),
                "quality_source_weighted_positive_mean": weighted_mean(group["finbert_positive"], weights),
                "quality_source_weighted_negative_mean": weighted_mean(group["finbert_negative"], weights),
                "quality_high_conf_positive_count": high_pos,
                "quality_high_conf_negative_count": high_neg,
                "quality_high_conf_neutral_count": high_neu,
                "quality_high_conf_net_count": high_pos - high_neg,
                "quality_high_conf_total_count": high_pos + high_neg + high_neu,
                "quality_high_conf_positive_share": high_pos / article_count if article_count else 0.0,
                "quality_high_conf_negative_share": high_neg / article_count if article_count else 0.0,
                "quality_positive_article_count": pos_count,
                "quality_negative_article_count": neg_count,
                "quality_neutral_article_count": neu_count,
                "quality_positive_negative_ratio": (pos_count + 1) / (neg_count + 1),
                "quality_max_positive_confidence": group["finbert_positive"].max(),
                "quality_max_negative_confidence": group["finbert_negative"].max(),
                "quality_sentiment_disagreement": group["finbert_sentiment_score"].std(ddof=0),
                "quality_dedup_sentiment_mean": dedup_group["finbert_sentiment_score"].mean(),
                "quality_dedup_positive_mean": dedup_group["finbert_positive"].mean(),
                "quality_dedup_negative_mean": dedup_group["finbert_negative"].mean(),
            }
        )

    quality = pd.DataFrame(base_rows)
    event_daily = build_event_daily_features(news)
    if not event_daily.empty:
        event_daily["trading_date"] = pd.to_datetime(event_daily["trading_date"])
        quality = quality.merge(event_daily, on=["ticker", "trading_date"], how="left")
    for col in category_feature_names():
        if col not in quality.columns:
            quality[col] = 0.0
        quality[col] = quality[col].fillna(0.0)
    return quality


def add_lagged_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    g = out.groupby("ticker", group_keys=False)
    lag_cols = [
        "quality_source_weighted_sentiment_mean",
        "quality_high_conf_net_count",
        "quality_high_conf_total_count",
        "quality_sentiment_disagreement",
        "quality_unique_story_count",
    ]
    for col in lag_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[f"{col}_lag1"] = g[col].shift(1)
        out[f"{col}_rolling5"] = g[col].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    return out


def run_pipeline(
    project_root: str | None,
    dataset_name: str,
    scored_news_name: str,
    output_suffix: str,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    model_df = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    quality = build_daily_quality_features(scored_news)
    enhanced = model_df.merge(quality, on=["ticker", "trading_date"], how="left")
    quality_cols = [col for col in quality.columns if col not in {"ticker", "trading_date"}]
    enhanced[quality_cols] = enhanced[quality_cols].fillna(0.0)
    enhanced = add_lagged_quality_features(enhanced).fillna(0.0)

    coverage = (
        quality.groupby("ticker", as_index=False)
        .agg(
            quality_rows=("trading_date", "size"),
            mean_unique_stories=("quality_unique_story_count", "mean"),
            mean_duplicate_ratio=("quality_duplicate_article_ratio", "mean"),
            mean_high_conf_positive=("quality_high_conf_positive_count", "mean"),
            mean_high_conf_negative=("quality_high_conf_negative_count", "mean"),
            mean_sentiment_disagreement=("quality_sentiment_disagreement", "mean"),
        )
    )
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "rows": len(enhanced),
                "quality_feature_count": len([col for col in enhanced.columns if col.startswith("quality_")]),
                "event_feature_count": len([col for col in enhanced.columns if any(col.startswith(name) for name in EVENT_PATTERNS)]),
            }
        ]
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    enhanced.to_csv(paths.processed_dir / f"model_dataset_finbert_quality_complete{suffix}.csv", index=False)
    quality.to_csv(paths.processed_dir / f"daily_news_quality_features{suffix}.csv", index=False)
    coverage.to_csv(paths.tables_dir / f"news_quality_feature_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"news_quality_feature_metadata{suffix}.csv", index=False)
    return {"enhanced": enhanced, "quality": quality, "coverage": coverage, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build duplicate-aware and source-weighted news quality features.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print()
    print(outputs["coverage"].to_string(index=False))


if __name__ == "__main__":
    main()
