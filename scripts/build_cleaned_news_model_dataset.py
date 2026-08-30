from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from build_news_quality_features import SOURCE_WEIGHTS, add_lagged_quality_features, build_daily_quality_features, normalize_title
from run_finbert_sentiment_pipeline import (
    BENCHMARK_TICKER,
    MARKET_CONTEXT_TICKER,
    TARGET_TICKERS,
    build_daily_finbert_features,
    build_model_dataset,
)
from run_modelling_baselines import build_paths


KEYWORD_PATTERNS = {
    "AAPL": "aapl|apple|iphone|ipad|macbook|ios|app store",
    "AMZN": "amzn|amazon|aws|prime video|bezos",
    "MSFT": "msft|microsoft|windows|azure|xbox|copilot|openai",
    "NVDA": "nvda|nvidia|gpu|blackwell|cuda|jensen huang",
    "TSLA": "tsla|tesla|elon|musk|cybertruck|model y|model 3",
}

CAP_COLUMNS = [
    "news_count",
    "log_news_count",
    "news_count_lag1",
    "news_count_rolling3",
    "news_count_rolling5",
    "quality_unique_story_count",
    "quality_high_conf_positive_count",
    "quality_high_conf_negative_count",
    "quality_high_conf_neutral_count",
    "quality_high_conf_net_count",
    "quality_high_conf_total_count",
    "quality_positive_article_count",
    "quality_negative_article_count",
    "quality_neutral_article_count",
    "quality_positive_negative_ratio",
    "quality_high_conf_net_count_lag1",
    "quality_high_conf_net_count_rolling5",
    "quality_high_conf_total_count_lag1",
    "quality_high_conf_total_count_rolling5",
    "quality_unique_story_count_lag1",
    "quality_unique_story_count_rolling5",
]

TWO_SIDED_CAP_COLUMNS = {
    "quality_high_conf_net_count",
    "quality_high_conf_net_count_lag1",
    "quality_high_conf_net_count_rolling5",
}


def keyword_relevant(row: pd.Series) -> bool:
    ticker = row["ticker"]
    if ticker == MARKET_CONTEXT_TICKER:
        return True
    pattern = KEYWORD_PATTERNS.get(ticker)
    if not pattern:
        return False
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    return bool(re.search(pattern, text))


def load_inputs(paths, scored_news_name: str, price_features_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_path = paths.processed_dir / scored_news_name
    price_path = paths.processed_dir / price_features_name
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored news file not found: {scored_path}")
    if not price_path.exists():
        raise FileNotFoundError(f"Price features file not found: {price_path}")
    scored = pd.read_csv(scored_path)
    price_features = pd.read_csv(price_path)
    scored["article_id"] = scored["article_id"].astype(str)
    scored["aligned_trading_date"] = pd.to_datetime(scored["aligned_trading_date"])
    price_features["date"] = price_features["date"].astype(str)
    return scored, price_features


def clean_scored_news(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = scored.dropna(subset=["aligned_trading_date"]).copy()
    work["keyword_relevant"] = work.apply(keyword_relevant, axis=1)
    keep_mask = (work["ticker"] == MARKET_CONTEXT_TICKER) | work["keyword_relevant"]
    cleaned = work[keep_mask].copy()
    dropped = work[~keep_mask].copy()

    summary_rows = []
    for ticker, part in work.groupby("ticker"):
        kept = cleaned[cleaned["ticker"] == ticker]
        summary_rows.append(
            {
                "ticker": ticker,
                "input_rows": len(part),
                "kept_rows": len(kept),
                "dropped_rows": len(part) - len(kept),
                "kept_rate": len(kept) / len(part) if len(part) else np.nan,
                "input_unique_articles": part["article_id"].nunique(),
                "kept_unique_articles": kept["article_id"].nunique(),
                "dropped_unique_articles": dropped[dropped["ticker"] == ticker]["article_id"].nunique(),
            }
        )
    return cleaned, dropped, pd.DataFrame(summary_rows).sort_values("ticker")


def cap_extreme_features(model_df: pd.DataFrame, reference_end: str = "2024-01-01", upper_quantile: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = model_df.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    reference = out[out["trading_date"] < pd.Timestamp(reference_end)]
    cap_rows = []
    for col in [c for c in CAP_COLUMNS if c in out.columns]:
        cap = reference[col].quantile(upper_quantile)
        lower_cap = reference[col].quantile(1 - upper_quantile) if col in TWO_SIDED_CAP_COLUMNS else np.nan
        before_max = out[col].max()
        before_min = out[col].min()
        upper_capped_rows = int((out[col] > cap).sum())
        lower_capped_rows = int((out[col] < lower_cap).sum()) if col in TWO_SIDED_CAP_COLUMNS else 0
        if col in TWO_SIDED_CAP_COLUMNS:
            out[col] = out[col].clip(lower=lower_cap, upper=cap)
        else:
            out[col] = out[col].clip(upper=cap)
        if col == "log_news_count" and "news_count" in out.columns:
            out[col] = np.log1p(out["news_count"])
        cap_rows.append(
            {
                "feature": col,
                "reference_period": f"before {reference_end}",
                "upper_quantile": upper_quantile,
                "lower_cap_value": lower_cap,
                "cap_value": cap,
                "upper_rows_capped": upper_capped_rows,
                "lower_rows_capped": lower_capped_rows,
                "rows_capped": upper_capped_rows + lower_capped_rows,
                "min_before": before_min,
                "min_after": out[col].min(),
                "max_before": before_max,
                "max_after": out[col].max(),
            }
        )
    return out, pd.DataFrame(cap_rows)


def build_quality_model_dataset(price_features: pd.DataFrame, cleaned_scored: pd.DataFrame, output_suffix: str) -> dict[str, pd.DataFrame]:
    cleaned_scored = cleaned_scored.copy()
    cleaned_scored["trading_date"] = pd.to_datetime(cleaned_scored["aligned_trading_date"])
    cleaned_scored["publisher"] = cleaned_scored["publisher"].fillna("Unknown").astype(str)
    cleaned_scored["source_weight"] = cleaned_scored["publisher"].map(SOURCE_WEIGHTS).fillna(0.8)
    cleaned_scored["title_key"] = cleaned_scored["title"].map(normalize_title)
    cleaned_scored["text_for_events"] = (
        cleaned_scored["title"].fillna("").astype(str) + " " + cleaned_scored["description"].fillna("").astype(str)
    ).str.lower()
    daily_finbert = build_daily_finbert_features(cleaned_scored)
    daily_finbert["date"] = pd.to_datetime(daily_finbert["date"]).dt.strftime("%Y-%m-%d")
    model_df, complete_model_df = build_model_dataset(
        price_features=price_features,
        daily_finbert=daily_finbert,
        tickers=TARGET_TICKERS,
        benchmark_ticker=BENCHMARK_TICKER,
        market_context_ticker=MARKET_CONTEXT_TICKER,
    )
    quality = build_daily_quality_features(cleaned_scored)
    complete_model_df["trading_date"] = pd.to_datetime(complete_model_df["trading_date"])
    quality["trading_date"] = pd.to_datetime(quality["trading_date"])
    enhanced = complete_model_df.merge(quality, on=["ticker", "trading_date"], how="left")
    quality_cols = [col for col in quality.columns if col not in {"ticker", "trading_date"}]
    enhanced[quality_cols] = enhanced[quality_cols].fillna(0.0)
    enhanced = add_lagged_quality_features(enhanced).fillna(0.0)
    capped, cap_summary = cap_extreme_features(enhanced)
    coverage = (
        cleaned_scored[cleaned_scored["ticker"] != MARKET_CONTEXT_TICKER]
        .groupby("ticker", as_index=False)
        .agg(
            cleaned_article_rows=("article_id", "size"),
            cleaned_unique_articles=("article_id", "nunique"),
            cleaned_news_days=("aligned_trading_date", "nunique"),
            first_aligned_date=("aligned_trading_date", "min"),
            last_aligned_date=("aligned_trading_date", "max"),
        )
    )
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "output_suffix": output_suffix,
                "cleaning_rule": "Keep market context rows and ticker rows whose title/description contains ticker/company keywords.",
                "cap_rule": "Cap selected news count/quality count features at the pre-2024 99th percentile.",
                "rows": len(capped),
                "columns": capped.shape[1],
                "first_trading_date": capped["trading_date"].min(),
                "last_trading_date": capped["trading_date"].max(),
            }
        ]
    )
    return {
        "daily_finbert": daily_finbert,
        "model_df": model_df,
        "complete_model_df": complete_model_df,
        "quality": quality,
        "enhanced": enhanced,
        "capped": capped,
        "cap_summary": cap_summary,
        "coverage": coverage,
        "metadata": metadata,
    }


def run_pipeline(
    project_root: str | None,
    scored_news_name: str,
    price_features_name: str,
    output_suffix: str,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    scored, price_features = load_inputs(paths, scored_news_name, price_features_name)
    cleaned, dropped, filter_summary = clean_scored_news(scored)
    outputs = build_quality_model_dataset(price_features, cleaned, output_suffix)

    suffix = f"_{output_suffix}" if output_suffix else ""
    cleaned.to_csv(paths.processed_dir / f"news_target_tickers_finbert_scored{suffix}.csv", index=False)
    dropped.to_csv(paths.processed_dir / f"news_target_tickers_finbert_scored_dropped{suffix}.csv", index=False)
    outputs["daily_finbert"].to_csv(paths.processed_dir / f"daily_finbert_sentiment_features{suffix}.csv", index=False)
    outputs["complete_model_df"].to_csv(paths.processed_dir / f"model_dataset_finbert_complete{suffix}.csv", index=False)
    outputs["quality"].to_csv(paths.processed_dir / f"daily_news_quality_features{suffix}.csv", index=False)
    outputs["capped"].to_csv(paths.processed_dir / f"model_dataset_finbert_quality_complete{suffix}.csv", index=False)
    filter_summary.to_csv(paths.tables_dir / f"cleaned_news_filter_summary{suffix}.csv", index=False)
    outputs["cap_summary"].to_csv(paths.tables_dir / f"cleaned_news_cap_summary{suffix}.csv", index=False)
    outputs["coverage"].to_csv(paths.tables_dir / f"cleaned_news_coverage{suffix}.csv", index=False)
    outputs["metadata"].to_csv(paths.tables_dir / f"cleaned_news_model_dataset_metadata{suffix}.csv", index=False)

    outputs["cleaned"] = cleaned
    outputs["dropped"] = dropped
    outputs["filter_summary"] = filter_summary
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned ticker-news modelling dataset without rerunning FinBERT.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--price-features-name", required=True)
    parser.add_argument("--output-suffix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.scored_news_name, args.price_features_name, args.output_suffix)
    print("Filter summary")
    print(outputs["filter_summary"].to_string(index=False))
    print("\nCap summary")
    print(outputs["cap_summary"].to_string(index=False))
    print("\nMetadata")
    print(outputs["metadata"].to_string(index=False))


if __name__ == "__main__":
    main()
