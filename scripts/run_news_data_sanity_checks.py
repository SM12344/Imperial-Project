from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ProjectPaths:
    root: Path
    processed_dir: Path
    tables_dir: Path


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def build_paths(project_root: str | None = None) -> ProjectPaths:
    root = resolve_project_root(project_root)
    processed_dir = root / "data" / "processed"
    tables_dir = root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, processed_dir=processed_dir, tables_dir=tables_dir)


def output_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}.csv" if suffix else f"{base}.csv"


def load_processed_news(paths: ProjectPaths, news_input_name: str) -> pd.DataFrame:
    path = paths.processed_dir / news_input_name
    if not path.exists():
        raise FileNotFoundError(f"News input not found: {path}")
    news = pd.read_csv(path)
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True)
    news["date"] = pd.to_datetime(news["date"])
    news["text"] = news["text"].fillna("").astype(str)
    news["publisher"] = news["publisher"].fillna("Unknown").astype(str)
    news["news_scope"] = news.get("news_scope", "ticker_specific")
    return news


def build_overview(news: pd.DataFrame, suffix: str, finbert_scores_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_suffix": suffix,
                "article_ticker_rows": len(news),
                "unique_articles": news["article_id"].nunique(),
                "tickers_or_scopes": ", ".join(sorted(news["ticker"].dropna().unique())),
                "first_timestamp_utc": news["published_utc"].min().isoformat(),
                "last_timestamp_utc": news["published_utc"].max().isoformat(),
                "first_calendar_date": news["date"].min().date().isoformat(),
                "last_calendar_date": news["date"].max().date().isoformat(),
                "calendar_news_days": news["date"].nunique(),
                "duplicate_article_ticker_rows": int(news.duplicated(["article_id", "ticker"]).sum()),
                "missing_text_rows": int((news["text"].str.len() == 0).sum()),
                "short_text_lt_40_rows": int((news["text"].str.len() < 40).sum()),
                "missing_publisher_rows": int((news["publisher"] == "Unknown").sum()),
                "finbert_score_file_present": int(finbert_scores_path.exists()),
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )


def build_ticker_coverage(news: pd.DataFrame) -> pd.DataFrame:
    return (
        news.groupby("ticker", as_index=False)
        .agg(
            article_ticker_rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_timestamp_utc=("published_utc", "min"),
            last_timestamp_utc=("published_utc", "max"),
            mean_text_chars=("text", lambda s: s.str.len().mean()),
            median_text_chars=("text", lambda s: s.str.len().median()),
        )
        .sort_values(["article_ticker_rows", "ticker"], ascending=[False, True])
    )


def build_publisher_coverage(news: pd.DataFrame) -> pd.DataFrame:
    return (
        news.groupby(["news_scope", "publisher"], as_index=False)
        .agg(
            article_ticker_rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["article_ticker_rows", "unique_articles"], ascending=[False, False])
    )


def build_market_topic_coverage(news: pd.DataFrame) -> pd.DataFrame:
    if "query_topic" not in news.columns:
        return pd.DataFrame()
    market = news[news["ticker"] == "__MARKET__"].copy()
    if market.empty:
        return pd.DataFrame()
    market["query_topic"] = market["query_topic"].fillna("unknown")
    return (
        market.groupby("query_topic", as_index=False)
        .agg(
            rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["rows", "query_topic"], ascending=[False, True])
    )


def build_timestamp_checks(news: pd.DataFrame) -> pd.DataFrame:
    ny = news["published_utc"].dt.tz_convert("America/New_York")
    after_close = (ny.dt.hour > 16) | ((ny.dt.hour == 16) & (ny.dt.minute >= 0))
    weekend = ny.dt.weekday >= 5
    by_ticker = (
        news.assign(after_market_close=after_close.astype(int), weekend_publication=weekend.astype(int))
        .groupby("ticker", as_index=False)
        .agg(
            rows=("ticker", "size"),
            after_market_close_rows=("after_market_close", "sum"),
            weekend_publication_rows=("weekend_publication", "sum"),
            pct_after_market_close=("after_market_close", "mean"),
            pct_weekend_publication=("weekend_publication", "mean"),
        )
    )
    return by_ticker.sort_values("rows", ascending=False)


def load_finbert_scores(finbert_scores_path: Path) -> pd.DataFrame:
    if not finbert_scores_path.exists():
        return pd.DataFrame()
    scores = pd.read_csv(finbert_scores_path)
    required = ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_predicted_label"]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"FinBERT scores missing columns: {missing}")
    return scores


def build_finbert_sanity(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "message": "FinBERT article score file is not present yet; run the Colab scoring notebook first.",
                }
            ]
        )
    prob_sum = scores[["finbert_positive", "finbert_negative", "finbert_neutral"]].sum(axis=1)
    label_from_argmax = (
        scores[["finbert_positive", "finbert_negative", "finbert_neutral"]]
        .idxmax(axis=1)
        .str.replace("finbert_", "", regex=False)
    )
    return pd.DataFrame(
        [
            {
                "status": "available",
                "articles_scored": len(scores),
                "unique_articles_scored": scores["article_id"].nunique(),
                "prob_sum_min": prob_sum.min(),
                "prob_sum_max": prob_sum.max(),
                "prob_sum_mean": prob_sum.mean(),
                "label_argmax_mismatches": int((label_from_argmax != scores["finbert_predicted_label"]).sum()),
                "positive_label_share": float((scores["finbert_predicted_label"] == "positive").mean()),
                "negative_label_share": float((scores["finbert_predicted_label"] == "negative").mean()),
                "neutral_label_share": float((scores["finbert_predicted_label"] == "neutral").mean()),
                "mean_finbert_positive": scores["finbert_positive"].mean(),
                "mean_finbert_negative": scores["finbert_negative"].mean(),
                "mean_finbert_neutral": scores["finbert_neutral"].mean(),
            }
        ]
    )


def run_pipeline(
    project_root: str | None,
    news_input_name: str,
    finbert_scores_name: str,
    output_suffix: str,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    news = load_processed_news(paths, news_input_name)
    finbert_scores_path = paths.processed_dir / finbert_scores_name
    scores = load_finbert_scores(finbert_scores_path)

    outputs = {
        "overview": build_overview(news, output_suffix, finbert_scores_path),
        "ticker_coverage": build_ticker_coverage(news),
        "publisher_coverage": build_publisher_coverage(news),
        "market_topic_coverage": build_market_topic_coverage(news),
        "timestamp_checks": build_timestamp_checks(news),
        "finbert_sanity": build_finbert_sanity(scores),
    }

    for name, table in outputs.items():
        table.to_csv(paths.tables_dir / output_name(f"news_sanity_{name}", output_suffix), index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity checks for prepared stock and market-context news data.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--news-input-name", required=True)
    parser.add_argument("--finbert-scores-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        news_input_name=args.news_input_name,
        finbert_scores_name=args.finbert_scores_name,
        output_suffix=args.output_suffix,
    )
    for name, table in outputs.items():
        print(f"\n{name}")
        print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
