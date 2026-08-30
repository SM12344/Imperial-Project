from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


COMMON_STOCK_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def load_article_ticker_rows(news_path: Path) -> pd.DataFrame:
    with news_path.open(encoding="utf-8") as f:
        news_raw = json.load(f)

    rows: list[dict[str, str]] = []
    for article in news_raw:
        published_utc = article.get("published_utc")
        if not published_utc:
            continue
        date = pd.to_datetime(published_utc, utc=True).date().isoformat()
        for insight in article.get("insights") or []:
            ticker = insight.get("ticker")
            if ticker:
                rows.append(
                    {
                        "article_id": str(article.get("id", "")),
                        "published_utc": published_utc,
                        "date": date,
                        "ticker": str(ticker),
                    }
                )
    return pd.DataFrame(rows)


def analyze_universe(news_path: Path, min_rows: int, min_days: int) -> pd.DataFrame:
    rows = load_article_ticker_rows(news_path)
    summary = (
        rows.groupby("ticker", as_index=False)
        .agg(
            news_rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["news_rows", "news_days"], ascending=False)
    )
    summary["common_stock_like"] = summary["ticker"].map(lambda x: bool(COMMON_STOCK_PATTERN.match(x)))
    filtered = summary[
        (summary["news_rows"] >= min_rows)
        & (summary["news_days"] >= min_days)
        & summary["common_stock_like"]
    ].copy()
    return filtered.sort_values(["news_rows", "news_days"], ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidate tickers from the local Polygon news file.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--news-path", default=None)
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--min-days", type=int, default=20)
    parser.add_argument("--output-name", default="candidate_ticker_universe.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root(args.project_root)
    news_path = Path(args.news_path).resolve() if args.news_path else root / "data" / "raw" / "polygon_news_sample.json"
    output_path = root / "outputs" / "tables" / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = analyze_universe(news_path, args.min_rows, args.min_days)
    candidates.to_csv(output_path, index=False)
    print(candidates.head(50).to_string(index=False))
    print()
    print(f"Wrote {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
