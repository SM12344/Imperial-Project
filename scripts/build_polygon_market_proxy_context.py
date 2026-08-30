from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MARKET_CONTEXT_TICKER = "__MARKET__"
DEFAULT_PROXY_TICKERS = {"SPY", "QQQ", "DIA", "IWM"}


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def article_key(article: dict[str, Any]) -> str:
    url = str(article.get("article_url") or "").strip()
    title = str(article.get("title") or "").strip().casefold()
    published = str(article.get("published_utc") or "")[:19]
    return url or f"{published}|{title}"


def market_id(source_key: str) -> str:
    return hashlib.sha1(f"polygon_market_proxy|{source_key}".encode("utf-8")).hexdigest()


def to_market_article(article: dict[str, Any], proxy_tickers: set[str]) -> dict[str, Any] | None:
    requested = str(article.get("requested_ticker") or "").upper()
    tickers = {str(ticker).upper() for ticker in article.get("tickers") or []}
    matched = sorted((tickers | {requested}) & proxy_tickers)
    if not matched:
        return None

    key = article_key(article)
    if not key:
        return None

    out = dict(article)
    out["id"] = market_id(key)
    out["tickers"] = [MARKET_CONTEXT_TICKER]
    out["insights"] = [
        {
            "ticker": MARKET_CONTEXT_TICKER,
            "sentiment": None,
            "sentiment_reasoning": (
                "Polygon market-proxy article converted from "
                f"{', '.join(matched)}; sentiment generated downstream by FinBERT."
            ),
        }
    ]
    out["news_scope"] = "market_context"
    out["query_topic"] = "polygon_market_proxy_" + "_".join(matched).lower()
    out["source_proxy_tickers"] = matched
    out["source_article_id"] = article.get("id")
    out["data_source"] = "polygon_market_proxy"
    return out


def dedupe(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for article in articles:
        key = str(article.get("id") or article_key(article))
        if key:
            by_key[key] = article
    return sorted(by_key.values(), key=lambda row: (row.get("published_utc") or "", row.get("id") or ""))


def write_summary(articles: list[dict[str, Any]], output_path: Path) -> None:
    import csv

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for article in articles:
        publisher = str((article.get("publisher") or {}).get("name") or "Unknown")
        proxy = ",".join(article.get("source_proxy_tickers") or ["Unknown"])
        key = (proxy, publisher)
        pub_date = str(article.get("published_utc") or "")[:10]
        row = rows.setdefault(
            key,
            {
                "proxy_tickers": proxy,
                "publisher": publisher,
                "articles": 0,
                "first_date": pub_date or "9999-99-99",
                "last_date": pub_date,
            },
        )
        row["articles"] += 1
        if pub_date:
            row["first_date"] = min(row["first_date"], pub_date)
            row["last_date"] = max(row["last_date"], pub_date)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["proxy_tickers", "publisher", "articles", "first_date", "last_date"])
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (-row["articles"], row["proxy_tickers"], row["publisher"])))


def run_pipeline(
    project_root: str | None,
    proxy_news_name: str,
    existing_market_news_name: str | None,
    output_name: str,
    summary_name: str,
    proxy_tickers: set[str],
) -> dict[str, Any]:
    root = resolve_project_root(project_root)
    raw_dir = root / "data" / "raw"
    tables_dir = root / "outputs" / "tables"

    proxy_raw = load_json_list(raw_dir / proxy_news_name)
    converted = [item for article in proxy_raw if (item := to_market_article(article, proxy_tickers))]
    existing = load_json_list(raw_dir / existing_market_news_name) if existing_market_news_name else []
    combined = dedupe(existing + converted)

    output_path = raw_dir / output_name
    summary_path = tables_dir / summary_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    write_summary(combined, summary_path)

    return {
        "proxy_raw_articles": len(proxy_raw),
        "converted_market_articles": len(dedupe(converted)),
        "existing_market_articles": len(existing),
        "combined_market_articles": len(combined),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Polygon ETF/index proxy news into synthetic __MARKET__ context news.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--proxy-news-name", default="polygon_market_proxy_news_2020_2024.json")
    parser.add_argument("--existing-market-news-name", default="google_news_rss_market_context_2020_2024.json")
    parser.add_argument("--output-name", default="combined_market_context_polygon_proxy_google_2020_2024.json")
    parser.add_argument("--summary-name", default="combined_market_context_polygon_proxy_google_2020_2024_summary.csv")
    parser.add_argument("--proxy-tickers", nargs="+", default=sorted(DEFAULT_PROXY_TICKERS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        project_root=args.project_root,
        proxy_news_name=args.proxy_news_name,
        existing_market_news_name=args.existing_market_news_name,
        output_name=args.output_name,
        summary_name=args.summary_name,
        proxy_tickers={ticker.upper() for ticker in args.proxy_tickers},
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
