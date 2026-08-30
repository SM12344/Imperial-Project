from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import ssl
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
MARKET_CONTEXT_TICKER = "__MARKET__"

QUERY_TERMS = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "TSLA": ["TSLA", "Tesla"],
    "AMZN": ["AMZN", "Amazon"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
}

MARKET_CONTEXT_QUERIES = {
    "fed_rates": '"Federal Reserve" OR "interest rates" OR "rate hike" OR "rate cut"',
    "inflation": 'inflation OR CPI OR PCE',
    "labor_growth": '"jobs report" OR payrolls OR unemployment OR GDP',
    "market_risk": '"S&P 500" OR Nasdaq OR "stock market" OR recession OR "Treasury yields"',
}

ALLOWED_SOURCES = {
    "Reuters",
    "CNBC",
    "Yahoo Finance",
    "The Motley Fool",
    "Zacks",
    "Zacks Investment Research",
    "Investing.com",
    "MarketWatch",
    "Barron's",
    "The Wall Street Journal",
}


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def build_ssl_context(allow_insecure: bool = False) -> ssl.SSLContext:
    if allow_insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def date_windows(start_date: str, end_date: str, window_days: int) -> list[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    windows: list[tuple[str, str]] = []
    current = start
    while current < end:
        stop = min(current + timedelta(days=window_days), end)
        windows.append((current.isoformat(), stop.isoformat()))
        current = stop
    return windows


def strip_html(value: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", value or "")
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_pubdate(value: str) -> str:
    dt = parsedate_to_datetime(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_query(ticker: str, start_date: str, end_date: str) -> str:
    terms = QUERY_TERMS.get(ticker, [ticker])
    term_group = " OR ".join(f'"{term}"' if " " in term else term for term in terms)
    return f"({term_group}) (stock OR shares OR earnings) after:{start_date} before:{end_date}"


def build_market_query(topic_query: str, start_date: str, end_date: str) -> str:
    return f"({topic_query}) (markets OR stocks OR economy OR investors) after:{start_date} before:{end_date}"


def fetch_rss(query: str, allow_insecure: bool, request_timeout: int) -> bytes:
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    url = f"{GOOGLE_NEWS_RSS_URL}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=request_timeout, context=build_ssl_context(allow_insecure=allow_insecure)) as response:
        return response.read()


def source_name(item: ET.Element) -> str:
    source = item.find("source")
    return source.text.strip() if source is not None and source.text else ""


def article_relevant(title: str, description: str, ticker: str) -> bool:
    if ticker == MARKET_CONTEXT_TICKER:
        return True
    haystack = f"{title} {description}".casefold()
    return any(term.casefold() in haystack for term in QUERY_TERMS.get(ticker, [ticker]))


def article_id(link: str, ticker: str) -> str:
    return hashlib.sha1(f"google_news_rss|{ticker}|{link}".encode("utf-8")).hexdigest()


def item_to_polygon_like(item: ET.Element, ticker: str, query_topic: str | None = None) -> dict[str, Any] | None:
    title = strip_html(item.findtext("title") or "")
    description = strip_html(item.findtext("description") or "")
    link = item.findtext("link") or ""
    publisher = source_name(item)
    pubdate = item.findtext("pubDate") or ""
    if not title or not link or not pubdate:
        return None
    if publisher not in ALLOWED_SOURCES:
        return None
    if not article_relevant(title, description, ticker):
        return None

    published_utc = parse_pubdate(pubdate)
    return {
        "article_url": link,
        "author": None,
        "description": description,
        "id": article_id(link, ticker),
        "image_url": None,
        "amp_url": None,
        "keywords": [],
        "published_utc": published_utc,
        "publisher": {
            "name": publisher,
            "homepage_url": None,
        },
        "tickers": [ticker],
        "insights": [
            {
                "ticker": ticker,
                "sentiment": None,
                "sentiment_reasoning": "Google News RSS metadata; sentiment generated downstream by FinBERT.",
            }
        ],
        "title": title,
        "data_source": "google_news_rss",
        "news_scope": "market_context" if ticker == MARKET_CONTEXT_TICKER else "ticker_specific",
        "query_topic": query_topic,
    }


def dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for article in articles:
        key = str(article.get("id") or "")
        if key:
            deduped[key] = article
    return sorted(deduped.values(), key=lambda item: (item.get("published_utc") or "", item.get("id") or ""))


def write_outputs(articles: list[dict[str, Any]], output_path: Path, summary_path: Path) -> None:
    import csv

    deduped = dedupe_articles(articles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for article in deduped:
        ticker = (article.get("tickers") or [""])[0]
        publisher = (article.get("publisher") or {}).get("name") or "Unknown"
        key = (ticker, publisher)
        row = rows.setdefault(
            key,
            {"ticker": ticker, "publisher": publisher, "articles": 0, "first_date": "9999-99-99", "last_date": ""},
        )
        pub_date = str(article.get("published_utc") or "")[:10]
        row["articles"] += 1
        row["first_date"] = min(row["first_date"], pub_date)
        row["last_date"] = max(row["last_date"], pub_date)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "publisher", "articles", "first_date", "last_date"])
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (row["ticker"], -row["articles"], row["publisher"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical metadata from Google News RSS search.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--market-context",
        action="store_true",
        help="Fetch broad US market/macro news as a synthetic __MARKET__ stream instead of ticker-specific news.",
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-01-01")
    parser.add_argument("--window-days", type=int, default=31)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--output-name", default="google_news_rss_2022_2023.json")
    parser.add_argument("--summary-name", default="google_news_rss_2022_2023_summary.csv")
    parser.add_argument("--allow-insecure-download", action="store_true")
    parser.add_argument("--checkpoint-each-request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root(args.project_root)
    output_path = root / "data" / "raw" / args.output_name
    summary_path = root / "outputs" / "tables" / args.summary_name

    all_articles: list[dict[str, Any]] = []
    windows = date_windows(args.start_date, args.end_date, args.window_days)
    requests_to_run: list[tuple[str, str | None, str, str]] = []
    if args.market_context:
        for start, end in windows:
            for topic_name, topic_query in MARKET_CONTEXT_QUERIES.items():
                requests_to_run.append((MARKET_CONTEXT_TICKER, topic_name, start, end))
    else:
        for ticker in args.tickers:
            for start, end in windows:
                requests_to_run.append((ticker, None, start, end))
    total = len(requests_to_run)
    request_count = 0
    for ticker, topic_name, start, end in requests_to_run:
        request_count += 1
        query = build_market_query(MARKET_CONTEXT_QUERIES[str(topic_name)], start, end) if args.market_context else build_query(ticker, start, end)
        body = fetch_rss(query, args.allow_insecure_download, args.request_timeout)
        root_xml = ET.fromstring(body)
        raw_items = root_xml.findall("./channel/item")
        kept = []
        for item in raw_items:
            article = item_to_polygon_like(item, ticker, query_topic=topic_name)
            if article is not None:
                kept.append(article)
        all_articles.extend(kept)
        label = f"{ticker}:{topic_name}" if topic_name else ticker
        print(f"{request_count}/{total} {label} {start} to {end}: {len(kept)} kept from {len(raw_items)}", flush=True)
        if args.checkpoint_each_request:
            write_outputs(all_articles, output_path, summary_path)
        if request_count < total:
            time.sleep(args.sleep_seconds)

    write_outputs(all_articles, output_path, summary_path)
    print(f"Wrote {len(dedupe_articles(all_articles))} articles to {output_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
