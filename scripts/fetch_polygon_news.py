from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_windows(start_date: str, end_date: str) -> list[tuple[date, date]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end <= start:
        raise ValueError("--end-date must be after --start-date")

    windows: list[tuple[date, date]] = []
    current = start
    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        stop = min(next_month, end)
        windows.append((current, stop))
        current = stop
    return windows


def build_ssl_context(allow_insecure: bool = False) -> ssl.SSLContext:
    if allow_insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def polygon_get(
    url: str,
    allow_insecure: bool = False,
    max_retries: int = 8,
    retry_base_seconds: float = 65.0,
) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "imperial-news-sentiment-project/1.0"})
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=60, context=build_ssl_context(allow_insecure=allow_insecure)) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else retry_base_seconds * (attempt + 1)
                print(f"Polygon rate limit hit; waiting {wait_seconds:.0f}s before retry {attempt + 1}/{max_retries}.")
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"Polygon HTTP {exc.code}: {body[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Polygon: {exc}") from exc
    raise RuntimeError("Polygon request failed after retries.")


def build_initial_url(ticker: str, start: date, stop: date, api_key: str, limit: int) -> str:
    # Polygon supports filter modifiers via published_utc.gte and published_utc.lt.
    params = {
        "ticker": ticker,
        "published_utc.gte": start.isoformat(),
        "published_utc.lt": stop.isoformat(),
        "order": "asc",
        "sort": "published_utc",
        "limit": str(limit),
        "apiKey": api_key,
    }
    return f"{POLYGON_NEWS_URL}?{urlencode(params)}"


def with_api_key(url: str, api_key: str) -> str:
    return url if "apiKey=" in url else f"{url}&apiKey={api_key}"


def fetch_ticker_window(
    ticker: str,
    start: date,
    stop: date,
    api_key: str,
    limit: int,
    sleep_seconds: float,
    allow_insecure: bool,
) -> list[dict[str, Any]]:
    url = build_initial_url(ticker, start, stop, api_key, limit)
    articles: list[dict[str, Any]] = []

    while url:
        payload = polygon_get(url, allow_insecure=allow_insecure)
        results = payload.get("results") or []
        articles.extend(results)
        next_url = payload.get("next_url")
        url = with_api_key(next_url, api_key) if next_url else ""
        if url and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return articles


def dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for article in articles:
        article_id = str(article.get("id") or article.get("article_url") or "")
        if not article_id:
            continue
        by_id[article_id] = article
    return sorted(by_id.values(), key=lambda item: (item.get("published_utc") or "", item.get("id") or ""))


def publisher_name(article: dict[str, Any]) -> str:
    return str((article.get("publisher") or {}).get("name") or "Unknown").strip()


def build_fetch_summary(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ticker_counts: dict[str, dict[str, Any]] = {}
    publisher_counts: dict[str, int] = {}

    for article in articles:
        published = article.get("published_utc") or ""
        pub_date = published[:10]
        publisher_counts[publisher_name(article)] = publisher_counts.get(publisher_name(article), 0) + 1
        for ticker in article.get("tickers") or []:
            row = ticker_counts.setdefault(
                ticker,
                {"ticker": ticker, "articles": 0, "first_date": pub_date, "last_date": pub_date},
            )
            row["articles"] += 1
            row["first_date"] = min(row["first_date"], pub_date)
            row["last_date"] = max(row["last_date"], pub_date)

    summary = sorted(ticker_counts.values(), key=lambda row: (-row["articles"], row["ticker"]))
    summary.append({"ticker": "__publishers__", "articles": len(publisher_counts), "first_date": "", "last_date": ""})
    for name, count in sorted(publisher_counts.items(), key=lambda item: (-item[1], item[0]))[:25]:
        summary.append({"ticker": f"publisher:{name}", "articles": count, "first_date": "", "last_date": ""})
    return summary


def write_csv_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "articles", "first_date", "last_date"])
        writer.writeheader()
        writer.writerows(rows)


def load_existing_articles(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    with output_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {output_path}")
    return [article for article in payload if isinstance(article, dict)]


def write_outputs(articles: list[dict[str, Any]], output_path: Path, summary_path: Path) -> list[dict[str, Any]]:
    deduped = dedupe_articles(articles)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    write_csv_summary(build_fetch_summary(deduped), summary_path)
    return deduped


def window_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".windows.json")


def load_completed_windows(output_path: Path) -> set[tuple[str, str, str]]:
    checkpoint_path = window_checkpoint_path(output_path)
    if not checkpoint_path.exists():
        return set()
    with checkpoint_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    completed: set[tuple[str, str, str]] = set()
    for row in payload:
        if isinstance(row, list) and len(row) == 3:
            completed.add((str(row[0]), str(row[1]), str(row[2])))
    return completed


def write_completed_windows(output_path: Path, completed: set[tuple[str, str, str]]) -> None:
    checkpoint_path = window_checkpoint_path(output_path)
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(completed), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch timestamped Polygon stock news for a fixed ticker universe.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-01-01")
    parser.add_argument("--output-name", default="polygon_news_2022_2023.json")
    parser.add_argument("--summary-name", default="polygon_news_2022_2023_fetch_summary.csv")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="POLYGON_API_KEY")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep-seconds", type=float, default=12.0)
    parser.add_argument(
        "--checkpoint-each-window",
        action="store_true",
        help="Write raw JSON and summary after each ticker/month window so rate-limited runs can resume.",
    )
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help="Disable TLS verification when local certificate interception prevents validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing Polygon API key. Set ${args.api_key_env} or pass --api-key. "
            "Do not commit the key into the project."
        )

    root = resolve_project_root(args.project_root)
    output_path = root / "data" / "raw" / args.output_name
    summary_path = root / "outputs" / "tables" / args.summary_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles = load_existing_articles(output_path)
    completed = load_completed_windows(output_path) | {
        (
            article.get("requested_ticker"),
            article.get("requested_window_start"),
            article.get("requested_window_stop"),
        )
        for article in all_articles
    }
    if all_articles:
        print(f"Loaded {len(all_articles)} existing articles from {output_path}")

    for ticker in args.tickers:
        for start, stop in month_windows(args.start_date, args.end_date):
            window_key = (ticker, start.isoformat(), stop.isoformat())
            if window_key in completed:
                print(f"{ticker} {start.isoformat()} to {stop.isoformat()}: already fetched")
                continue
            articles = fetch_ticker_window(
                ticker,
                start,
                stop,
                api_key,
                args.limit,
                args.sleep_seconds,
                args.allow_insecure_download,
            )
            for article in articles:
                article["requested_ticker"] = ticker
                article["requested_window_start"] = start.isoformat()
                article["requested_window_stop"] = stop.isoformat()
            all_articles.extend(articles)
            completed.add(window_key)
            print(f"{ticker} {start.isoformat()} to {stop.isoformat()}: {len(articles)} articles")
            if args.checkpoint_each_window:
                deduped = write_outputs(all_articles, output_path, summary_path)
                write_completed_windows(output_path, completed)
                print(f"Checkpointed {len(deduped)} unique articles")

    deduped = write_outputs(all_articles, output_path, summary_path)
    write_completed_windows(output_path, completed)
    print(f"Wrote {len(deduped)} unique articles to {output_path}")
    print(f"Wrote fetch summary to {summary_path}")


if __name__ == "__main__":
    main()
