from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERY_TERMS = {
    "AAPL": '("Apple Inc" OR AAPL) (stock OR shares OR earnings)',
    "MSFT": '(Microsoft OR MSFT) (stock OR shares OR earnings)',
    "TSLA": '(Tesla OR TSLA) (stock OR shares OR earnings)',
    "AMZN": '(Amazon OR AMZN) (stock OR shares OR earnings)',
    "NVDA": '(Nvidia OR NVDA) (stock OR shares OR earnings)',
}

ALLOWED_DOMAINS = {
    "finance.yahoo.com",
    "fool.com",
    "zacks.com",
    "benzinga.com",
    "investing.com",
    "marketwatch.com",
    "reuters.com",
    "cnbc.com",
    "wsj.com",
    "barrons.com",
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


def gdelt_datetime(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")[:14]


def parse_gdelt_seen_date(value: str) -> str:
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return value


def domain_allowed(domain: str, allowed_domains: set[str]) -> bool:
    normalized = domain.lower().removeprefix("www.")
    return any(normalized == allowed or normalized.endswith(f".{allowed}") for allowed in allowed_domains)


def build_url(query: str, start_date: str, end_date: str, maxrecords: int) -> str:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": gdelt_datetime(f"{start_date}T00:00:00Z"),
        "enddatetime": gdelt_datetime(f"{end_date}T00:00:00Z"),
        "maxrecords": str(maxrecords),
        "sort": "hybridrel",
    }
    return f"{GDELT_DOC_URL}?{urlencode(params)}"


def date_windows(start_date: str, end_date: str, window_days: int) -> list[tuple[str, str]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end <= start:
        raise ValueError("--end-date must be after --start-date")
    if window_days <= 0:
        return [(start.isoformat(), end.isoformat())]

    windows: list[tuple[str, str]] = []
    current = start
    while current < end:
        stop = min(current + timedelta(days=window_days), end)
        windows.append((current.isoformat(), stop.isoformat()))
        current = stop
    return windows


def fetch_json(
    url: str,
    allow_insecure: bool,
    retries: int,
    sleep_seconds: float,
    request_timeout: int,
) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "imperial-news-sentiment-project/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=request_timeout, context=build_ssl_context(allow_insecure=allow_insecure)) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == retries:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"GDELT HTTP {exc.code}: {body[:500]}") from exc
            wait = sleep_seconds * (attempt + 2)
            print(f"GDELT rate limited; waiting {wait:.1f}s before retry", flush=True)
            time.sleep(wait)
        except URLError as exc:
            last_error = exc
            if attempt == retries:
                raise RuntimeError(f"Could not reach GDELT: {exc}") from exc
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"GDELT request failed: {last_error}")


def article_id(url: str, ticker: str) -> str:
    return hashlib.sha1(f"{ticker}|{url}".encode("utf-8")).hexdigest()


def to_polygon_like(article: dict[str, Any], ticker: str) -> dict[str, Any]:
    url = str(article.get("url") or "")
    title = str(article.get("title") or "").strip()
    domain = str(article.get("domain") or "Unknown").strip()
    published_utc = parse_gdelt_seen_date(str(article.get("seendate") or ""))
    return {
        "article_url": url,
        "author": None,
        "description": title,
        "id": article_id(url, ticker),
        "image_url": article.get("socialimage"),
        "amp_url": None,
        "keywords": [],
        "published_utc": published_utc,
        "publisher": {
            "name": domain,
            "homepage_url": f"https://{domain}" if domain and domain != "Unknown" else None,
        },
        "tickers": [ticker],
        "insights": [
            {
                "ticker": ticker,
                "sentiment": None,
                "sentiment_reasoning": "GDELT article-list result; sentiment generated downstream by FinBERT.",
            }
        ],
        "title": title,
        "data_source": "gdelt_doc_api",
        "source_country": article.get("sourceCountry"),
        "language": article.get("language"),
    }


def dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for article in articles:
        key = str(article.get("id") or article.get("article_url") or "")
        if key:
            deduped[key] = article
    return sorted(deduped.values(), key=lambda item: (item.get("published_utc") or "", item.get("id") or ""))


def write_articles(articles: list[dict[str, Any]], output_path: Path, summary_path: Path) -> None:
    deduped = dedupe_articles(articles)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    write_summary(deduped, summary_path)


def write_summary(articles: list[dict[str, Any]], output_path: Path) -> None:
    import csv

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for article in articles:
        ticker = ((article.get("tickers") or [""])[0])
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "publisher", "articles", "first_date", "last_date"])
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (row["ticker"], -row["articles"], row["publisher"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch no-key historical article metadata from the GDELT DOC API.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-01-01")
    parser.add_argument("--output-name", default="gdelt_news_2022_2023.json")
    parser.add_argument("--summary-name", default="gdelt_news_2022_2023_fetch_summary.csv")
    parser.add_argument("--maxrecords", type=int, default=250)
    parser.add_argument("--sleep-seconds", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--window-days", type=int, default=0)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--checkpoint-each-request", action="store_true")
    parser.add_argument("--allow-insecure-download", action="store_true")
    parser.add_argument("--include-all-domains", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root(args.project_root)
    output_path = root / "data" / "raw" / args.output_name
    summary_path = root / "outputs" / "tables" / args.summary_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict[str, Any]] = []
    domains = args.domains or [None]
    windows = date_windows(args.start_date, args.end_date, args.window_days)
    request_count = 0
    total_requests = len(args.tickers) * len(domains) * len(windows)

    for ticker in args.tickers:
        base_query = QUERY_TERMS.get(ticker, f"{ticker} stock")
        for domain in domains:
            query = f"{base_query} domain:{domain}" if domain else base_query
            for window_start, window_end in windows:
                request_count += 1
                url = build_url(query, window_start, window_end, args.maxrecords)
                payload = fetch_json(
                    url,
                    args.allow_insecure_download,
                    args.retries,
                    args.sleep_seconds,
                    args.request_timeout,
                )
                raw_articles = payload.get("articles") or []
                kept = []
                for article in raw_articles:
                    article_domain = str(article.get("domain") or "")
                    if args.include_all_domains or domain or domain_allowed(article_domain, ALLOWED_DOMAINS):
                        kept.append(to_polygon_like(article, ticker))
                all_articles.extend(kept)
                print(
                    f"{request_count}/{total_requests} {ticker} {domain or 'all-domains'} "
                    f"{window_start} to {window_end}: {len(kept)} kept from {len(raw_articles)}",
                    flush=True,
                )
                if args.checkpoint_each_request:
                    write_articles(all_articles, output_path, summary_path)
                if request_count < total_requests:
                    time.sleep(args.sleep_seconds)

    write_articles(all_articles, output_path, summary_path)
    deduped = dedupe_articles(all_articles)
    print(f"Wrote {len(deduped)} GDELT-derived articles to {output_path}", flush=True)
    print(f"Wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
