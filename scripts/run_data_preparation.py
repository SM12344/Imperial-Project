from __future__ import annotations

import argparse
import json
import ssl
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
DEFAULT_BENCHMARK = "SPY"
MARKET_CONTEXT_TICKER = "__MARKET__"
DEFAULT_RELIABLE_PUBLISHERS = [
    "Reuters",
    "The Motley Fool",
    "Yahoo Finance",
    "Zacks Investment Research",
    "Zacks",
    "Benzinga",
    "MarketWatch",
    "Investing.com",
    "Investor's Business Daily",
    "Barrons",
    "Barron's",
    "The Wall Street Journal",
    "CNBC",
    "GlobeNewswire",
    "GlobeNewswire Inc.",
    "Business Wire",
    "PR Newswire",
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
]


@dataclass
class ProjectPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    tables_dir: Path
    cache_dir: Path


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def build_paths(project_root: str | None = None) -> ProjectPaths:
    root = resolve_project_root(project_root)
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    tables_dir = root / "outputs" / "tables"
    cache_dir = root / "data" / "cache"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, raw_dir=raw_dir, processed_dir=processed_dir, tables_dir=tables_dir, cache_dir=cache_dir)


def output_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}.csv" if suffix else f"{base}.csv"


def canonical_publisher(value: Any) -> str:
    return str(value or "").strip()


def parse_csv_arg(values: list[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parsed


MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "Â": "",
}


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return " ".join(text.split())


def build_article_text(article: dict[str, Any]) -> str:
    title = clean_text(article.get("title"))
    description = clean_text(article.get("description"))

    if article.get("data_source") == "google_news_rss":
        # Google RSS descriptions usually repeat the title plus source name.
        # Avoid scoring the same short headline twice.
        description = ""

    text_parts = [part for part in [title, description] if part]
    return ". ".join(text_parts)


def flatten_news(news_path: Path) -> pd.DataFrame:
    if not news_path.exists():
        raise FileNotFoundError(f"Raw news file not found: {news_path}")
    with news_path.open(encoding="utf-8") as f:
        news_raw = json.load(f)

    rows: list[dict[str, Any]] = []
    for article in news_raw:
        published_utc = article.get("published_utc")
        if not published_utc:
            continue
        text = build_article_text(article)
        insights = article.get("insights") or []
        if not insights:
            insights = [
                {
                    "ticker": ticker,
                    "sentiment": None,
                    "sentiment_reasoning": "Polygon ticker metadata; sentiment generated downstream by FinBERT.",
                }
                for ticker in article.get("tickers") or []
            ]
        for insight in insights:
            ticker = insight.get("ticker")
            if not ticker:
                continue
            rows.append(
                {
                    "article_id": str(article.get("id", "")),
                    "published_utc": published_utc,
                    "date": pd.to_datetime(published_utc, utc=True).date().isoformat(),
                    "ticker": str(ticker),
                    "sentiment": insight.get("sentiment"),
                    "title": clean_text(article.get("title")),
                    "description": clean_text(article.get("description")),
                    "text": text,
                    "publisher": (article.get("publisher") or {}).get("name"),
                    "article_url": article.get("article_url"),
                    "insight_reasoning": insight.get("sentiment_reasoning"),
                    "news_scope": article.get("news_scope") or "ticker_specific",
                    "query_topic": article.get("query_topic"),
                }
            )
    return pd.DataFrame(rows)


def select_target_news(
    news_flat: pd.DataFrame,
    tickers: list[str],
    allowed_publishers: list[str] | None = None,
    min_text_chars: int = 80,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    target = news_flat[news_flat["ticker"].isin(tickers)].copy()
    if start_date:
        target = target[target["date"] >= start_date].copy()
    if end_date:
        target = target[target["date"] < end_date].copy()
    if min_text_chars > 0:
        target = target[target["text"].fillna("").astype(str).str.len() >= min_text_chars].copy()
    if allowed_publishers:
        allowed = {canonical_publisher(name).casefold() for name in allowed_publishers}
        target = target[target["publisher"].map(lambda x: canonical_publisher(x).casefold()).isin(allowed)].copy()
    target = target.drop_duplicates(subset=["article_id", "ticker"]).copy()
    return target.sort_values(["date", "ticker", "article_id"]).reset_index(drop=True)


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    price_source: str = "yahoo_chart",
    allow_insecure_price_download: bool = False,
) -> pd.DataFrame:
    if price_source == "yahoo_chart":
        return download_prices_yahoo_chart(
            tickers,
            start_date,
            end_date,
            allow_insecure=allow_insecure_price_download,
        )
    if price_source != "yfinance":
        raise ValueError(f"Unsupported price source: {price_source}")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for price download. Install it before running this script.") from exc

    if cache_dir is not None and hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir.resolve()))

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        px = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if px.empty:
            continue
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px = px.reset_index()
        px.columns = [str(col).lower().replace(" ", "_") for col in px.columns]
        px["ticker"] = ticker
        rename_map = {"adj_close": "adj_close"}
        px = px.rename(columns=rename_map)
        required = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        missing = [col for col in required if col not in px.columns]
        if missing:
            raise ValueError(f"Missing price columns for {ticker}: {missing}")
        frames.append(px[required])

    if not frames:
        raise ValueError("No price data downloaded.")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date.astype(str)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_ssl_context(allow_insecure: bool = False) -> ssl.SSLContext:
    if allow_insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def yahoo_period(value: str) -> int:
    dt = datetime.combine(pd.to_datetime(value).date(), time.min, tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_prices_yahoo_chart(
    tickers: list[str],
    start_date: str,
    end_date: str,
    allow_insecure: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    context = build_ssl_context(allow_insecure=allow_insecure)
    period1 = yahoo_period(start_date)
    period2 = yahoo_period(end_date)

    for ticker in tickers:
        params = urlencode(
            {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
        request = Request(url, headers={"User-Agent": "imperial-news-sentiment-project/1.0"})
        with urlopen(request, timeout=60, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))

        chart = payload.get("chart") or {}
        error = chart.get("error")
        if error:
            raise RuntimeError(f"Yahoo chart error for {ticker}: {error}")
        results = chart.get("result") or []
        if not results:
            continue
        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        if not timestamps:
            continue

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).date.astype(str),
                "open": quote.get("open") or [],
                "high": quote.get("high") or [],
                "low": quote.get("low") or [],
                "close": quote.get("close") or [],
                "adj_close": adjclose,
                "volume": quote.get("volume") or [],
            }
        )
        df["ticker"] = ticker
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]])

    if not frames:
        raise ValueError("No price data downloaded.")
    prices = pd.concat(frames, ignore_index=True)
    return prices.dropna(subset=["adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker"):
        df = group.sort_values("date").copy()
        df["return_1d"] = df["adj_close"].pct_change()
        df["return_3d"] = df["adj_close"].pct_change(3)
        df["return_5d"] = df["adj_close"].pct_change(5)
        df["volatility_5d"] = df["return_1d"].rolling(5).std()
        df["volume_change_1d"] = df["volume"].pct_change()
        df["moving_avg_5d"] = df["adj_close"].rolling(5).mean()
        df["moving_avg_20d"] = df["adj_close"].rolling(20).mean()
        df["ma_5_20_gap"] = (df["moving_avg_5d"] - df["moving_avg_20d"]) / df["moving_avg_20d"]
        df["target_next_day_up"] = (df["return_1d"].shift(-1) > 0).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_existing_sentiment_proxy(news_target: pd.DataFrame) -> pd.DataFrame:
    if news_target.empty:
        return pd.DataFrame(columns=["ticker", "date", "news_count", "existing_sentiment_score_mean"])

    sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    tmp = news_target.copy()
    tmp["existing_sentiment_score"] = tmp["sentiment"].map(sentiment_map).fillna(0.0)
    daily = (
        tmp.groupby(["ticker", "date"], as_index=False)
        .agg(
            news_count=("article_id", "count"),
            existing_sentiment_score_mean=("existing_sentiment_score", "mean"),
            existing_sentiment_score_std=("existing_sentiment_score", "std"),
        )
    )
    daily["log_news_count"] = np.log1p(daily["news_count"])
    return daily


def build_model_dataset(
    price_features: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
) -> pd.DataFrame:
    stock_features = price_features[price_features["ticker"].isin(tickers)].copy()
    benchmark = price_features[price_features["ticker"] == benchmark_ticker].copy()
    spy_features = benchmark[["date", "return_1d", "return_5d", "volatility_5d"]].rename(
        columns={
            "return_1d": "spy_return_1d",
            "return_5d": "spy_return_5d",
            "volatility_5d": "spy_volatility_5d",
        }
    )

    model_df = stock_features.merge(daily_sentiment, on=["ticker", "date"], how="left")
    model_df = model_df.merge(spy_features, on="date", how="left")
    model_df = model_df.rename(columns={"date": "trading_date"})
    model_df["has_news"] = model_df["news_count"].notna().astype(int)

    fill_cols = [
        "news_count",
        "existing_sentiment_score_mean",
        "existing_sentiment_score_std",
        "log_news_count",
    ]
    for col in fill_cols:
        if col not in model_df.columns:
            model_df[col] = 0.0
        model_df[col] = model_df[col].fillna(0.0)
    return model_df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)


def build_news_coverage(news_target: pd.DataFrame) -> pd.DataFrame:
    if news_target.empty:
        return pd.DataFrame()
    return (
        news_target.groupby("ticker", as_index=False)
        .agg(
            rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["rows", "ticker"], ascending=[False, True])
    )


def build_publisher_coverage(news_target: pd.DataFrame) -> pd.DataFrame:
    if news_target.empty:
        return pd.DataFrame()
    return (
        news_target.assign(publisher=news_target["publisher"].fillna("Unknown"))
        .groupby("publisher", as_index=False)
        .agg(
            article_ticker_rows=("ticker", "size"),
            unique_articles=("article_id", "nunique"),
            news_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["article_ticker_rows", "unique_articles"], ascending=[False, False])
    )


def build_dataset_summary(
    model_df: pd.DataFrame,
    suffix: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    complete_rows = model_df.dropna().shape[0]
    return pd.DataFrame(
        [
            {
                "dataset_suffix": suffix,
                "start_date": start_date,
                "end_date": end_date,
                "tickers": ", ".join(tickers),
                "rows": len(model_df),
                "complete_rows": complete_rows,
                "rows_with_news": int(model_df["has_news"].sum()),
                "unique_dates": model_df["trading_date"].nunique(),
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )


def run_pipeline(
    project_root: str | None = None,
    news_path: str | None = None,
    market_news_path: str | None = None,
    tickers: list[str] | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK,
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    output_suffix: str = "2023",
    allowed_publishers: list[str] | None = None,
    min_text_chars: int = 80,
    price_source: str = "yahoo_chart",
    allow_insecure_price_download: bool = False,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    selected_tickers = tickers or DEFAULT_TICKERS
    all_price_tickers = list(dict.fromkeys([*selected_tickers, benchmark_ticker]))
    raw_news_path = Path(news_path).resolve() if news_path else paths.raw_dir / "polygon_news_sample.json"

    news_flat = flatten_news(raw_news_path)
    market_news_flat = pd.DataFrame()
    if market_news_path:
        market_raw_news_path = Path(market_news_path).resolve()
        market_news_flat = flatten_news(market_raw_news_path)
        news_flat = pd.concat([news_flat, market_news_flat], ignore_index=True)

    news_target = select_target_news(
        news_flat,
        selected_tickers,
        allowed_publishers=allowed_publishers,
        min_text_chars=min_text_chars,
        start_date=start_date,
        end_date=end_date,
    )
    if not market_news_flat.empty:
        market_target = select_target_news(
            market_news_flat,
            [MARKET_CONTEXT_TICKER],
            allowed_publishers=allowed_publishers,
            min_text_chars=min_text_chars,
            start_date=start_date,
            end_date=end_date,
        )
        news_target = (
            pd.concat([news_target, market_target], ignore_index=True)
            .sort_values(["date", "ticker", "article_id"])
            .reset_index(drop=True)
        )
    prices = download_prices(
        all_price_tickers,
        start_date,
        end_date,
        cache_dir=paths.cache_dir / "yfinance",
        price_source=price_source,
        allow_insecure_price_download=allow_insecure_price_download,
    )
    price_features = build_price_features(prices)
    daily_sentiment = build_existing_sentiment_proxy(news_target)
    model_df = build_model_dataset(price_features, daily_sentiment, selected_tickers, benchmark_ticker)
    coverage_df = build_news_coverage(news_target)
    publisher_coverage_df = build_publisher_coverage(news_target)
    summary_df = build_dataset_summary(model_df, output_suffix, selected_tickers, start_date, end_date)

    news_flat.to_csv(paths.processed_dir / output_name("news_flattened", output_suffix), index=False)
    news_target.to_csv(paths.processed_dir / output_name("news_target_tickers", output_suffix), index=False)
    prices.to_csv(paths.processed_dir / output_name("prices", output_suffix), index=False)
    price_features.to_csv(paths.processed_dir / output_name("price_features", output_suffix), index=False)
    daily_sentiment.to_csv(paths.processed_dir / output_name("daily_existing_sentiment_features", output_suffix), index=False)
    model_df.to_csv(paths.processed_dir / output_name("model_dataset_existing_sentiment_proxy", output_suffix), index=False)
    coverage_df.to_csv(paths.tables_dir / output_name("target_ticker_news_coverage", output_suffix), index=False)
    publisher_coverage_df.to_csv(paths.tables_dir / output_name("target_ticker_publisher_coverage", output_suffix), index=False)
    summary_df.to_csv(paths.tables_dir / output_name("model_dataset_summary", output_suffix), index=False)

    return {
        "news_flat": news_flat,
        "news_target": news_target,
        "prices": prices,
        "price_features": price_features,
        "daily_sentiment": daily_sentiment,
        "model_df": model_df,
        "coverage_df": coverage_df,
        "publisher_coverage_df": publisher_coverage_df,
        "summary_df": summary_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare price/news data for configurable ticker/date experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--news-path", default=None)
    parser.add_argument(
        "--market-news-path",
        default=None,
        help="Optional raw news JSON containing synthetic __MARKET__ macro/US-market context articles.",
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-01-01")
    parser.add_argument("--output-suffix", default="2023")
    parser.add_argument(
        "--allowed-publishers",
        nargs="*",
        default=None,
        help="Optional publisher allowlist. Accepts space-separated names or comma-separated groups.",
    )
    parser.add_argument(
        "--use-default-reliable-publishers",
        action="store_true",
        help="Filter news to a conservative built-in source allowlist.",
    )
    parser.add_argument("--min-text-chars", type=int, default=80)
    parser.add_argument("--price-source", choices=["yahoo_chart", "yfinance"], default="yahoo_chart")
    parser.add_argument(
        "--allow-insecure-price-download",
        action="store_true",
        help="Disable TLS verification for Yahoo chart downloads when local certificate interception prevents validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_publishers = parse_csv_arg(args.allowed_publishers)
    if args.use_default_reliable_publishers:
        allowed_publishers = DEFAULT_RELIABLE_PUBLISHERS

    outputs = run_pipeline(
        project_root=args.project_root,
        news_path=args.news_path,
        market_news_path=args.market_news_path,
        tickers=args.tickers,
        benchmark_ticker=args.benchmark_ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        output_suffix=args.output_suffix,
        allowed_publishers=allowed_publishers or None,
        min_text_chars=args.min_text_chars,
        price_source=args.price_source,
        allow_insecure_price_download=args.allow_insecure_price_download,
    )
    print(outputs["summary_df"].to_string(index=False))
    print()
    print(outputs["coverage_df"].to_string(index=False))
    if not outputs["publisher_coverage_df"].empty:
        print()
        print(outputs["publisher_coverage_df"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
