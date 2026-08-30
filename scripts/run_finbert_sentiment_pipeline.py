from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional at import time
    torch = None

from transformers import AutoModelForSequenceClassification, AutoTokenizer


TARGET_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
BENCHMARK_TICKER = "SPY"
MARKET_CONTEXT_TICKER = "__MARKET__"
MODEL_NAME = "ProsusAI/finbert"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR = 16


@dataclass
class ProjectPaths:
    root: Path
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
    processed_dir = root / "data" / "processed"
    tables_dir = root / "outputs" / "tables"
    cache_dir = root / "models" / "finbert"

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        cache_dir=cache_dir,
    )


def override_cache_dir(paths: ProjectPaths, cache_dir: str | None) -> ProjectPaths:
    if not cache_dir:
        return paths
    selected = Path(cache_dir)
    if not selected.is_absolute():
        selected = paths.root / selected
    selected.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        root=paths.root,
        processed_dir=paths.processed_dir,
        tables_dir=paths.tables_dir,
        cache_dir=selected,
    )


def output_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}.csv" if suffix else f"{base}.csv"


def load_required_inputs(
    paths: ProjectPaths,
    news_input_name: str,
    price_features_input_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = paths.processed_dir / news_input_name
    price_features_path = paths.processed_dir / price_features_input_name

    missing = [str(p) for p in [news_path, price_features_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files. Re-run notebook 02 first.\n"
            + "\n".join(missing)
        )

    news_target = pd.read_csv(news_path)
    price_features = pd.read_csv(price_features_path)

    news_target["text"] = news_target["text"].fillna("").astype(str)
    news_target["article_id"] = news_target["article_id"].astype(str)
    news_target["date"] = news_target["date"].astype(str)
    price_features["date"] = price_features["date"].astype(str)

    return news_target, price_features


def load_existing_finbert_scores(paths: ProjectPaths, scored_input_name: str) -> pd.DataFrame:
    scored_path = paths.processed_dir / scored_input_name
    if not scored_path.exists():
        raise FileNotFoundError(
            f"Existing scored file not found: {scored_path}\n"
            "Run the FinBERT scoring step once before using --reuse-existing-scores."
        )

    scored = pd.read_csv(scored_path)
    required_cols = [
        "article_id",
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "finbert_sentiment_score",
        "finbert_predicted_label",
    ]
    missing_cols = [col for col in required_cols if col not in scored.columns]
    if missing_cols:
        raise ValueError(
            "Existing scored file is missing required columns:\n" + "\n".join(missing_cols)
        )
    return scored[required_cols].drop_duplicates(subset=["article_id"]).copy()


def detect_device() -> int:
    if torch is not None and torch.cuda.is_available():
        return 0
    return -1


def canonical_label(label: str, id_aliases: dict[str, str]) -> str:
    base = str(label).strip().lower()
    aliases = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "pos": "positive",
        "neg": "negative",
    }
    return aliases.get(id_aliases.get(base, base), id_aliases.get(base, base))


def build_trading_calendar(price_features: pd.DataFrame, tickers: list[str]) -> pd.DatetimeIndex:
    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(
            price_features[price_features["ticker"].isin(tickers)]["date"]
        )
        .drop_duplicates()
        .sort_values()
    )
    if trading_dates.empty:
        raise ValueError("Trading calendar is empty.")
    return trading_dates


def prepare_local_safetensors_model_dir(cache_dir: Path) -> Path | None:
    snapshots_root = cache_dir / "models--ProsusAI--finbert" / "snapshots"
    if not snapshots_root.exists():
        return None

    config_source = None
    safetensors_source = None
    for snapshot_dir in snapshots_root.iterdir():
        if not snapshot_dir.is_dir():
            continue
        if (snapshot_dir / "config.json").exists() and (snapshot_dir / "vocab.txt").exists():
            config_source = snapshot_dir
        if (snapshot_dir / "model.safetensors").exists():
            safetensors_source = snapshot_dir

    if config_source is None or safetensors_source is None:
        return None

    target_dir = cache_dir / "local-finbert-safe"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]:
        source = config_source / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)
    shutil.copy2(safetensors_source / "model.safetensors", target_dir / "model.safetensors")
    return target_dir


def align_news_to_trading_date(
    news_target: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    news = news_target.copy()
    published_utc = pd.to_datetime(news["published_utc"], utc=True)
    published_ny = published_utc.dt.tz_convert(MARKET_TIMEZONE)

    news["published_datetime_utc"] = published_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    news["published_datetime_new_york"] = published_ny.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    news["after_market_close"] = (
        (published_ny.dt.hour > MARKET_CLOSE_HOUR)
        | ((published_ny.dt.hour == MARKET_CLOSE_HOUR) & (published_ny.dt.minute >= 0))
    ).astype(int)

    candidate_dates = (
        published_ny.dt.tz_localize(None).dt.normalize()
        + pd.to_timedelta(news["after_market_close"], unit="D")
    )
    news["candidate_trading_date"] = candidate_dates.dt.strftime("%Y-%m-%d")

    insert_positions = trading_calendar.searchsorted(candidate_dates.to_numpy(), side="left")
    aligned = pd.Series(pd.NaT, index=news.index, dtype="datetime64[ns]")
    valid_mask = insert_positions < len(trading_calendar)
    if valid_mask.any():
        aligned.loc[valid_mask] = trading_calendar.take(insert_positions[valid_mask]).to_numpy()

    news["aligned_trading_date"] = aligned.dt.strftime("%Y-%m-%d")
    news["shifted_to_future_trading_day"] = (
        news["aligned_trading_date"].notna()
        & (news["aligned_trading_date"] != news["date"])
    ).astype(int)
    news["shifted_due_to_after_market_close"] = (
        (news["after_market_close"] == 1)
        & news["aligned_trading_date"].notna()
        & (news["aligned_trading_date"] != news["date"])
    ).astype(int)
    news["shifted_due_to_non_trading_day"] = (
        (news["after_market_close"] == 0)
        & news["aligned_trading_date"].notna()
        & (news["aligned_trading_date"] != news["date"])
    ).astype(int)

    alignment_summary = pd.DataFrame(
        [
            {
                "rows": len(news),
                "unique_articles": news["article_id"].nunique(),
                "after_market_close_rows": int(news["after_market_close"].sum()),
                "shifted_to_future_trading_day_rows": int(news["shifted_to_future_trading_day"].sum()),
                "shifted_due_to_after_market_close_rows": int(news["shifted_due_to_after_market_close"].sum()),
                "shifted_due_to_non_trading_day_rows": int(news["shifted_due_to_non_trading_day"].sum()),
                "unmatched_rows": int(news["aligned_trading_date"].isna().sum()),
            }
        ]
    )
    return news, alignment_summary


def load_finbert_classifier(
    model_name: str = MODEL_NAME,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
):
    if torch is None:
        raise ImportError("PyTorch is required for FinBERT inference in this script.")

    model_source = model_name
    use_safetensors = None
    if local_files_only and cache_dir is not None and model_name == MODEL_NAME:
        safe_dir = prepare_local_safetensors_model_dir(cache_dir)
        if safe_dir is not None:
            model_source = str(safe_dir.resolve())
            use_safetensors = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
    )
    device = detect_device()
    if device == 0 and torch is not None:
        model = model.to("cuda")
    model.eval()
    id_aliases = {
        f"label_{idx}".lower(): str(name).strip().lower()
        for idx, name in model.config.id2label.items()
    }
    return tokenizer, model, id_aliases


def score_unique_articles(
    news_target: pd.DataFrame,
    model_name: str = MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 512,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> pd.DataFrame:
    article_text = (
        news_target[["article_id", "published_utc", "text"]]
        .drop_duplicates(subset=["article_id"])
        .copy()
        .reset_index(drop=True)
    )

    tokenizer, model, id_aliases = load_finbert_classifier(
        model_name=model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    device = "cuda" if detect_device() == 0 and torch is not None else "cpu"

    scored_batches: list[dict[str, Any]] = []
    for start in range(0, len(article_text), batch_size):
        stop = min(start + batch_size, len(article_text))
        batch = article_text.iloc[start:stop].copy()
        encoded = tokenizer(
            batch["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        predictions: list[list[dict[str, float | str]]] = []
        for row_scores in probabilities:
            pred = []
            for idx, score in enumerate(row_scores):
                raw_label = model.config.id2label[idx]
                pred.append({"label": raw_label, "score": float(score)})
            predictions.append(pred)

        for (_, row), pred in zip(batch.iterrows(), predictions):
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for item in pred:
                label = canonical_label(item["label"], id_aliases)
                if label in scores:
                    scores[label] = float(item["score"])
            scored_batches.append(
                {
                    "article_id": row["article_id"],
                    "published_utc": row["published_utc"],
                    "text": row["text"],
                    "finbert_positive": scores["positive"],
                    "finbert_negative": scores["negative"],
                    "finbert_neutral": scores["neutral"],
                    "finbert_sentiment_score": scores["positive"] - scores["negative"],
                }
            )

    article_scores = pd.DataFrame(scored_batches)
    article_scores["finbert_predicted_label"] = article_scores[
        ["finbert_positive", "finbert_negative", "finbert_neutral"]
    ].idxmax(axis=1).str.replace("finbert_", "", regex=False)
    return article_scores


def build_daily_finbert_features(news_scored: pd.DataFrame) -> pd.DataFrame:
    daily = (
        news_scored.dropna(subset=["aligned_trading_date"])
        .groupby(["ticker", "aligned_trading_date"])
        .agg(
            news_count=("article_id", "count"),
            finbert_positive_mean=("finbert_positive", "mean"),
            finbert_negative_mean=("finbert_negative", "mean"),
            finbert_neutral_mean=("finbert_neutral", "mean"),
            finbert_sentiment_score_mean=("finbert_sentiment_score", "mean"),
            finbert_positive_std=("finbert_positive", "std"),
            finbert_negative_std=("finbert_negative", "std"),
            finbert_neutral_std=("finbert_neutral", "std"),
            finbert_sentiment_score_std=("finbert_sentiment_score", "std"),
            finbert_positive_max=("finbert_positive", "max"),
            finbert_negative_max=("finbert_negative", "max"),
            finbert_neutral_max=("finbert_neutral", "max"),
            finbert_sentiment_score_max=("finbert_sentiment_score", "max"),
            finbert_sentiment_score_min=("finbert_sentiment_score", "min"),
        )
        .reset_index()
        .rename(columns={"aligned_trading_date": "date"})
    )
    fill_zero_cols = [
        "finbert_positive_std",
        "finbert_negative_std",
        "finbert_neutral_std",
        "finbert_sentiment_score_std",
    ]
    daily[fill_zero_cols] = daily[fill_zero_cols].fillna(0)
    daily["log_news_count"] = np.log1p(daily["news_count"])
    daily["news_count_above_1"] = (daily["news_count"] > 1).astype(int)
    daily["news_count_above_2"] = (daily["news_count"] > 2).astype(int)
    return daily


def add_sentiment_lag_features(model_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.sort_values(["ticker", "trading_date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["news_count_lag1"] = g["news_count"].shift(1)
    df["news_count_rolling3"] = (
        g["news_count"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["news_count_rolling5"] = (
        g["news_count"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["finbert_sentiment_score_lag1"] = g["finbert_sentiment_score_mean"].shift(1)
    df["finbert_sentiment_score_rolling3"] = (
        g["finbert_sentiment_score_mean"]
        .shift(1)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["finbert_sentiment_score_rolling5"] = (
        g["finbert_sentiment_score_mean"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["finbert_positive_mean_lag1"] = g["finbert_positive_mean"].shift(1)
    df["finbert_negative_mean_lag1"] = g["finbert_negative_mean"].shift(1)
    df["finbert_neutral_mean_lag1"] = g["finbert_neutral_mean"].shift(1)
    df["finbert_positive_mean_rolling5"] = (
        g["finbert_positive_mean"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["finbert_negative_mean_rolling5"] = (
        g["finbert_negative_mean"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["finbert_neutral_mean_rolling5"] = (
        g["finbert_neutral_mean"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    df["finbert_sentiment_score_surprise"] = (
        df["finbert_sentiment_score_mean"] - df["finbert_sentiment_score_lag1"]
    )
    df["finbert_positive_mean_surprise"] = df["finbert_positive_mean"] - df["finbert_positive_mean_lag1"]
    df["finbert_negative_mean_surprise"] = df["finbert_negative_mean"] - df["finbert_negative_mean_lag1"]

    df["sentiment_x_has_news"] = df["finbert_sentiment_score_mean"] * df["has_news"]
    df["sentiment_x_volatility_5d"] = df["finbert_sentiment_score_mean"] * df["volatility_5d"].fillna(0)
    df["sentiment_x_abs_return_1d"] = df["finbert_sentiment_score_mean"] * df["return_1d"].abs().fillna(0)

    market_lag_cols: list[str] = []
    if "market_finbert_sentiment_score_mean" in df.columns:
        df["market_finbert_sentiment_score_lag1"] = g["market_finbert_sentiment_score_mean"].shift(1)
        df["market_finbert_sentiment_score_rolling3"] = (
            g["market_finbert_sentiment_score_mean"]
            .shift(1)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["market_finbert_sentiment_score_rolling5"] = (
            g["market_finbert_sentiment_score_mean"]
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["market_news_count_lag1"] = g["market_news_count"].shift(1)
        df["market_news_count_rolling5"] = (
            g["market_news_count"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df["market_sentiment_x_spy_return_1d"] = (
            df["market_finbert_sentiment_score_mean"] * df["spy_return_1d"].fillna(0)
        )
        market_lag_cols = [
            "market_finbert_sentiment_score_lag1",
            "market_finbert_sentiment_score_rolling3",
            "market_finbert_sentiment_score_rolling5",
            "market_news_count_lag1",
            "market_news_count_rolling5",
            "market_sentiment_x_spy_return_1d",
        ]

    lag_cols = [
        "news_count_lag1",
        "news_count_rolling3",
        "news_count_rolling5",
        "finbert_sentiment_score_lag1",
        "finbert_sentiment_score_rolling3",
        "finbert_sentiment_score_rolling5",
        "finbert_positive_mean_lag1",
        "finbert_negative_mean_lag1",
        "finbert_neutral_mean_lag1",
        "finbert_positive_mean_rolling5",
        "finbert_negative_mean_rolling5",
        "finbert_neutral_mean_rolling5",
        "finbert_sentiment_score_surprise",
        "finbert_positive_mean_surprise",
        "finbert_negative_mean_surprise",
    ]
    lag_cols.extend(market_lag_cols)
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def build_model_dataset(
    price_features: pd.DataFrame,
    daily_finbert: pd.DataFrame,
    tickers: list[str],
    benchmark_ticker: str,
    market_context_ticker: str = MARKET_CONTEXT_TICKER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_features = price_features[price_features["ticker"].isin(tickers)].copy()
    spy_features = price_features[price_features["ticker"] == benchmark_ticker][
        ["date", "return_1d", "return_5d", "volatility_5d"]
    ].rename(
        columns={
            "return_1d": "spy_return_1d",
            "return_5d": "spy_return_5d",
            "volatility_5d": "spy_volatility_5d",
        }
    )

    stock_daily_finbert = daily_finbert[daily_finbert["ticker"].isin(tickers)].copy()
    market_daily_finbert = daily_finbert[daily_finbert["ticker"] == market_context_ticker].copy()

    model_df = stock_features.merge(stock_daily_finbert, on=["ticker", "date"], how="left")
    if not market_daily_finbert.empty:
        market_feature_cols = [
            col for col in market_daily_finbert.columns if col not in {"ticker", "date"}
        ]
        market_daily_finbert = market_daily_finbert[["date", *market_feature_cols]].rename(
            columns={col: f"market_{col}" for col in market_feature_cols}
        )
        model_df = model_df.merge(market_daily_finbert, on="date", how="left")
    model_df = model_df.merge(spy_features, on="date", how="left")
    model_df = model_df.rename(columns={"date": "trading_date"})

    finbert_cols = [
        "news_count",
        "finbert_positive_mean",
        "finbert_negative_mean",
        "finbert_neutral_mean",
        "finbert_sentiment_score_mean",
    ]
    model_df["has_news"] = model_df["news_count"].notna().astype(int)
    model_df[finbert_cols] = model_df[finbert_cols].fillna(0)
    market_cols = [col for col in model_df.columns if col.startswith("market_")]
    if market_cols:
        model_df["has_market_news"] = model_df["market_news_count"].notna().astype(int)
        model_df[market_cols] = model_df[market_cols].fillna(0)
    else:
        model_df["has_market_news"] = 0
    model_df = add_sentiment_lag_features(model_df)

    complete_model_df = model_df.dropna(
        subset=["return_5d", "moving_avg_20d", "spy_return_5d", "target_next_day_up"]
    ).copy()

    return model_df, complete_model_df


def build_summary_table(model_df: pd.DataFrame, complete_model_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "model_dataset_finbert",
                "rows": len(model_df),
                "complete_rows": len(complete_model_df),
                "rows_with_news": int(model_df["has_news"].sum()),
                "tickers": ", ".join(tickers),
            }
        ]
    )


def build_article_score_summary(article_scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "articles_scored": len(article_scores),
                "first_timestamp": article_scores["published_utc"].min(),
                "last_timestamp": article_scores["published_utc"].max(),
                "mean_finbert_positive": article_scores["finbert_positive"].mean(),
                "mean_finbert_negative": article_scores["finbert_negative"].mean(),
                "mean_finbert_neutral": article_scores["finbert_neutral"].mean(),
            }
        ]
    )


def build_daily_coverage_table(daily_finbert: pd.DataFrame) -> pd.DataFrame:
    return (
        daily_finbert.groupby("ticker")
        .agg(
            news_days=("date", "nunique"),
            total_articles=("news_count", "sum"),
            avg_daily_news_count=("news_count", "mean"),
        )
        .reset_index()
        .sort_values(["total_articles", "ticker"], ascending=[False, True])
    )


def build_run_metadata(
    batch_size: int,
    max_length: int,
    model_name: str,
    device: int,
    local_files_only: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "batch_size": batch_size,
                "max_length": max_length,
                "device": "cuda:0" if device == 0 else "cpu",
                "local_files_only": int(local_files_only),
                "market_timezone": "America/New_York",
                "market_close_hour_local": MARKET_CLOSE_HOUR,
            }
        ]
    )


def save_outputs(
    paths: ProjectPaths,
    article_scores: pd.DataFrame,
    news_scored: pd.DataFrame,
    daily_finbert: pd.DataFrame,
    model_df: pd.DataFrame,
    complete_model_df: pd.DataFrame,
    summary: pd.DataFrame,
    article_summary: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    output_suffix: str,
) -> None:
    article_scores.to_csv(paths.processed_dir / output_name("finbert_article_scores", output_suffix), index=False)
    news_scored.to_csv(paths.processed_dir / output_name("news_target_tickers_finbert_scored", output_suffix), index=False)
    daily_finbert.to_csv(paths.processed_dir / output_name("daily_finbert_sentiment_features", output_suffix), index=False)
    model_df.to_csv(paths.processed_dir / output_name("model_dataset_finbert", output_suffix), index=False)
    complete_model_df.to_csv(paths.processed_dir / output_name("model_dataset_finbert_complete", output_suffix), index=False)

    summary.to_csv(paths.tables_dir / output_name("model_dataset_finbert_summary", output_suffix), index=False)
    article_summary.to_csv(paths.tables_dir / output_name("finbert_article_score_summary", output_suffix), index=False)
    daily_coverage.to_csv(paths.tables_dir / output_name("finbert_daily_coverage", output_suffix), index=False)
    alignment_summary.to_csv(paths.tables_dir / output_name("finbert_alignment_summary", output_suffix), index=False)
    metadata.to_csv(paths.tables_dir / output_name("finbert_run_metadata", output_suffix), index=False)


def run_pipeline(
    project_root: str | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    model_name: str = MODEL_NAME,
    local_files_only: bool = False,
    reuse_existing_scores: bool = False,
    news_input_name: str = "news_target_tickers.csv",
    price_features_input_name: str = "price_features_2023.csv",
    scored_input_name: str = "news_target_tickers_finbert_scored.csv",
    output_suffix: str = "",
    tickers: list[str] | None = None,
    benchmark_ticker: str = BENCHMARK_TICKER,
    model_cache_dir: str | None = None,
) -> dict[str, Any]:
    paths = override_cache_dir(build_paths(project_root), model_cache_dir)
    selected_tickers = tickers or TARGET_TICKERS
    news_target, price_features = load_required_inputs(paths, news_input_name, price_features_input_name)
    trading_calendar = build_trading_calendar(price_features, selected_tickers)
    aligned_news, alignment_summary = align_news_to_trading_date(news_target, trading_calendar)

    if reuse_existing_scores:
        article_scores = (
            aligned_news[["article_id", "published_utc", "text"]]
            .drop_duplicates(subset=["article_id"])
            .merge(load_existing_finbert_scores(paths, scored_input_name), on="article_id", how="inner", validate="one_to_one")
        )
    else:
        article_scores = score_unique_articles(
            news_target=aligned_news,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=paths.cache_dir,
            local_files_only=local_files_only,
        )

    news_scored = aligned_news.merge(
        article_scores.drop(columns=[c for c in ["published_utc", "text"] if c in article_scores.columns]),
        on="article_id",
        how="left",
        validate="many_to_one",
    )
    daily_finbert = build_daily_finbert_features(news_scored)
    model_df, complete_model_df = build_model_dataset(price_features, daily_finbert, selected_tickers, benchmark_ticker)

    summary = build_summary_table(model_df, complete_model_df, selected_tickers)
    article_summary = build_article_score_summary(article_scores)
    daily_coverage = build_daily_coverage_table(daily_finbert)
    metadata = build_run_metadata(
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name,
        device=detect_device(),
        local_files_only=local_files_only,
    )

    save_outputs(
        paths=paths,
        article_scores=article_scores,
        news_scored=news_scored,
        daily_finbert=daily_finbert,
        model_df=model_df,
        complete_model_df=complete_model_df,
        summary=summary,
        article_summary=article_summary,
        daily_coverage=daily_coverage,
        alignment_summary=alignment_summary,
        metadata=metadata,
        output_suffix=output_suffix,
    )

    return {
        "paths": paths,
        "news_target": aligned_news,
        "article_scores": article_scores,
        "news_scored": news_scored,
        "daily_finbert": daily_finbert,
        "model_df": model_df,
        "complete_model_df": complete_model_df,
        "summary": summary,
        "article_summary": article_summary,
        "daily_coverage": daily_coverage,
        "alignment_summary": alignment_summary,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FinBERT article scoring and rebuild the modelling dataset."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project folder containing data/processed and outputs/tables. Defaults to the current directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of unique articles to score per inference batch.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length passed to FinBERT after tokenizer truncation.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the FinBERT model to already exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local path to a downloaded FinBERT model directory. Use this for fully offline or manually cached runs.",
    )
    parser.add_argument(
        "--reuse-existing-scores",
        action="store_true",
        help="Skip FinBERT inference and rebuild the aligned dataset from an existing news_target_tickers_finbert_scored.csv file.",
    )
    parser.add_argument("--news-input-name", default="news_target_tickers.csv")
    parser.add_argument("--price-features-input-name", default="price_features_2023.csv")
    parser.add_argument("--scored-input-name", default="news_target_tickers_finbert_scored.csv")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--tickers", nargs="+", default=TARGET_TICKERS)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--model-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = str(Path(args.model_path).resolve()) if args.model_path else MODEL_NAME
    outputs = run_pipeline(
        project_root=args.project_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_name=model_name,
        local_files_only=args.local_files_only,
        reuse_existing_scores=args.reuse_existing_scores,
        news_input_name=args.news_input_name,
        price_features_input_name=args.price_features_input_name,
        scored_input_name=args.scored_input_name,
        output_suffix=args.output_suffix,
        tickers=args.tickers,
        benchmark_ticker=args.benchmark_ticker,
        model_cache_dir=args.model_cache_dir,
    )

    summary = outputs["summary"]
    article_summary = outputs["article_summary"]
    metadata = outputs["metadata"]

    print(summary.to_string(index=False))
    print()
    print(article_summary.to_string(index=False))
    print()
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
