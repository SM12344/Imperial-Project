from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
DATASET_NAME = "model_dataset_finbert_quality_complete_2020_2024_polygon_proxy_market_cleaned_news.csv"
SUFFIX = "2020_2024_polygon_proxy_market_cleaned_news"


PRICE_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volume_change_1d",
    "ma_5_20_gap",
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_5d",
]

NEWS_FEATURES = [
    "news_count",
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score_mean",
    "finbert_sentiment_score_lag1",
    "finbert_sentiment_score_rolling5",
    "finbert_sentiment_score_surprise",
    "market_news_count",
    "market_finbert_sentiment_score_mean",
    "market_finbert_sentiment_score_lag1",
    "market_finbert_sentiment_score_rolling5",
    "quality_unique_story_count",
    "quality_source_weighted_sentiment_mean",
    "quality_high_conf_net_count",
    "quality_sentiment_disagreement",
]

SOCIAL_FEATURES = [
    "reddit_mentions_total",
    "reddit_mentions_log1p",
    "reddit_mentions_lag1",
    "reddit_mentions_rolling3",
    "reddit_mentions_rolling5",
    "reddit_mentions_surprise",
    "reddit_mentions_zscore_60d",
    "reddit_mentions_share_of_day",
    "reddit_has_mentions",
    "reddit_wsb_mentions",
    "reddit_stocks_mentions",
    "reddit_investing_mentions",
    "reddit_stockmarket_mentions",
    "reddit_options_mentions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def load_yolostocks(root: Path) -> pd.DataFrame:
    source_root = root / "data" / "raw" / "yolostocks-data" / "yolostocks-data-main"
    if not source_root.exists():
        raise FileNotFoundError(f"YoloStocks folder not found: {source_root}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(source_root.glob("*/*.csv")):
        year = int(csv_path.parent.name)
        if year < 2021 or year > 2024:
            continue
        subreddit = csv_path.stem.rsplit("_", 1)[0]
        wide = pd.read_csv(csv_path)
        if "ticker" not in wide.columns:
            continue
        wide = wide[wide["ticker"].isin(TICKERS)].copy()
        if wide.empty:
            continue
        date_cols = [c for c in wide.columns if c not in {"ticker", "overall_rank"}]
        long = wide.melt(id_vars=["ticker"], value_vars=date_cols, var_name="calendar_date", value_name="mentions")
        long["calendar_date"] = pd.to_datetime(long["calendar_date"], format="%m/%d/%y", errors="coerce")
        long["mentions"] = pd.to_numeric(long["mentions"], errors="coerce").fillna(0)
        long["subreddit"] = subreddit
        frames.append(long.dropna(subset=["calendar_date"]))

    if not frames:
        raise ValueError("No usable YoloStocks rows for target tickers.")
    mentions = pd.concat(frames, ignore_index=True)
    mentions = mentions.groupby(["ticker", "calendar_date", "subreddit"], as_index=False)["mentions"].sum()
    return mentions


def build_social_features(base: pd.DataFrame, mentions: pd.DataFrame) -> pd.DataFrame:
    trading_dates = pd.DataFrame({"trading_date": sorted(pd.to_datetime(base["trading_date"]).unique())})
    trading_dates["calendar_date"] = trading_dates["trading_date"]
    calendar_to_trading = pd.DataFrame({"calendar_date": pd.date_range("2021-01-01", "2024-12-31", freq="D")})
    calendar_to_trading = pd.merge_asof(
        calendar_to_trading.sort_values("calendar_date"),
        trading_dates.sort_values("calendar_date"),
        on="calendar_date",
        direction="forward",
    ).dropna(subset=["trading_date"])

    aligned = mentions.merge(calendar_to_trading, on="calendar_date", how="left").dropna(subset=["trading_date"])
    aligned["trading_date"] = pd.to_datetime(aligned["trading_date"])

    totals = aligned.groupby(["ticker", "trading_date"], as_index=False)["mentions"].sum()
    totals = totals.rename(columns={"mentions": "reddit_mentions_total"})
    pivot = aligned.pivot_table(
        index=["ticker", "trading_date"],
        columns="subreddit",
        values="mentions",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    rename = {
        "wallstreetbets": "reddit_wsb_mentions",
        "stocks": "reddit_stocks_mentions",
        "investing": "reddit_investing_mentions",
        "stockmarket": "reddit_stockmarket_mentions",
        "options": "reddit_options_mentions",
    }
    pivot = pivot.rename(columns=rename)
    keep_subs = list(rename.values())
    for col in keep_subs:
        if col not in pivot.columns:
            pivot[col] = 0

    social = totals.merge(pivot[["ticker", "trading_date", *keep_subs]], on=["ticker", "trading_date"], how="left")
    base_grid = base[["ticker", "trading_date"]].drop_duplicates().copy()
    base_grid["trading_date"] = pd.to_datetime(base_grid["trading_date"])
    social = base_grid.merge(social, on=["ticker", "trading_date"], how="left")
    social_cols = ["reddit_mentions_total", *keep_subs]
    social[social_cols] = social[social_cols].fillna(0)
    social["reddit_has_mentions"] = (social["reddit_mentions_total"] > 0).astype(int)
    social["reddit_mentions_log1p"] = np.log1p(social["reddit_mentions_total"])
    day_total = social.groupby("trading_date")["reddit_mentions_total"].transform("sum")
    social["reddit_mentions_share_of_day"] = np.where(day_total > 0, social["reddit_mentions_total"] / day_total, 0.0)

    social = social.sort_values(["ticker", "trading_date"])
    grouped = social.groupby("ticker", group_keys=False)
    social["reddit_mentions_lag1"] = grouped["reddit_mentions_total"].shift(1).fillna(0)
    social["reddit_mentions_rolling3"] = grouped["reddit_mentions_total"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    social["reddit_mentions_rolling5"] = grouped["reddit_mentions_total"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    social["reddit_mentions_surprise"] = social["reddit_mentions_total"] - social["reddit_mentions_rolling5"]
    rolling_mean = grouped["reddit_mentions_total"].transform(lambda s: s.shift(1).rolling(60, min_periods=10).mean())
    rolling_std = grouped["reddit_mentions_total"].transform(lambda s: s.shift(1).rolling(60, min_periods=10).std())
    social["reddit_mentions_zscore_60d"] = ((social["reddit_mentions_total"] - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0)
    return social


def add_forward_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "trading_date"]).copy()
    for horizon in [3, 10]:
        fwd_price = df.groupby("ticker")["adj_close"].shift(-horizon)
        stock_return = fwd_price / df["adj_close"] - 1
        spy_fwd = (
            df.sort_values(["ticker", "trading_date"])
            .groupby("ticker")["spy_return_1d"]
            .transform(lambda s: (1 + s.shift(-1)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1)
        )
        excess = stock_return - spy_fwd
        df[f"target_excess_{horizon}d_gt0"] = (excess > 0).astype(float)
        df.loc[excess.isna(), f"target_excess_{horizon}d_gt0"] = np.nan
        df[f"target_excess_{horizon}d_gt2pct"] = (excess > 0.02).astype(float)
        df.loc[excess.isna(), f"target_excess_{horizon}d_gt2pct"] = np.nan
    return df


def make_model(model_name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, numeric_features), ("cat", categorical, categorical_features)])
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=8, l2_regularization=1.0, random_state=42)
    else:
        raise ValueError(model_name)
    return Pipeline([("pre", pre), ("model", clf)])


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    majority = float(max(y_true.mean(), 1 - y_true.mean()))
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
        "majority_baseline": majority,
    }
    row["accuracy_minus_baseline"] = row["accuracy"] - row["majority_baseline"]
    return row


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.25, 0.75, 51):
        score = balanced_accuracy_score(y_true, (proba >= threshold).astype(int))
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def run_models(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "price": PRICE_FEATURES,
        "price_news": PRICE_FEATURES + NEWS_FEATURES,
        "price_social": PRICE_FEATURES + SOCIAL_FEATURES,
        "price_news_social": PRICE_FEATURES + NEWS_FEATURES + SOCIAL_FEATURES,
        "social_only": SOCIAL_FEATURES,
    }
    targets = ["target_next_day_up", "target_excess_3d_gt0", "target_excess_10d_gt2pct"]
    model_names = ["logistic_regression", "random_forest", "hist_gradient_boosting"]
    dates = pd.to_datetime(df["trading_date"])
    rows: list[dict[str, Any]] = []

    for target in targets:
        usable = df.dropna(subset=[target]).copy()
        train = usable[dates.loc[usable.index].dt.year <= 2022]
        val = usable[dates.loc[usable.index].dt.year == 2023]
        test = usable[dates.loc[usable.index].dt.year == 2024]
        if train.empty or val.empty or test.empty:
            continue
        for feature_set_name, feature_cols in feature_sets.items():
            feature_cols = [c for c in feature_cols if c in usable.columns]
            for model_name in model_names:
                pipeline = make_model(model_name, feature_cols, ["ticker"])
                y_train = train[target].astype(int)
                y_val = val[target].astype(int)
                y_test = test[target].astype(int)
                if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
                    continue
                pipeline.fit(train[feature_cols + ["ticker"]], y_train)
                val_proba = pipeline.predict_proba(val[feature_cols + ["ticker"]])[:, 1]
                threshold, val_ba = best_threshold(y_val, val_proba)
                test_proba = pipeline.predict_proba(test[feature_cols + ["ticker"]])[:, 1]
                row = {
                    "target": target,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "selected_threshold": threshold,
                    "validation_balanced_accuracy": val_ba,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                    "test_start": test["trading_date"].min().date().isoformat(),
                    "test_end": test["trading_date"].max().date().isoformat(),
                }
                row.update(metrics(y_test, test_proba, threshold))
                rows.append(row)
    return pd.DataFrame(rows)


def make_outputs(root: Path, enriched: pd.DataFrame, results: pd.DataFrame, mentions: pd.DataFrame) -> None:
    processed = root / "data" / "processed"
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    docs = root / "docs"
    for path in [processed, tables, figures, docs]:
        path.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(processed / f"model_dataset_finbert_social_attention_{SUFFIX}.csv", index=False)
    social_cols = ["ticker", "trading_date", *SOCIAL_FEATURES]
    enriched[social_cols].to_csv(processed / f"daily_reddit_attention_features_{SUFFIX}.csv", index=False)
    results.to_csv(tables / f"social_attention_model_results_{SUFFIX}.csv", index=False)

    summary = (
        results.sort_values(["target", "balanced_accuracy", "roc_auc"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .head(5)
    )
    summary.to_csv(tables / f"social_attention_best_results_{SUFFIX}.csv", index=False)

    coverage_rows = []
    for ticker, group in enriched.groupby("ticker"):
        social_period = group[pd.to_datetime(group["trading_date"]).dt.year >= 2021]
        coverage_rows.append(
            {
                "ticker": ticker,
                "rows_2021_2024": len(social_period),
                "rows_with_mentions": int(social_period["reddit_has_mentions"].sum()),
                "mention_coverage": float(social_period["reddit_has_mentions"].mean()),
                "mean_daily_mentions": float(social_period["reddit_mentions_total"].mean()),
                "max_daily_mentions": float(social_period["reddit_mentions_total"].max()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(tables / f"social_attention_coverage_{SUFFIX}.csv", index=False)

    monthly = enriched.copy()
    monthly["month"] = pd.to_datetime(monthly["trading_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = monthly[pd.to_datetime(monthly["trading_date"]).dt.year >= 2021]
    monthly_piv = monthly.groupby(["month", "ticker"])["reddit_mentions_total"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for ticker, group in monthly_piv.groupby("ticker"):
        ax.plot(group["month"], group["reddit_mentions_total"], label=ticker, linewidth=1.7)
    ax.set_title("Monthly Reddit Ticker Mentions from YoloStocks")
    ax.set_ylabel("Mention count")
    ax.set_xlabel("Month")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "social_attention_monthly_mentions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    best = summary.copy()
    def markdown_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report = [
        "# Social Attention Experiment",
        "",
        "This experiment adds public Reddit ticker-mention volume from the YoloStocks dataset as a social-media attention proxy.",
        "",
        "Important limitation: this is not post-text sentiment. The public dataset contains daily ticker mention counts by subreddit, so the added signal measures retail attention/chatter volume rather than positive or negative tone.",
        "",
        "Source: https://github.com/youyanggu/yolostocks-data",
        "",
        "## Data",
        "",
        f"- Rows in enriched modelling dataset: {len(enriched):,}",
        f"- Social data period used: 2021-01-01 to 2024-12-31",
        f"- Target tickers: {', '.join(TICKERS)}",
        "- Subreddit features retained: wallstreetbets, stocks, investing, stockmarket, options.",
        "",
        "## Best test results by target",
        "",
        markdown_table(best),
        "",
        "## Coverage by ticker",
        "",
        markdown_table(coverage),
        "",
    ]
    (docs / "social_attention_experiment_summary.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    base = pd.read_csv(root / "data" / "processed" / DATASET_NAME)
    base["trading_date"] = pd.to_datetime(base["trading_date"])
    mentions = load_yolostocks(root)
    social = build_social_features(base, mentions)
    enriched = base.merge(social, on=["ticker", "trading_date"], how="left")
    for col in SOCIAL_FEATURES:
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna(0)
    enriched = add_forward_targets(enriched)
    results = run_models(enriched)
    make_outputs(root, enriched, results, mentions)
    print(results.sort_values(["target", "balanced_accuracy", "roc_auc"], ascending=[True, False, False]).groupby("target").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
