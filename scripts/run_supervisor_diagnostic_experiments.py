from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import StandardScaler

from run_modelling_baselines import ALL_FINBERT_FEATURES, PRICE_ONLY_FEATURES, load_dataset


SENTIMENT_DIAGNOSTIC_FEATURES = [
    "news_count",
    "has_news",
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score_mean",
    "finbert_sentiment_score_surprise",
]


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


def prepare_base_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    out = out.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    out["next_day_stock_return"] = out.groupby("ticker")["return_1d"].shift(-1)
    out["next_day_spy_return"] = out.groupby("ticker")["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    out["target_next_day_excess_gt_0"] = (out["next_day_excess_return"] > 0).astype(int)
    out["target_same_day_excess_gt_0"] = (out["return_1d"] > out["spy_return_1d"]).astype(int)
    return out


def safe_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    if len(unique_dates) < 10:
        raise ValueError("Not enough dates for time split.")
    split_idx = min(max(1, int(len(unique_dates) * (1 - test_fraction))), len(unique_dates) - 1)
    cutoff = unique_dates[split_idx]
    train_df = df[df["trading_date"] < cutoff].copy()
    test_df = df[df["trading_date"] >= cutoff].copy()
    return train_df, test_df


def build_model(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    if model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
) -> dict[str, Any] | None:
    train_df = train_df.dropna(subset=[target_col]).copy()
    test_df = test_df.dropna(subset=[target_col]).copy()
    if len(train_df) < 20 or len(test_df) < 8:
        return None
    y_train = train_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    model = build_model(model_name)
    model.fit(train_df[feature_cols], y_train)
    pred = model.predict(test_df[feature_cols])
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    metrics = safe_metrics(y_test, pred, proba)
    return {
        "model_name": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        **metrics,
    }


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Stock prices",
                "frequency": "Daily",
                "variables": "OHLCV, returns, volatility, moving averages, SPY controls",
                "rows": len(df),
                "coverage": f"{df['trading_date'].min().date()} to {df['trading_date'].max().date()}",
                "challenge": "Short time series and market-wide co-movement",
            },
            {
                "dataset": "Financial news",
                "frequency": "Timestamped, aggregated daily",
                "variables": "Article-ticker mentions, publication time, news count",
                "rows": int(df["has_news"].sum()),
                "coverage": f"{df['ticker'].nunique()} tickers",
                "challenge": "Sparse observations and multiple articles per ticker-day",
            },
            {
                "dataset": "FinBERT sentiment",
                "frequency": "Daily ticker-level aggregation",
                "variables": "Positive, negative, neutral, sentiment score, lags, rolling means",
                "rows": int((df["has_news"] == 1).sum()),
                "coverage": f"{df['news_count'].sum():.0f} aligned article-ticker contributions",
                "challenge": "Noisy labels and possible timing mismatch with price reaction",
            },
        ]
    )


def run_news_time_shift_experiment(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = df.sort_values(["ticker", "trading_date"]).copy()
    feature_cols = [f"shifted_{col}" for col in SENTIMENT_DIAGNOSTIC_FEATURES]

    for shift_days in range(-3, 4):
        diagnostic_df = base.copy()
        grouped = diagnostic_df.groupby("ticker", group_keys=False)
        for col in SENTIMENT_DIAGNOSTIC_FEATURES:
            diagnostic_df[f"shifted_{col}"] = grouped[col].shift(-shift_days)
        diagnostic_df = diagnostic_df[diagnostic_df["shifted_has_news"].fillna(0) == 1].copy()

        if diagnostic_df.empty:
            continue
        train_df, test_df = time_split(diagnostic_df)
        for model_name in ["logistic_regression", "random_forest"]:
            result = evaluate_candidate(
                train_df,
                test_df,
                feature_cols,
                "target_same_day_excess_gt_0",
                model_name,
            )
            if result is None:
                continue
            result.update(
                {
                    "experiment": "news_time_shift",
                    "news_day_offset": shift_days,
                    "interpretation": (
                        "trading-day offset; negative = older news explains current return; "
                        "zero = same-day association; positive = future news diagnostic only"
                    ),
                }
            )
            rows.append(result)
    return pd.DataFrame(rows).sort_values(["roc_auc", "balanced_accuracy"], ascending=False)


def run_per_stock_experiment(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = {
        "next_day_up": "target_next_day_up",
        "next_day_excess_gt_0": "target_next_day_excess_gt_0",
    }
    feature_sets = {
        "price_only": PRICE_ONLY_FEATURES,
                    "price_plus_finbert": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
    }
    slices = {
        "all_days": lambda x: x.copy(),
        "news_days": lambda x: x[x["has_news"] == 1].copy(),
    }

    for ticker, ticker_df in df.groupby("ticker"):
        for slice_name, slice_fn in slices.items():
            slice_df = slice_fn(ticker_df).dropna(subset=["next_day_stock_return"]).copy()
            if len(slice_df) < 30:
                continue
            train_df, test_df = time_split(slice_df)
            for target_name, target_col in targets.items():
                for feature_set_name, feature_cols in feature_sets.items():
                    result = evaluate_candidate(
                        train_df,
                        test_df,
                        feature_cols,
                        target_col,
                        "random_forest",
                    )
                    if result is None:
                        continue
                    result.update(
                        {
                            "experiment": "per_stock",
                            "ticker": ticker,
                            "slice_name": slice_name,
                            "target_name": target_name,
                            "feature_set": feature_set_name,
                        }
                    )
                    rows.append(result)
    return pd.DataFrame(rows).sort_values(["target_name", "slice_name", "ticker", "roc_auc"])


def run_correlation_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = [
        "news_count",
        "finbert_positive_mean",
        "finbert_negative_mean",
        "finbert_neutral_mean",
        "finbert_sentiment_score_mean",
        "finbert_sentiment_score_surprise",
    ]
    for ticker, ticker_df in df.groupby("ticker"):
        news_df = ticker_df[ticker_df["has_news"] == 1].copy()
        for feature in cols:
            valid = news_df[[feature, "next_day_excess_return", "return_1d"]].dropna()
            if len(valid) < 10 or valid[feature].nunique() < 2:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "feature": feature,
                    "rows": len(valid),
                    "corr_with_same_day_return": valid[feature].corr(valid["return_1d"]),
                    "corr_with_next_day_excess_return": valid[feature].corr(valid["next_day_excess_return"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "corr_with_next_day_excess_return",
        key=lambda s: s.abs(),
        ascending=False,
    )


def summarize_existing_target_comparison(paths: ProjectPaths, output_suffix: str = "") -> pd.DataFrame:
    suffix = f"_{output_suffix}" if output_suffix else ""
    summary_path = paths.tables_dir / f"general_target_redesign_walkforward_summary{suffix}.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    best = (
        df.sort_values(["target_name", "mean_roc_auc"], ascending=[True, False])
        .groupby("target_name", as_index=False)
        .head(1)
    )
    return best[
        [
            "target_name",
            "feature_set",
            "model_name",
            "mean_accuracy",
            "mean_balanced_accuracy",
            "mean_f1",
            "mean_roc_auc",
            "mean_test_positive_rate",
            "num_folds",
        ]
    ].sort_values("mean_roc_auc", ascending=False)


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    output_suffix: str = "",
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    raw_df = load_dataset(paths, dataset_name)
    df = prepare_base_dataset(raw_df)

    outputs = {
        "supervisor_dataset_summary": build_dataset_summary(df),
        "supervisor_news_shift_results": run_news_time_shift_experiment(df),
        "supervisor_per_stock_results": run_per_stock_experiment(df),
        "supervisor_correlation_diagnostics": run_correlation_diagnostics(df),
        "supervisor_target_comparison": summarize_existing_target_comparison(paths, output_suffix),
        "supervisor_diagnostic_metadata": pd.DataFrame(
            [
                {
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_name": dataset_name,
                    "rows": len(df),
                    "tickers": ", ".join(sorted(df["ticker"].unique())),
                    "purpose": "Supervisor-driven diagnostic experiments for thesis interpretation",
                }
            ]
        ),
    }

    suffix = f"_{output_suffix}" if output_suffix else ""
    for name, table in outputs.items():
        table.to_csv(paths.tables_dir / f"{name}{suffix}.csv", index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervisor-driven diagnostic thesis experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.output_suffix)
    for name, table in outputs.items():
        print(f"\n{name}")
        print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
