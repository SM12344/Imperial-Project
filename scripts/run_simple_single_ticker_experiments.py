from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from run_modelling_baselines import build_paths, load_dataset


TARGETS = {
    "next_day_up": ("target_next_day_up", None),
    "next_day_excess_gt_0": ("next_day_excess_return", 0.0),
    "next_day_excess_gt_0_5pct": ("next_day_excess_return", 0.005),
}

PRICE_COMPACT = [
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

NEWS_COMPACT = [
    "news_count",
    "log_news_count",
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
]

ENGINEERED_COMPACT = [
    "same_day_excess_return_lag1",
    "same_day_excess_return_rolling5",
    "sentiment_intensity",
    "sentiment_weighted_by_log_news",
    "stock_minus_market_sentiment",
    "stock_sentiment_x_market_sentiment",
]

FEATURE_SETS = {
    "price_compact": PRICE_COMPACT,
    "news_compact": NEWS_COMPACT,
    "price_news_compact": PRICE_COMPACT + NEWS_COMPACT,
    "price_news_engineered_compact": PRICE_COMPACT + NEWS_COMPACT + ENGINEERED_COMPACT,
}


def add_compact_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    g = out.groupby("ticker", group_keys=False)
    out["next_day_stock_return"] = g["return_1d"].shift(-1)
    out["next_day_spy_return"] = g["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    out["same_day_excess_return"] = out["return_1d"] - out["spy_return_1d"]
    out["same_day_excess_return_lag1"] = g["same_day_excess_return"].shift(1)
    out["same_day_excess_return_rolling5"] = (
        g["same_day_excess_return"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["sentiment_intensity"] = out["finbert_positive_mean"] + out["finbert_negative_mean"]
    out["sentiment_weighted_by_log_news"] = out["finbert_sentiment_score_mean"] * out["log_news_count"]
    out["stock_minus_market_sentiment"] = out["finbert_sentiment_score_mean"] - out["market_finbert_sentiment_score_mean"]
    out["stock_sentiment_x_market_sentiment"] = (
        out["finbert_sentiment_score_mean"] * out["market_finbert_sentiment_score_mean"]
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_day_stock_return", "next_day_spy_return"])


def assign_target(df: pd.DataFrame, target_name: str) -> pd.Series:
    source_col, threshold = TARGETS[target_name]
    if threshold is None:
        return df[source_col].astype(int)
    return (df[source_col] > threshold).astype(int)


def build_splits(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    dates = sorted(df["trading_date"].drop_duplicates())
    splits: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    train_end = max(20, int(len(dates) * 0.5))
    test_size = max(5, int(len(dates) * 0.15))
    step = max(5, int(len(dates) * 0.15))
    fold = 1
    while train_end < len(dates) - test_size:
        train_dates = dates[:train_end]
        test_dates = dates[train_end : train_end + test_size]
        train = df[df["trading_date"].isin(train_dates)].copy()
        test = df[df["trading_date"].isin(test_dates)].copy()
        splits.append((train, test, fold))
        train_end += step
        fold += 1
    return splits


def build_model(model_name: str) -> Pipeline:
    if model_name == "logistic_l2":
        model = LogisticRegression(max_iter=2000, random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "logistic_balanced":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "adaboost_stumps":
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, random_state=42)
        model = AdaBoostClassifier(estimator=stump, n_estimators=80, learning_rate=0.05, random_state=42)
    elif model_name == "shallow_gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=1, min_samples_leaf=10, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


MODEL_NAMES = ["logistic_l2", "logistic_balanced", "adaboost_stumps", "shallow_gradient_boosting"]


def metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def run_pipeline(project_root: str | None, dataset_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    df = add_compact_features(load_dataset(paths, dataset_name))
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for ticker, ticker_df in df.groupby("ticker"):
        ticker_df = ticker_df.sort_values("trading_date").reset_index(drop=True)
        for train, test, fold in build_splits(ticker_df):
            split_rows.append(
                {
                    "ticker": ticker,
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_start": train["trading_date"].min().date().isoformat(),
                    "train_end": train["trading_date"].max().date().isoformat(),
                    "test_start": test["trading_date"].min().date().isoformat(),
                    "test_end": test["trading_date"].max().date().isoformat(),
                    "overlapping_dates": int(len(set(train["trading_date"]).intersection(set(test["trading_date"])))),
                }
            )
            for target_name in TARGETS:
                y_train = assign_target(train, target_name)
                y_test = assign_target(test, target_name)
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                for feature_set_name, feature_cols in FEATURE_SETS.items():
                    feature_cols = [col for col in feature_cols if col in train.columns]
                    for model_name in MODEL_NAMES:
                        model = build_model(model_name)
                        model.fit(train[feature_cols], y_train)
                        pred = model.predict(test[feature_cols])
                        proba = model.predict_proba(test[feature_cols])[:, 1]
                        rows.append(
                            {
                                "ticker": ticker,
                                "fold": fold,
                                "target_name": target_name,
                                "feature_set": feature_set_name,
                                "model_name": model_name,
                                "feature_count": len(feature_cols),
                                "train_rows": len(train),
                                "test_rows": len(test),
                                "test_positive_rate": float(y_test.mean()),
                                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                                **metric_row(y_test, pred, proba),
                            }
                        )

    results = pd.DataFrame(rows)
    split_audit = pd.DataFrame(split_rows)
    summary = (
        results.groupby(["ticker", "target_name", "feature_set", "model_name"], as_index=False)[
            ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "majority_baseline_accuracy", "test_positive_rate"]
        ]
        .mean()
        .rename(columns=lambda c: f"mean_{c}" if c not in {"ticker", "target_name", "feature_set", "model_name"} else c)
    )
    fold_counts = (
        results.groupby(["ticker", "target_name", "feature_set", "model_name"], as_index=False)["fold"]
        .count()
        .rename(columns={"fold": "num_folds"})
    )
    summary = summary.merge(fold_counts, on=["ticker", "target_name", "feature_set", "model_name"], how="left")
    best = summary.sort_values(["ticker", "target_name", "mean_roc_auc"], ascending=[True, True, False]).groupby(
        ["ticker", "target_name"], as_index=False
    ).head(5)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "rows": len(df),
                "tickers": ", ".join(sorted(df["ticker"].unique())),
                "models": ", ".join(MODEL_NAMES),
                "feature_sets": ", ".join(FEATURE_SETS.keys()),
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"simple_single_ticker_results{suffix}.csv", index=False)
    summary.to_csv(paths.tables_dir / f"simple_single_ticker_summary{suffix}.csv", index=False)
    best.to_csv(paths.tables_dir / f"simple_single_ticker_best{suffix}.csv", index=False)
    split_audit.to_csv(paths.tables_dir / f"simple_single_ticker_split_audit{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"simple_single_ticker_metadata{suffix}.csv", index=False)
    return {"results": results, "summary": summary, "best": best, "split_audit": split_audit, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-complexity single-ticker experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print()
    print(outputs["split_audit"].drop_duplicates(["ticker", "fold"]).to_string(index=False))
    print()
    print(outputs["best"].to_string(index=False))


if __name__ == "__main__":
    main()
