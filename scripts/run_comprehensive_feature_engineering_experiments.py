from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from run_modelling_baselines import ALL_FINBERT_FEATURES, PRICE_ONLY_FEATURES, build_paths, load_dataset

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


TARGETS = {
    "next_day_up": ("target_next_day_up", None),
    "next_day_excess_gt_0": ("next_day_excess_return", 0.0),
    "next_day_excess_gt_0_5pct": ("next_day_excess_return", 0.005),
}


BASE_PRICE_FEATURES = PRICE_ONLY_FEATURES
BASE_NEWS_FEATURES = ALL_FINBERT_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    g = out.groupby("ticker", group_keys=False)

    out["same_day_excess_return"] = out["return_1d"] - out["spy_return_1d"]
    out["same_day_excess_return_lag1"] = g["same_day_excess_return"].shift(1)
    out["same_day_excess_return_rolling3"] = g["same_day_excess_return"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["same_day_excess_return_rolling5"] = g["same_day_excess_return"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["abs_return_1d"] = out["return_1d"].abs()
    out["abs_excess_return_1d"] = out["same_day_excess_return"].abs()

    out["intraday_return"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["daily_range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["close_to_high"] = (out["high"] - out["close"]) / out["close"].replace(0, np.nan)
    out["close_to_low"] = (out["close"] - out["low"]) / out["close"].replace(0, np.nan)

    out["volume_z_20"] = (
        (out["volume"] - g["volume"].shift(1).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True))
        / g["volume"].shift(1).rolling(20, min_periods=5).std().replace(0, np.nan).reset_index(level=0, drop=True)
    )
    out["volatility_ratio_5d_to_spy"] = out["volatility_5d"] / out["spy_volatility_5d"].replace(0, np.nan)
    out["momentum_3_minus_5"] = out["return_3d"] - out["return_5d"]
    out["price_above_ma5"] = (out["adj_close"] - out["moving_avg_5d"]) / out["moving_avg_5d"].replace(0, np.nan)
    out["price_above_ma20"] = (out["adj_close"] - out["moving_avg_20d"]) / out["moving_avg_20d"].replace(0, np.nan)

    if "finbert_sentiment_score_mean" in out.columns:
        out["sentiment_abs_mean"] = out["finbert_sentiment_score_mean"].abs()
        out["positive_minus_neutral"] = out["finbert_positive_mean"] - out["finbert_neutral_mean"]
        out["negative_minus_neutral"] = out["finbert_negative_mean"] - out["finbert_neutral_mean"]
        out["sentiment_intensity"] = out["finbert_positive_mean"] + out["finbert_negative_mean"]
        out["sentiment_weighted_by_log_news"] = out["finbert_sentiment_score_mean"] * out["log_news_count"]
        out["negative_weighted_by_log_news"] = out["finbert_negative_mean"] * out["log_news_count"]
        out["positive_weighted_by_log_news"] = out["finbert_positive_mean"] * out["log_news_count"]
        out["news_count_vs_ticker_20d"] = (
            out["news_count"] - g["news_count"].shift(1).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
        )
        out["sentiment_vs_ticker_20d"] = (
            out["finbert_sentiment_score_mean"]
            - g["finbert_sentiment_score_mean"].shift(1).rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
        )

    if "market_finbert_sentiment_score_mean" in out.columns:
        out["stock_minus_market_sentiment"] = out["finbert_sentiment_score_mean"] - out["market_finbert_sentiment_score_mean"]
        out["stock_plus_market_sentiment"] = out["finbert_sentiment_score_mean"] + out["market_finbert_sentiment_score_mean"]
        out["market_sentiment_abs_mean"] = out["market_finbert_sentiment_score_mean"].abs()
        out["market_sentiment_weighted_by_log_news"] = out["market_finbert_sentiment_score_mean"] * out["market_log_news_count"]
        out["stock_sentiment_x_market_sentiment"] = out["finbert_sentiment_score_mean"] * out["market_finbert_sentiment_score_mean"]
        out["market_negative_weighted_by_log_news"] = out["market_finbert_negative_mean"] * out["market_log_news_count"]

    return out.replace([np.inf, -np.inf], np.nan)


ENGINEERED_PRICE_FEATURES = [
    "same_day_excess_return",
    "same_day_excess_return_lag1",
    "same_day_excess_return_rolling3",
    "same_day_excess_return_rolling5",
    "abs_return_1d",
    "abs_excess_return_1d",
    "intraday_return",
    "daily_range_pct",
    "close_to_high",
    "close_to_low",
    "volume_z_20",
    "volatility_ratio_5d_to_spy",
    "momentum_3_minus_5",
    "price_above_ma5",
    "price_above_ma20",
]

ENGINEERED_NEWS_FEATURES = [
    "sentiment_abs_mean",
    "positive_minus_neutral",
    "negative_minus_neutral",
    "sentiment_intensity",
    "sentiment_weighted_by_log_news",
    "negative_weighted_by_log_news",
    "positive_weighted_by_log_news",
    "news_count_vs_ticker_20d",
    "sentiment_vs_ticker_20d",
    "stock_minus_market_sentiment",
    "stock_plus_market_sentiment",
    "market_sentiment_abs_mean",
    "market_sentiment_weighted_by_log_news",
    "stock_sentiment_x_market_sentiment",
    "market_negative_weighted_by_log_news",
]

FEATURE_SETS = {
    "price_only": BASE_PRICE_FEATURES,
    "price_engineered": BASE_PRICE_FEATURES + ENGINEERED_PRICE_FEATURES,
    "news_only": BASE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES,
    "price_news": BASE_PRICE_FEATURES + BASE_NEWS_FEATURES,
    "price_news_engineered": BASE_PRICE_FEATURES + ENGINEERED_PRICE_FEATURES + BASE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES,
}


def build_model(model_name: str) -> Pipeline:
    if model_name == "logistic_l2":
        model = LogisticRegression(max_iter=3000, random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "logistic_balanced":
        model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1)
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1)
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.03, l2_regularization=0.1, random_state=42)
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=2, min_samples_leaf=8, random_state=42)
    elif model_name == "adaboost_stumps":
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=8, random_state=42)
        model = AdaBoostClassifier(estimator=stump, n_estimators=250, learning_rate=0.03, random_state=42)
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        model = XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    elif model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostClassifier(
            iterations=250,
            depth=2,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
            thread_count=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


MODEL_NAMES = [
    "logistic_l2",
    "logistic_balanced",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "gradient_boosting",
    "adaboost_stumps",
    "xgboost",
    "lightgbm",
    "catboost",
]


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    out["next_day_stock_return"] = out.groupby("ticker")["return_1d"].shift(-1)
    out["next_day_spy_return"] = out.groupby("ticker")["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    return out.dropna(subset=["next_day_stock_return", "next_day_spy_return"]).reset_index(drop=True)


def assign_target(df: pd.DataFrame, target_name: str) -> pd.Series:
    source_col, threshold = TARGETS[target_name]
    if threshold is None:
        return df[source_col].astype(int)
    return (df[source_col] > threshold).astype(int)


def build_walk_forward_splits(df: pd.DataFrame, start_fraction: float, test_fraction: float, step_fraction: float) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    dates = sorted(df["trading_date"].drop_duplicates())
    n_dates = len(dates)
    test_size = max(5, int(n_dates * test_fraction))
    step_size = max(5, int(n_dates * step_fraction))
    train_end_idx = max(20, int(n_dates * start_fraction))
    splits = []
    fold = 1
    while train_end_idx < n_dates - test_size:
        train_end_date = dates[train_end_idx]
        test_end_idx = min(train_end_idx + test_size, n_dates)
        test_start_date = dates[train_end_idx]
        test_end_date = dates[test_end_idx - 1]
        train = df[df["trading_date"] < train_end_date].copy()
        test = df[(df["trading_date"] >= test_start_date) & (df["trading_date"] <= test_end_date)].copy()
        if not train.empty and not test.empty:
            splits.append((train, test, fold))
        train_end_idx += step_size
        fold += 1
    if not splits:
        raise ValueError("No valid walk-forward splits.")
    return splits


def metrics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
    }


def run_candidate(train: pd.DataFrame, test: pd.DataFrame, target_name: str, feature_set_name: str, model_name: str) -> dict[str, Any] | None:
    y_train = assign_target(train, target_name)
    y_test = assign_target(test, target_name)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    feature_cols = [col for col in FEATURE_SETS[feature_set_name] if col in train.columns]
    model = build_model(model_name)
    model.fit(train[feature_cols], y_train)
    pred = model.predict(test[feature_cols])
    proba = model.predict_proba(test[feature_cols])[:, 1]
    return {
        "target_name": target_name,
        "feature_set": feature_set_name,
        "model_name": model_name,
        "feature_count": len(feature_cols),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train["trading_date"].min().date().isoformat(),
        "train_end": train["trading_date"].max().date().isoformat(),
        "test_start": test["trading_date"].min().date().isoformat(),
        "test_end": test["trading_date"].max().date().isoformat(),
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        **metrics(y_test, pred, proba),
    }


def summarize(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "majority_baseline_accuracy", "test_positive_rate"]
    out = results.groupby(group_cols, as_index=False)[metric_cols].mean().rename(columns={c: f"mean_{c}" for c in metric_cols})
    counts = results.groupby(group_cols, as_index=False)["fold"].count().rename(columns={"fold": "num_folds"})
    return out.merge(counts, on=group_cols, how="left").sort_values(group_cols + ["mean_roc_auc"], ascending=[True] * len(group_cols) + [False])


def build_split_audit(df: pd.DataFrame, scope: str, splits: list[tuple[pd.DataFrame, pd.DataFrame, int]]) -> pd.DataFrame:
    rows = []
    for train, test, fold in splits:
        rows.append(
            {
                "scope": scope,
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_start": train["trading_date"].min().date().isoformat(),
                "train_end": train["trading_date"].max().date().isoformat(),
                "test_start": test["trading_date"].min().date().isoformat(),
                "test_end": test["trading_date"].max().date().isoformat(),
                "overlapping_dates": int(len(set(train["trading_date"]).intersection(set(test["trading_date"])))),
                "unique_train_dates": train["trading_date"].nunique(),
                "unique_test_dates": test["trading_date"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def run_scope(df: pd.DataFrame, scope_name: str, start_fraction: float, test_fraction: float, step_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = build_walk_forward_splits(df, start_fraction, test_fraction, step_fraction)
    rows = []
    for train, test, fold in splits:
        for target_name in TARGETS:
            for feature_set_name in FEATURE_SETS:
                for model_name in MODEL_NAMES:
                    row = run_candidate(train, test, target_name, feature_set_name, model_name)
                    if row is not None:
                        row["scope"] = scope_name
                        row["fold"] = fold
                        rows.append(row)
    return pd.DataFrame(rows), build_split_audit(df, scope_name, splits)


def best_news_delta(summary: pd.DataFrame, scope_cols: list[str]) -> pd.DataFrame:
    idx_cols = scope_cols + ["target_name"]
    best = summary.sort_values(idx_cols + ["mean_roc_auc"], ascending=[True] * len(idx_cols) + [False])
    best = best.groupby(idx_cols + ["feature_set"], as_index=False).head(1)
    price = best[best["feature_set"] == "price_only"][idx_cols + ["model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy"]].rename(
        columns={"model_name": "best_price_model", "mean_roc_auc": "best_price_roc_auc", "mean_accuracy": "best_price_accuracy", "mean_balanced_accuracy": "best_price_balanced_accuracy"}
    )
    news = best[best["feature_set"].isin(["price_news", "price_news_engineered"])].sort_values(idx_cols + ["mean_roc_auc"], ascending=[True] * len(idx_cols) + [False])
    news = news.groupby(idx_cols, as_index=False).head(1)
    news = news[idx_cols + ["feature_set", "model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy"]].rename(
        columns={"feature_set": "best_news_feature_set", "model_name": "best_news_model", "mean_roc_auc": "best_news_roc_auc", "mean_accuracy": "best_news_accuracy", "mean_balanced_accuracy": "best_news_balanced_accuracy"}
    )
    out = price.merge(news, on=idx_cols, how="inner")
    out["roc_auc_delta_news_minus_price"] = out["best_news_roc_auc"] - out["best_price_roc_auc"]
    out["accuracy_delta_news_minus_price"] = out["best_news_accuracy"] - out["best_price_accuracy"]
    out["balanced_accuracy_delta_news_minus_price"] = out["best_news_balanced_accuracy"] - out["best_price_balanced_accuracy"]
    return out.sort_values(idx_cols + ["roc_auc_delta_news_minus_price"], ascending=[True] * len(idx_cols) + [False])


def run_pipeline(
    project_root: str | None,
    dataset_name: str,
    output_suffix: str,
    start_fraction: float,
    test_fraction: float,
    step_fraction: float,
    scope_mode: str = "all",
    model_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    df = add_engineered_features(prepare_targets(load_dataset(paths, dataset_name)))
    selected_model_names = model_names or MODEL_NAMES

    original_model_names = MODEL_NAMES[:]
    MODEL_NAMES[:] = selected_model_names
    try:
        pooled_results, pooled_audit = run_scope(df, "pooled_all_tickers", start_fraction, test_fraction, step_fraction)
        single_results = []
        single_audits = []
        if scope_mode in {"all", "single"}:
            for ticker, ticker_df in df.groupby("ticker"):
                result, audit = run_scope(ticker_df.copy(), ticker, start_fraction, test_fraction, step_fraction)
                single_results.append(result)
                single_audits.append(audit)
    finally:
        MODEL_NAMES[:] = original_model_names

    all_results = pd.concat([pooled_results, *single_results], ignore_index=True)
    split_audit = pd.concat([pooled_audit, *single_audits], ignore_index=True)
    pooled_summary = summarize(pooled_results, ["scope", "target_name", "feature_set", "model_name"])
    single_summary = (
        summarize(pd.concat(single_results, ignore_index=True), ["scope", "target_name", "feature_set", "model_name"])
        if single_results
        else pd.DataFrame()
    )
    pooled_deltas = best_news_delta(pooled_summary, ["scope"])
    single_deltas = best_news_delta(single_summary, ["scope"]) if not single_summary.empty else pd.DataFrame()
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "rows": len(df),
                "tickers": ", ".join(sorted(df["ticker"].unique())),
                "start_fraction": start_fraction,
                "test_fraction": test_fraction,
                "step_fraction": step_fraction,
                "models": ", ".join(selected_model_names),
                "feature_sets": ", ".join(FEATURE_SETS.keys()),
                "scope_mode": scope_mode,
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    all_results.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_results{suffix}.csv", index=False)
    pooled_summary.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_pooled_summary{suffix}.csv", index=False)
    single_summary.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_single_ticker_summary{suffix}.csv", index=False)
    pooled_deltas.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_pooled_deltas{suffix}.csv", index=False)
    single_deltas.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_single_ticker_deltas{suffix}.csv", index=False)
    split_audit.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_split_audit{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"comprehensive_feature_engineering_metadata{suffix}.csv", index=False)
    return {
        "all_results": all_results,
        "pooled_summary": pooled_summary,
        "single_summary": single_summary,
        "pooled_deltas": pooled_deltas,
        "single_deltas": single_deltas,
        "split_audit": split_audit,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive feature engineering and model experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--step-fraction", type=float, default=0.15)
    parser.add_argument("--scope-mode", choices=["all", "pooled", "single"], default="all")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        args.project_root,
        args.dataset_name,
        args.output_suffix,
        args.start_fraction,
        args.test_fraction,
        args.step_fraction,
        scope_mode=args.scope_mode,
        model_names=args.models,
    )
    print(outputs["metadata"].to_string(index=False))
    print("\nPooled deltas")
    print(outputs["pooled_deltas"].to_string(index=False))
    print("\nSingle-ticker deltas")
    print(outputs["single_deltas"].to_string(index=False))
    print("\nSplit audit")
    print(outputs["split_audit"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
