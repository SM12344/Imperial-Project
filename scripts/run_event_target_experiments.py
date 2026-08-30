from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_modelling_baselines import ALL_FINBERT_FEATURES, PRICE_ONLY_FEATURES, build_paths, load_dataset

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None


EVENT_PATTERNS = {
    "earnings": r"\b(earnings|eps|revenue|quarterly|q[1-4]|profit|guidance|forecast)\b",
    "analyst": r"\b(upgrade|downgrade|analyst|price target|rating|initiates|maintains)\b",
    "product_ai_chips": r"\b(product|launch|iphone|ipad|cloud|azure|aws|ai|artificial intelligence|chip|chips|semiconductor|gpu|ev|vehicle|model y|model 3)\b",
    "legal_regulation": r"\b(lawsuit|legal|regulation|regulatory|antitrust|sec|ftc|doj|investigation|ban|fine)\b",
    "macro": r"\b(federal reserve|fed|inflation|cpi|pce|jobs report|payrolls|unemployment|gdp|treasury|rates|recession)\b",
    "deals": r"\b(acquisition|merger|deal|buyout|stake|partnership|investment)\b",
    "layoffs_costs": r"\b(layoff|layoffs|job cuts|cost cuts|restructuring|hiring freeze)\b",
}


CORE_PRICE_FEATURES = [
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

CORE_NEWS_FEATURES = [
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
    "market_finbert_positive_mean",
    "market_finbert_negative_mean",
    "market_finbert_neutral_mean",
    "market_finbert_sentiment_score_mean",
    "market_finbert_sentiment_score_lag1",
    "market_finbert_sentiment_score_rolling5",
]

ENGINEERED_NEWS_FEATURES = [
    "sentiment_intensity",
    "sentiment_weighted_news_count",
    "negative_weighted_news_count",
    "positive_weighted_news_count",
    "stock_minus_market_sentiment",
    "stock_sentiment_x_market_sentiment",
]


def category_feature_names() -> list[str]:
    names: list[str] = []
    for event_name in EVENT_PATTERNS:
        names.extend([f"{event_name}_count", f"{event_name}_sentiment_mean"])
    return names


FEATURE_SETS = {
    "price_only": CORE_PRICE_FEATURES,
    "price_news_core": CORE_PRICE_FEATURES + CORE_NEWS_FEATURES,
    "price_news_events": CORE_PRICE_FEATURES + CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names(),
    "news_events_only": CORE_NEWS_FEATURES + ENGINEERED_NEWS_FEATURES + category_feature_names(),
}


def load_scored_news(paths, scored_news_name: str) -> pd.DataFrame:
    path = paths.processed_dir / scored_news_name
    if not path.exists():
        raise FileNotFoundError(f"Scored news file not found: {path}")
    news = pd.read_csv(path)
    news = news.dropna(subset=["aligned_trading_date"]).copy()
    news["aligned_trading_date"] = pd.to_datetime(news["aligned_trading_date"])
    news["text_for_events"] = (
        news["title"].fillna("").astype(str) + " " + news["description"].fillna("").astype(str)
    ).str.lower()
    return news


def build_event_daily_features(news: pd.DataFrame) -> pd.DataFrame:
    ticker_news = news[news["ticker"] != "__MARKET__"].copy()
    rows = []
    for event_name, pattern in EVENT_PATTERNS.items():
        mask = ticker_news["text_for_events"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        event = ticker_news[mask].copy()
        if event.empty:
            continue
        daily = (
            event.groupby(["ticker", "aligned_trading_date"], as_index=False)
            .agg(
                **{
                    f"{event_name}_count": ("article_id", "count"),
                    f"{event_name}_sentiment_mean": ("finbert_sentiment_score", "mean"),
                }
            )
            .rename(columns={"aligned_trading_date": "trading_date"})
        )
        rows.append(daily)

    if not rows:
        return pd.DataFrame(columns=["ticker", "trading_date", *category_feature_names()])

    merged = rows[0]
    for frame in rows[1:]:
        merged = merged.merge(frame, on=["ticker", "trading_date"], how="outer")
    return merged


def add_targets_and_features(base: pd.DataFrame, event_daily: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    if not event_daily.empty:
        event_daily = event_daily.copy()
        event_daily["trading_date"] = pd.to_datetime(event_daily["trading_date"])
        df = df.merge(event_daily, on=["ticker", "trading_date"], how="left")

    for col in category_feature_names():
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["ticker", "trading_date"]).copy()
    g = df.groupby("ticker", group_keys=False)
    for horizon in [1, 3, 5]:
        future_stock = g["adj_close"].shift(-horizon) / df["adj_close"] - 1
        future_spy = (1 + g["spy_return_1d"].shift(-1)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
        # The rolling expression above is backward-looking after shift; recompute by explicit grouped forward sums for stability.
        future_spy = g["spy_return_1d"].transform(lambda s: (1 + s.shift(-1)).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1)
        df[f"fwd_{horizon}d_return"] = future_stock
        df[f"fwd_{horizon}d_spy_return"] = future_spy
        df[f"fwd_{horizon}d_excess_return"] = future_stock - future_spy
        df[f"target_{horizon}d_excess_gt_0"] = (df[f"fwd_{horizon}d_excess_return"] > 0).astype(int)
        df[f"target_{horizon}d_excess_gt_0_5pct"] = (df[f"fwd_{horizon}d_excess_return"] > 0.005).astype(int)
        df[f"target_{horizon}d_abs_excess_gt_1pct"] = (df[f"fwd_{horizon}d_excess_return"].abs() > 0.01).astype(int)
        df[f"target_{horizon}d_abs_return_gt_1pct"] = (df[f"fwd_{horizon}d_return"].abs() > 0.01).astype(int)

    df["sentiment_intensity"] = df["finbert_positive_mean"] + df["finbert_negative_mean"]
    df["sentiment_weighted_news_count"] = df["finbert_sentiment_score_mean"] * df["log_news_count"]
    df["negative_weighted_news_count"] = df["finbert_negative_mean"] * df["log_news_count"]
    df["positive_weighted_news_count"] = df["finbert_positive_mean"] * df["log_news_count"]
    df["stock_minus_market_sentiment"] = df["finbert_sentiment_score_mean"] - df["market_finbert_sentiment_score_mean"]
    df["stock_sentiment_x_market_sentiment"] = df["finbert_sentiment_score_mean"] * df["market_finbert_sentiment_score_mean"]
    df["event_count_total"] = df[[f"{name}_count" for name in EVENT_PATTERNS]].sum(axis=1)
    df["event_signal_day"] = (
        (df["news_count"] >= 20)
        | (df["sentiment_intensity"] >= 0.65)
        | (df["market_news_count"] >= 3)
        | (df["event_count_total"] > 0)
    ).astype(int)
    df["earnings_day"] = (df["earnings_count"] > 0).astype(int)
    df["macro_or_market_day"] = ((df["macro_count"] > 0) | (df["market_news_count"] >= 3)).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


CLASSIFICATION_TARGETS = [
    "target_1d_excess_gt_0",
    "target_1d_excess_gt_0_5pct",
    "target_1d_abs_excess_gt_1pct",
    "target_3d_excess_gt_0",
    "target_3d_excess_gt_0_5pct",
    "target_3d_abs_excess_gt_1pct",
    "target_5d_excess_gt_0",
    "target_5d_excess_gt_0_5pct",
    "target_5d_abs_excess_gt_1pct",
]

REGRESSION_TARGETS = ["fwd_1d_excess_return", "fwd_3d_excess_return", "fwd_5d_excess_return"]


def build_classifier(model_name: str) -> Pipeline:
    if model_name == "logistic_balanced":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ]
        )
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            min_child_samples=25,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    elif model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostClassifier(
            iterations=180,
            depth=2,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
            thread_count=1,
        )
    else:
        raise ValueError(f"Unsupported classifier: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


def build_regressor(model_name: str) -> Pipeline:
    if model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMRegressor(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            min_child_samples=25,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    elif model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostRegressor(
            iterations=180,
            depth=2,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
            thread_count=1,
        )
    else:
        raise ValueError(f"Unsupported regressor: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


CLASSIFIERS = ["logistic_balanced", "lightgbm", "catboost"]
REGRESSORS = ["ridge", "lightgbm", "catboost"]


def build_splits(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    dates = sorted(df["trading_date"].drop_duplicates())
    train_end = max(20, int(len(dates) * 0.5))
    test_size = max(5, int(len(dates) * 0.15))
    step = max(5, int(len(dates) * 0.15))
    splits = []
    fold = 1
    while train_end < len(dates) - test_size:
        train_dates = dates[:train_end]
        test_dates = dates[train_end : train_end + test_size]
        train = df[df["trading_date"].isin(train_dates)].copy()
        test = df[df["trading_date"].isin(test_dates)].copy()
        if not train.empty and not test.empty:
            splits.append((train, test, fold))
        train_end += step
        fold += 1
    return splits


def classification_metrics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
    }


def run_classification(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for train, test, fold in build_splits(df):
        for filter_name, filter_col in {
            "all_days": None,
            "event_signal_days": "event_signal_day",
            "earnings_days": "earnings_day",
            "macro_or_market_days": "macro_or_market_day",
        }.items():
            train_f = train if filter_col is None else train[train[filter_col] == 1].copy()
            test_f = test if filter_col is None else test[test[filter_col] == 1].copy()
            if len(train_f) < 50 or len(test_f) < 15:
                continue
            for target in CLASSIFICATION_TARGETS:
                train_t = train_f.dropna(subset=[target]).copy()
                test_t = test_f.dropna(subset=[target]).copy()
                y_train = train_t[target].astype(int)
                y_test = test_t[target].astype(int)
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                for feature_set, cols in FEATURE_SETS.items():
                    feature_cols = [col for col in cols if col in train_t.columns]
                    for model_name in CLASSIFIERS:
                        model = build_classifier(model_name)
                        model.fit(train_t[feature_cols], y_train)
                        pred = model.predict(test_t[feature_cols])
                        proba = model.predict_proba(test_t[feature_cols])[:, 1]
                        rows.append(
                            {
                                "scope": scope,
                                "fold": fold,
                                "filter": filter_name,
                                "target": target,
                                "feature_set": feature_set,
                                "model_name": model_name,
                                "feature_count": len(feature_cols),
                                "train_rows": len(train_t),
                                "test_rows": len(test_t),
                                "test_positive_rate": float(y_test.mean()),
                                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                                **classification_metrics(y_test, pred, proba),
                            }
                        )
    return pd.DataFrame(rows)


def run_regression(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for train, test, fold in build_splits(df):
        for target in REGRESSION_TARGETS:
            train_t = train.dropna(subset=[target]).copy()
            test_t = test.dropna(subset=[target]).copy()
            if len(train_t) < 50 or len(test_t) < 15:
                continue
            baseline_pred = np.repeat(train_t[target].mean(), len(test_t))
            baseline_mae = mean_absolute_error(test_t[target], baseline_pred)
            for feature_set, cols in FEATURE_SETS.items():
                feature_cols = [col for col in cols if col in train_t.columns]
                for model_name in REGRESSORS:
                    model = build_regressor(model_name)
                    model.fit(train_t[feature_cols], train_t[target])
                    pred = model.predict(test_t[feature_cols])
                    rows.append(
                        {
                            "scope": scope,
                            "fold": fold,
                            "target": target,
                            "feature_set": feature_set,
                            "model_name": model_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train_t),
                            "test_rows": len(test_t),
                            "mae": mean_absolute_error(test_t[target], pred),
                            "baseline_mae": baseline_mae,
                            "mae_improvement_vs_mean_baseline": baseline_mae - mean_absolute_error(test_t[target], pred),
                            "r2": r2_score(test_t[target], pred),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_classification(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "majority_baseline_accuracy", "test_positive_rate"]
    summary = (
        results.groupby(["scope", "filter", "target", "feature_set", "model_name"], as_index=False)[metrics]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metrics})
    )
    counts = results.groupby(["scope", "filter", "target", "feature_set", "model_name"], as_index=False)["fold"].count().rename(columns={"fold": "num_folds"})
    return summary.merge(counts, on=["scope", "filter", "target", "feature_set", "model_name"], how="left")


def summarize_regression(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "baseline_mae", "mae_improvement_vs_mean_baseline", "r2"]
    summary = (
        results.groupby(["scope", "target", "feature_set", "model_name"], as_index=False)[metrics]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metrics})
    )
    counts = results.groupby(["scope", "target", "feature_set", "model_name"], as_index=False)["fold"].count().rename(columns={"fold": "num_folds"})
    return summary.merge(counts, on=["scope", "target", "feature_set", "model_name"], how="left")


def best_news_vs_price(class_summary: pd.DataFrame) -> pd.DataFrame:
    idx = ["scope", "filter", "target"]
    best = class_summary.sort_values(idx + ["mean_roc_auc"], ascending=[True, True, True, False])
    best = best.groupby(idx + ["feature_set"], as_index=False).head(1)
    price = best[best["feature_set"] == "price_only"][idx + ["model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy"]].rename(
        columns={"model_name": "best_price_model", "mean_roc_auc": "best_price_roc_auc", "mean_accuracy": "best_price_accuracy", "mean_balanced_accuracy": "best_price_balanced_accuracy"}
    )
    news = best[best["feature_set"].isin(["price_news_core", "price_news_events"])].sort_values(idx + ["mean_roc_auc"], ascending=[True, True, True, False])
    news = news.groupby(idx, as_index=False).head(1)
    news = news[idx + ["feature_set", "model_name", "mean_roc_auc", "mean_accuracy", "mean_balanced_accuracy"]].rename(
        columns={"feature_set": "best_news_feature_set", "model_name": "best_news_model", "mean_roc_auc": "best_news_roc_auc", "mean_accuracy": "best_news_accuracy", "mean_balanced_accuracy": "best_news_balanced_accuracy"}
    )
    out = price.merge(news, on=idx, how="inner")
    out["roc_auc_delta_news_minus_price"] = out["best_news_roc_auc"] - out["best_price_roc_auc"]
    out["accuracy_delta_news_minus_price"] = out["best_news_accuracy"] - out["best_price_accuracy"]
    out["balanced_accuracy_delta_news_minus_price"] = out["best_news_balanced_accuracy"] - out["best_price_balanced_accuracy"]
    return out.sort_values(["scope", "filter", "roc_auc_delta_news_minus_price"], ascending=[True, True, False])


def run_pipeline(
    project_root: str | None,
    dataset_name: str,
    scored_news_name: str,
    output_suffix: str,
    scope_mode: str = "all",
    classification_targets: list[str] | None = None,
    regression_targets: list[str] | None = None,
    classifiers: list[str] | None = None,
    regressors: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    selected_class_targets = classification_targets or CLASSIFICATION_TARGETS
    selected_reg_targets = regression_targets or REGRESSION_TARGETS
    selected_classifiers = classifiers or CLASSIFIERS
    selected_regressors = regressors or REGRESSORS

    original_class_targets = CLASSIFICATION_TARGETS[:]
    original_reg_targets = REGRESSION_TARGETS[:]
    original_classifiers = CLASSIFIERS[:]
    original_regressors = REGRESSORS[:]
    CLASSIFICATION_TARGETS[:] = selected_class_targets
    REGRESSION_TARGETS[:] = selected_reg_targets
    CLASSIFIERS[:] = selected_classifiers
    REGRESSORS[:] = selected_regressors

    base = load_dataset(paths, dataset_name)
    news = load_scored_news(paths, scored_news_name)
    df = add_targets_and_features(base, build_event_daily_features(news))
    df = df.dropna(subset=["fwd_1d_excess_return", "fwd_3d_excess_return", "fwd_5d_excess_return"]).copy()

    try:
        class_frames = []
        reg_frames = []
        if scope_mode in {"all", "pooled"}:
            class_frames.append(run_classification(df, "pooled_all_tickers"))
            reg_frames.append(run_regression(df, "pooled_all_tickers"))
        if scope_mode in {"all", "single"}:
            for ticker, ticker_df in df.groupby("ticker"):
                class_frames.append(run_classification(ticker_df.copy(), ticker))
                reg_frames.append(run_regression(ticker_df.copy(), ticker))
    finally:
        CLASSIFICATION_TARGETS[:] = original_class_targets
        REGRESSION_TARGETS[:] = original_reg_targets
        CLASSIFIERS[:] = original_classifiers
        REGRESSORS[:] = original_regressors

    class_results = pd.concat([frame for frame in class_frames if not frame.empty], ignore_index=True)
    reg_results = pd.concat([frame for frame in reg_frames if not frame.empty], ignore_index=True)
    class_summary = summarize_classification(class_results)
    reg_summary = summarize_regression(reg_results)
    comparison = best_news_vs_price(class_summary)

    event_coverage = (
        df.groupby("ticker", as_index=False)
        .agg(
            rows=("ticker", "size"),
            event_signal_days=("event_signal_day", "sum"),
            earnings_days=("earnings_day", "sum"),
            macro_or_market_days=("macro_or_market_day", "sum"),
            mean_news_count=("news_count", "mean"),
            mean_market_news_count=("market_news_count", "mean"),
        )
    )
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "rows": len(df),
                "tickers": ", ".join(sorted(df["ticker"].unique())),
                "classification_targets": ", ".join(selected_class_targets),
                "regression_targets": ", ".join(selected_reg_targets),
                "classifiers": ", ".join(selected_classifiers),
                "regressors": ", ".join(selected_regressors),
                "scope_mode": scope_mode,
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    class_results.to_csv(paths.tables_dir / f"event_target_classification_results{suffix}.csv", index=False)
    class_summary.to_csv(paths.tables_dir / f"event_target_classification_summary{suffix}.csv", index=False)
    reg_results.to_csv(paths.tables_dir / f"event_target_regression_results{suffix}.csv", index=False)
    reg_summary.to_csv(paths.tables_dir / f"event_target_regression_summary{suffix}.csv", index=False)
    comparison.to_csv(paths.tables_dir / f"event_target_news_vs_price_comparison{suffix}.csv", index=False)
    event_coverage.to_csv(paths.tables_dir / f"event_target_event_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"event_target_metadata{suffix}.csv", index=False)

    return {
        "classification_results": class_results,
        "classification_summary": class_summary,
        "regression_results": reg_results,
        "regression_summary": reg_summary,
        "comparison": comparison,
        "event_coverage": event_coverage,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run event-filter and alternative-target experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--scope-mode", choices=["all", "pooled", "single"], default="all")
    parser.add_argument("--classification-targets", nargs="*", default=None)
    parser.add_argument("--regression-targets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--regressors", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        args.project_root,
        args.dataset_name,
        args.scored_news_name,
        args.output_suffix,
        scope_mode=args.scope_mode,
        classification_targets=args.classification_targets,
        regression_targets=args.regression_targets,
        classifiers=args.classifiers,
        regressors=args.regressors,
    )
    print(outputs["metadata"].to_string(index=False))
    print("\nEvent coverage")
    print(outputs["event_coverage"].to_string(index=False))
    print("\nBest news vs price comparison")
    print(outputs["comparison"].head(40).to_string(index=False))
    print("\nBest classification rows")
    print(outputs["classification_summary"].sort_values("mean_roc_auc", ascending=False).head(30).to_string(index=False))
    print("\nBest regression rows")
    print(outputs["regression_summary"].sort_values("mean_mae_improvement_vs_mean_baseline", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
