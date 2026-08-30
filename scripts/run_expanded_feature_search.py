from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    build_ablation_feature_sets,
    build_logistic_model,
    split_holdout,
    tune_model,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
SCOPES = ["pooled_all_tickers", *TICKERS]
HORIZONS = [1, 2, 3, 5]

PRICE_ENGINEERED_FEATURES = [
    "return_2d",
    "return_10d",
    "return_20d",
    "price_slope_3d",
    "price_slope_5d",
    "price_slope_10d",
    "close_vs_5d_mean",
    "close_vs_10d_mean",
    "close_vs_20d_mean",
    "zscore_close_5d",
    "zscore_close_20d",
    "return_1d_minus_5d_mean",
    "return_3d_minus_20d_mean",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "volatility_3d",
    "volatility_10d",
    "volatility_20d",
    "volatility_ratio_5d_20d",
    "abs_return_1d",
    "abs_return_3d",
    "volume_vs_5d_mean",
    "volume_vs_20d_mean",
    "dollar_volume",
    "dollar_volume_vs_20d_mean",
    "return_minus_spy_1d",
    "return_minus_spy_2d",
    "return_minus_spy_3d",
    "return_minus_spy_5d",
    "return_minus_spy_10d",
    "spy_return_2d",
    "spy_return_3d",
    "spy_return_10d",
    "spy_volatility_3d",
    "spy_volatility_10d",
    "spy_volatility_20d",
    "beta_20d",
]

NEWS_ENGINEERED_FEATURES = [
    "news_count_rolling10",
    "news_count_surprise_vs_5d",
    "news_count_surprise_vs_10d",
    "market_news_count_rolling10",
    "market_news_count_surprise_vs_5d",
    "market_news_count_surprise_vs_10d",
    "sentiment_surprise_vs_5d",
    "market_sentiment_surprise_vs_5d",
    "news_count_x_abs_return_1d",
    "news_count_x_abs_return_3d",
    "sentiment_x_return_1d",
    "sentiment_x_return_3d",
    "sentiment_x_volume_spike",
    "negative_sentiment_x_volatility_5d",
    "positive_sentiment_x_momentum_5d",
    "market_sentiment_x_stock_return_1d",
    "market_news_x_spy_volatility_5d",
    "stock_news_minus_market_news",
]

REGRESSION_MODELS = [
    ("ridge", {"alpha": 0.1}),
    ("ridge", {"alpha": 1.0}),
    ("ridge", {"alpha": 10.0}),
    ("elastic_net", {"alpha": 0.005, "l1_ratio": 0.1}),
    ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.2}),
    ("random_forest", {"n_estimators": 160, "max_depth": 2, "min_samples_leaf": 30}),
    ("lightgbm", {"n_estimators": 120, "max_depth": 1, "num_leaves": 3, "learning_rate": 0.025, "min_child_samples": 35, "reg_lambda": 10.0, "reg_alpha": 0.1}),
    ("xgboost", {"n_estimators": 120, "max_depth": 1, "learning_rate": 0.025, "min_child_weight": 8, "reg_lambda": 10.0, "reg_alpha": 0.1}),
    ("catboost", {"iterations": 120, "depth": 1, "learning_rate": 0.025, "l2_leaf_reg": 10.0}),
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values - values.mean()
        return float(np.dot(x, y) / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def add_expanded_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    g = out.groupby("ticker", group_keys=False)

    for horizon in HORIZONS:
        out[f"fwd_{horizon}d_return"] = g["adj_close"].shift(-horizon) / out["adj_close"] - 1
        out[f"fwd_{horizon}d_spy_return"] = g["spy_return_1d"].transform(
            lambda s, h=horizon: (1 + s.shift(-1)).rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1
        )
        out[f"fwd_{horizon}d_excess_return"] = out[f"fwd_{horizon}d_return"] - out[f"fwd_{horizon}d_spy_return"]
        out[f"target_{horizon}d_excess_gt_0"] = (out[f"fwd_{horizon}d_excess_return"] > 0).astype(int)

    for horizon in [2, 10, 20]:
        out[f"return_{horizon}d"] = g["adj_close"].pct_change(horizon)

    for window in [3, 5, 10]:
        raw_slope = g["adj_close"].transform(lambda s, w=window: rolling_slope(s, w))
        rolling_mean = g["adj_close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        out[f"price_slope_{window}d"] = safe_divide(raw_slope, rolling_mean)

    for window in [5, 10, 20]:
        rolling_mean = g["adj_close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        out[f"close_vs_{window}d_mean"] = safe_divide(out["adj_close"], rolling_mean) - 1

    for window in [5, 20]:
        rolling_mean = g["adj_close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
        rolling_std = g["adj_close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).std())
        out[f"zscore_close_{window}d"] = safe_divide(out["adj_close"] - rolling_mean, rolling_std)

    out["return_1d_minus_5d_mean"] = out["return_1d"] - g["return_1d"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    out["return_3d_minus_20d_mean"] = out["return_3d"] - g["return_1d"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    rolling_high_20 = g["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    rolling_low_20 = g["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    out["distance_from_20d_high"] = safe_divide(out["adj_close"], rolling_high_20) - 1
    out["distance_from_20d_low"] = safe_divide(out["adj_close"], rolling_low_20) - 1

    for window in [3, 10, 20]:
        out[f"volatility_{window}d"] = g["return_1d"].transform(lambda s, w=window: s.rolling(w, min_periods=w).std())
    out["volatility_ratio_5d_20d"] = safe_divide(out["volatility_5d"], out["volatility_20d"])
    out["abs_return_1d"] = out["return_1d"].abs()
    out["abs_return_3d"] = out["return_3d"].abs()

    volume_5d = g["volume"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    volume_20d = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["volume_vs_5d_mean"] = safe_divide(out["volume"], volume_5d) - 1
    out["volume_vs_20d_mean"] = safe_divide(out["volume"], volume_20d) - 1
    out["dollar_volume"] = out["adj_close"] * out["volume"]
    dollar_volume_20d = g["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    out["dollar_volume_vs_20d_mean"] = safe_divide(out["dollar_volume"], dollar_volume_20d) - 1

    for horizon in [2, 3, 10]:
        out[f"spy_return_{horizon}d"] = g["spy_return_1d"].transform(
            lambda s, h=horizon: (1 + s).rolling(h, min_periods=h).apply(np.prod, raw=True) - 1
        )
    for horizon in [1, 2, 3, 5, 10]:
        stock_col = "return_1d" if horizon == 1 else f"return_{horizon}d"
        spy_col = "spy_return_1d" if horizon == 1 else f"spy_return_{horizon}d"
        out[f"return_minus_spy_{horizon}d"] = out[stock_col] - out[spy_col]
    for window in [3, 10, 20]:
        out[f"spy_volatility_{window}d"] = g["spy_return_1d"].transform(lambda s, w=window: s.rolling(w, min_periods=w).std())
    rolling_cov = g[["return_1d", "spy_return_1d"]].apply(
        lambda frame: frame["return_1d"].rolling(20, min_periods=20).cov(frame["spy_return_1d"])
    ).reset_index(level=0, drop=True)
    spy_var = g["spy_return_1d"].transform(lambda s: s.rolling(20, min_periods=20).var())
    out["beta_20d"] = safe_divide(rolling_cov, spy_var)

    out["news_count_rolling10"] = g["news_count"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    out["news_count_surprise_vs_5d"] = out["news_count"] - out["news_count_rolling5"]
    out["news_count_surprise_vs_10d"] = out["news_count"] - out["news_count_rolling10"]
    out["market_news_count_rolling10"] = g["market_news_count"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    out["market_news_count_surprise_vs_5d"] = out["market_news_count"] - out["market_news_count_rolling5"]
    out["market_news_count_surprise_vs_10d"] = out["market_news_count"] - out["market_news_count_rolling10"]
    out["sentiment_surprise_vs_5d"] = out["finbert_sentiment_score_mean"] - out["finbert_sentiment_score_rolling5"]
    out["market_sentiment_surprise_vs_5d"] = out["market_finbert_sentiment_score_mean"] - out["market_finbert_sentiment_score_rolling5"]
    out["news_count_x_abs_return_1d"] = out["log_news_count"] * out["abs_return_1d"]
    out["news_count_x_abs_return_3d"] = out["log_news_count"] * out["abs_return_3d"]
    out["sentiment_x_return_1d"] = out["finbert_sentiment_score_mean"] * out["return_1d"]
    out["sentiment_x_return_3d"] = out["finbert_sentiment_score_mean"] * out["return_3d"]
    out["sentiment_x_volume_spike"] = out["finbert_sentiment_score_mean"] * out["volume_vs_20d_mean"]
    out["negative_sentiment_x_volatility_5d"] = out["finbert_negative_mean"] * out["volatility_5d"]
    out["positive_sentiment_x_momentum_5d"] = out["finbert_positive_mean"] * out["return_5d"]
    out["market_sentiment_x_stock_return_1d"] = out["market_finbert_sentiment_score_mean"] * out["return_1d"]
    out["market_news_x_spy_volatility_5d"] = out["market_log_news_count"] * out["spy_volatility_5d"]
    out["stock_news_minus_market_news"] = out["log_news_count"] - out["market_log_news_count"]

    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"fwd_{h}d_excess_return" for h in HORIZONS])


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    base = build_ablation_feature_sets(df)
    price_engineered = [col for col in PRICE_ENGINEERED_FEATURES if col in df.columns]
    news_engineered = [col for col in NEWS_ENGINEERED_FEATURES if col in df.columns]
    return {
        "price_base": base["price_only"],
        "price_expanded": base["price_only"] + price_engineered,
        "news_base": base["news_all_only"],
        "news_expanded": base["news_all_only"] + news_engineered,
        "price_news_base": base["price_news_quality"],
        "price_news_expanded": base["price_news_quality"] + price_engineered + news_engineered,
        "price_quality_expanded": base["price_quality"] + price_engineered,
    }


def fit_classifier(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> dict[str, Any] | None:
    selected = tune_model(train, val, feature_cols, target)
    if selected is None:
        return None
    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[target].astype(int)
    model = build_logistic_model() if selected["selected_model"] == "logistic_balanced" else build_boosting_model(
        selected["selected_model"],
        selected["selected_params"],
        float(y_train_full.mean()),
    )
    model.fit(train_full[feature_cols], y_train_full)
    y_test = test[target].astype(int)
    proba = model.predict_proba(test[feature_cols])[:, 1]
    pred_tuned = (proba >= selected["selected_threshold"]).astype(int)
    pred_default = (proba >= 0.5).astype(int)
    return {
        **selected,
        "accuracy_tuned": accuracy_score(y_test, pred_tuned),
        "balanced_accuracy_tuned": balanced_accuracy_score(y_test, pred_tuned),
        "precision_tuned": precision_score(y_test, pred_tuned, zero_division=0),
        "recall_tuned": recall_score(y_test, pred_tuned, zero_division=0),
        "f1_tuned": f1_score(y_test, pred_tuned, zero_division=0),
        "accuracy_0_5": accuracy_score(y_test, pred_default),
        "balanced_accuracy_0_5": balanced_accuracy_score(y_test, pred_default),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan,
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        "predicted_positive_rate_tuned": float(pred_tuned.mean()),
        "predicted_positive_rate_0_5": float(pred_default.mean()),
    }


def build_regression_model(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", Ridge(**params))])
    if model_name == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**params, max_iter=10000, random_state=42)),
            ]
        )
    if model_name == "random_forest":
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", RandomForestRegressor(**params, random_state=42, n_jobs=1))])
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1)),
            ]
        )
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", XGBRegressor(**params, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1)),
            ]
        )
    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", CatBoostRegressor(**params, loss_function="RMSE", verbose=False, random_seed=42, thread_count=1))])
    raise ValueError(model_name)


def available_regression_models() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for model_name, params in REGRESSION_MODELS:
        if model_name == "lightgbm" and LGBMRegressor is None:
            continue
        if model_name == "xgboost" and XGBRegressor is None:
            continue
        if model_name == "catboost" and CatBoostRegressor is None:
            continue
        out.append((model_name, params))
    return out


def regression_scores(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    actual_direction = (y_true > 0).astype(int)
    pred_direction = (pred > 0).astype(int)
    baseline = np.repeat(float(y_true.mean()), len(y_true))
    return {
        "mae": mean_absolute_error(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": r2_score(y_true, pred),
        "correlation": float(np.corrcoef(y_true, pred)[0, 1]) if np.std(pred) > 0 and np.std(y_true) > 0 else np.nan,
        "baseline_rmse_mean_return": float(np.sqrt(mean_squared_error(y_true, baseline))),
        "directional_accuracy": accuracy_score(actual_direction, pred_direction),
        "directional_balanced_accuracy": balanced_accuracy_score(actual_direction, pred_direction),
        "directional_majority_baseline": float(max(actual_direction.mean(), 1 - actual_direction.mean())),
        "predicted_positive_rate": float(pred_direction.mean()),
    }


def fit_regressor(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> dict[str, Any] | None:
    if len(train) < 80 or len(val) < 25:
        return None
    y_train = train[target].astype(float)
    y_val = val[target].astype(float)
    best: dict[str, Any] | None = None
    for model_name, params in available_regression_models():
        model = build_regression_model(model_name, params)
        model.fit(train[feature_cols], y_train)
        val_pred = model.predict(val[feature_cols])
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        corr = float(np.corrcoef(y_val, val_pred)[0, 1]) if np.std(val_pred) > 0 and np.std(y_val) > 0 else np.nan
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "validation_rmse": rmse,
            "validation_correlation": corr,
            "validation_r2": r2_score(y_val, val_pred),
        }
        if best is None or (row["validation_rmse"], -np.nan_to_num(row["validation_correlation"], nan=-999.0)) < (
            best["validation_rmse"],
            -np.nan_to_num(best["validation_correlation"], nan=-999.0),
        ):
            best = row
    if best is None:
        return None
    train_full = pd.concat([train, val], ignore_index=True)
    model = build_regression_model(best["selected_model"], best["selected_params"])
    model.fit(train_full[feature_cols], train_full[target].astype(float))
    pred = model.predict(test[feature_cols])
    return {**best, **regression_scores(test[target].astype(float), pred)}


def feature_group_summary(results: pd.DataFrame, task: str, metric: str, baseline_feature_set: str, comparison_feature_set: str) -> pd.DataFrame:
    subset = results[results["task"] == task].copy()
    keys = ["scope", "horizon"]
    base = subset[subset["feature_set"] == baseline_feature_set][keys + [metric]].rename(columns={metric: f"{metric}_baseline"})
    comp = subset[subset["feature_set"] == comparison_feature_set][keys + [metric]].rename(columns={metric: f"{metric}_expanded"})
    merged = base.merge(comp, on=keys, how="inner")
    merged["delta"] = merged[f"{metric}_expanded"] - merged[f"{metric}_baseline"]
    merged["baseline_feature_set"] = baseline_feature_set
    merged["comparison_feature_set"] = comparison_feature_set
    merged["metric"] = metric
    return merged


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news))))
    feature_sets = build_feature_sets(df)

    class_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        scoped = scope_frame(df, scope)
        train, val, test = split_holdout(scoped)
        if len(test) < 100 and scope == "pooled_all_tickers":
            continue
        for horizon in HORIZONS:
            class_target = f"target_{horizon}d_excess_gt_0"
            reg_target = f"fwd_{horizon}d_excess_return"
            for feature_set_name, features in feature_sets.items():
                feature_cols = [col for col in features if col in scoped.columns]
                class_result = fit_classifier(train, val, test, feature_cols, class_target)
                if class_result is not None:
                    class_rows.append(
                        {
                            "task": "classification",
                            "scope": scope,
                            "horizon": horizon,
                            "target": class_target,
                            "feature_set": feature_set_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train),
                            "validation_rows": len(val),
                            "test_rows": len(test),
                            **class_result,
                        }
                    )
                reg_result = fit_regressor(train, val, test, feature_cols, reg_target)
                if reg_result is not None:
                    reg_rows.append(
                        {
                            "task": "regression",
                            "scope": scope,
                            "horizon": horizon,
                            "target": reg_target,
                            "feature_set": feature_set_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train),
                            "validation_rows": len(val),
                            "test_rows": len(test),
                            **reg_result,
                        }
                    )

    classification = pd.DataFrame(class_rows)
    regression = pd.DataFrame(reg_rows)
    combined_for_summary = pd.concat(
        [
            classification.assign(score_metric_value=classification["balanced_accuracy_tuned"]),
            regression.assign(score_metric_value=regression["directional_balanced_accuracy"]),
        ],
        ignore_index=True,
        sort=False,
    )
    summary_parts = [
        feature_group_summary(combined_for_summary, "classification", "balanced_accuracy_tuned", "price_base", "price_expanded"),
        feature_group_summary(combined_for_summary, "classification", "balanced_accuracy_tuned", "price_news_base", "price_news_expanded"),
        feature_group_summary(combined_for_summary, "regression", "directional_balanced_accuracy", "price_base", "price_expanded"),
        feature_group_summary(combined_for_summary, "regression", "directional_balanced_accuracy", "price_news_base", "price_news_expanded"),
    ]
    feature_value = pd.concat(summary_parts, ignore_index=True)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "horizons": ", ".join(str(h) for h in HORIZONS),
                "scopes": ", ".join(SCOPES),
                "feature_sets": ", ".join(feature_sets),
                "large_data_rule": "Primary conclusions use pooled_all_tickers with 1,235 test rows; single-ticker rows are reported separately as lower-sample diagnostics.",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    classification.to_csv(paths.tables_dir / f"expanded_feature_classification_results{suffix}.csv", index=False)
    regression.to_csv(paths.tables_dir / f"expanded_feature_regression_results{suffix}.csv", index=False)
    feature_value.to_csv(paths.tables_dir / f"expanded_feature_value_summary{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"expanded_feature_search_metadata{suffix}.csv", index=False)
    return {"classification": classification, "regression": regression, "feature_value": feature_value, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded feature classification and regression search.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nBest pooled classification rows")
    pooled_class = outputs["classification"][outputs["classification"]["scope"] == "pooled_all_tickers"]
    print(
        pooled_class.sort_values(["horizon", "balanced_accuracy_tuned", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon")
        .head(5)
        .to_string(index=False)
    )
    print("\nBest pooled regression directional rows")
    pooled_reg = outputs["regression"][outputs["regression"]["scope"] == "pooled_all_tickers"]
    print(
        pooled_reg.sort_values(["horizon", "directional_balanced_accuracy", "correlation"], ascending=[True, False, False])
        .groupby("horizon")
        .head(5)
        .to_string(index=False)
    )
    print("\nLargest expanded feature deltas")
    print(outputs["feature_value"].sort_values("delta", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
