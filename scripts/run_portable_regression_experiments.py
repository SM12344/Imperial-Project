from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_2024_holdout_ablation import TEST_END, TEST_START, TRAIN_END, VALIDATION_START, add_holdout_filter_columns, split_holdout
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets, build_regression_model
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_target_formulation_search import add_extended_targets


DEFAULT_DATASET = "model_dataset_finbert_quality_complete_2020_2024_polygon_proxy_market_cleaned_news.csv"
DEFAULT_SCORED_NEWS = "news_target_tickers_finbert_scored_2020_2024_polygon_proxy_market_cleaned_news.csv"
DEFAULT_SUFFIX = "2020_2024_polygon_proxy_market_cleaned_news"

TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
SCOPES = ["pooled_all_tickers", *TICKERS]
HORIZONS = [1, 2, 3, 5, 10, 20]
RETURN_KINDS = ["excess", "raw"]

EARLY_FEATURE_SETS = [
    "price_expanded_context",
    "news_expanded_context",
    "price_news_expanded_context",
    "price_quality_expanded_context",
]
FUSION_FEATURE_SETS = {
    "price": "price_expanded_context",
    "news": "news_expanded_context",
    "combined": "price_news_expanded_context",
}
GLOBAL_BLEND_FEATURE_SET = "price_news_expanded_context"

SELECTIVE_COUNTS_POOLED = [300, 500, 800]
SELECTIVE_COUNTS_SINGLE = [50, 100, 150]
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 21)
BOOTSTRAP_SAMPLE_FRACTIONS = [0.75, 1.0, 1.25]
BOOTSTRAP_BAGS = 3

REGRESSION_MODEL_GRID = [
    ("ridge", {"alpha": 0.1}),
    ("ridge", {"alpha": 1.0}),
    ("elastic_net", {"alpha": 0.005, "l1_ratio": 0.1}),
    ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.2}),
    (
        "lightgbm",
        {
            "n_estimators": 80,
            "max_depth": 1,
            "num_leaves": 3,
            "learning_rate": 0.03,
            "min_child_samples": 40,
            "reg_lambda": 15.0,
            "reg_alpha": 0.2,
        },
    ),
]

SLOW_REGRESSION_MODEL_GRID = [
    ("ridge", {"alpha": 10.0}),
    ("random_forest", {"n_estimators": 160, "max_depth": 2, "min_samples_leaf": 30}),
    (
        "xgboost",
        {
            "n_estimators": 120,
            "max_depth": 1,
            "learning_rate": 0.025,
            "min_child_weight": 8,
            "reg_lambda": 10.0,
            "reg_alpha": 0.1,
        },
    ),
    ("catboost", {"iterations": 120, "depth": 1, "learning_rate": 0.025, "l2_leaf_reg": 10.0}),
]


def target_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        specs.append(
            {
                "horizon": horizon,
                "return_kind": "excess",
                "target": f"fwd_{horizon}d_excess_return_audit",
                "target_label": f"{horizon}d_excess_return",
            }
        )
        specs.append(
            {
                "horizon": horizon,
                "return_kind": "raw",
                "target": f"fwd_{horizon}d_return_audit",
                "target_label": f"{horizon}d_raw_return",
            }
        )
    return specs


def add_context(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    context_cols: list[str] = []
    for ticker in TICKERS:
        col = f"ticker_is_{ticker.lower()}"
        out[col] = (out["ticker"] == ticker).astype(int)
        context_cols.append(col)
    out["month_sin"] = np.sin(2 * np.pi * out["trading_date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["trading_date"].dt.month / 12)
    context_cols.extend(["month_sin", "month_cos"])
    return out, context_cols


def load_experiment_frame(project_root: str | None, dataset_name: str, scored_news_name: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    event_daily = build_event_daily_features(scored_news)
    df = add_features_and_targets(base, event_daily)
    df = add_holdout_filter_columns(df)
    df = add_expanded_features(df)
    df = add_extended_targets(df)
    df, context_cols = add_context(df)
    feature_sets = build_feature_sets(df)
    feature_sets["price_base_context"] = feature_sets["price_base"] + context_cols
    feature_sets["price_expanded_context"] = feature_sets["price_expanded"] + context_cols
    feature_sets["news_expanded_context"] = feature_sets["news_expanded"] + context_cols
    feature_sets["price_news_expanded_context"] = feature_sets["price_news_expanded"] + context_cols
    feature_sets["price_quality_expanded_context"] = feature_sets["price_quality_expanded"] + context_cols
    return df, feature_sets


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def clean_feature_cols(feature_cols: list[str], df: pd.DataFrame) -> list[str]:
    blocked = ("target_", "fwd_")
    cleaned = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if col.startswith(blocked) or "future" in col.lower():
            continue
        cleaned.append(col)
    return list(dict.fromkeys(cleaned))


def available_model_grid(include_slow_models: bool = False) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    grid = REGRESSION_MODEL_GRID + (SLOW_REGRESSION_MODEL_GRID if include_slow_models else [])
    for model_name, params in grid:
        try:
            build_regression_model(model_name, params)
        except ImportError:
            continue
        out.append((model_name, params))
    return out


def inner_chronological_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(train["trading_date"].drop_duplicates())
    cut = int(len(dates) * 0.8)
    cut = max(5, min(cut, len(dates) - 1))
    inner_train_dates = dates[:cut]
    inner_val_dates = dates[cut:]
    return train[train["trading_date"].isin(inner_train_dates)].copy(), train[train["trading_date"].isin(inner_val_dates)].copy()


def regression_metrics(y_true: pd.Series, pred: np.ndarray, baseline_value: float) -> dict[str, float]:
    y = y_true.astype(float)
    pred = np.asarray(pred, dtype=float)
    baseline_pred = np.repeat(float(baseline_value), len(y))
    actual_direction = (y > 0).astype(int)
    pred_direction = (pred > 0).astype(int)
    majority_baseline = float(max(actual_direction.mean(), 1 - actual_direction.mean()))
    mse = mean_squared_error(y, pred)
    baseline_mse = mean_squared_error(y, baseline_pred)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mse)),
        "baseline_rmse_train_mean": float(np.sqrt(baseline_mse)),
        "rmse_improvement_vs_train_mean": float(np.sqrt(baseline_mse) - np.sqrt(mse)),
        "r2": r2_score(y, pred),
        "correlation": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else np.nan,
        "directional_accuracy": accuracy_score(actual_direction, pred_direction),
        "directional_balanced_accuracy": balanced_accuracy_score(actual_direction, pred_direction),
        "directional_majority_baseline": majority_baseline,
        "directional_accuracy_minus_baseline": accuracy_score(actual_direction, pred_direction) - majority_baseline,
        "actual_positive_rate": float(actual_direction.mean()),
        "predicted_positive_rate": float(pred_direction.mean()),
        "actual_return_mean": float(y.mean()),
        "predicted_return_mean": float(pred.mean()),
        "actual_return_std": float(y.std()),
        "predicted_return_std": float(pred.std()),
    }


def tune_regressor(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    include_slow_models: bool = False,
) -> dict[str, Any] | None:
    train = train.dropna(subset=[target]).copy()
    val = val.dropna(subset=[target]).copy()
    if len(train) < 80 or len(val) < 25:
        return None
    y_train = train[target].astype(float)
    y_val = val[target].astype(float)
    best: dict[str, Any] | None = None
    for model_name, params in available_model_grid(include_slow_models):
        model = build_regression_model(model_name, params)
        model.fit(train[feature_cols], y_train)
        val_pred = model.predict(val[feature_cols])
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        corr = float(np.corrcoef(y_val, val_pred)[0, 1]) if np.std(val_pred) > 0 and np.std(y_val) > 0 else np.nan
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "validation_rmse": rmse,
            "validation_mae": float(mean_absolute_error(y_val, val_pred)),
            "validation_correlation": corr,
            "validation_r2": float(r2_score(y_val, val_pred)),
        }
        if best is None or (row["validation_rmse"], -np.nan_to_num(row["validation_correlation"], nan=-999.0)) < (
            best["validation_rmse"],
            -np.nan_to_num(best["validation_correlation"], nan=-999.0),
        ):
            best = row
    return best


def fit_selected_regressor(train: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]) -> Pipeline:
    model = build_regression_model(selected["selected_model"], selected["selected_params"])
    model.fit(train[feature_cols], train[target].astype(float))
    return model


def fit_predict_tuned(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    refit_with_validation: bool,
    include_slow_models: bool,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    selected = tune_regressor(train, val, feature_cols, target, include_slow_models)
    if selected is None:
        return None
    fit_df = pd.concat([train, val], ignore_index=True) if refit_with_validation else train
    model = fit_selected_regressor(fit_df, feature_cols, target, selected)
    return model.predict(test[feature_cols]), selected


def run_early_regression(df: pd.DataFrame, feature_sets: dict[str, list[str]], include_slow_models: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in target_specs():
        for scope in SCOPES:
            scoped = scope_frame(df, scope).dropna(subset=[spec["target"]]).copy()
            train, val, test = split_holdout(scoped)
            if len(train) < 80 or len(val) < 25 or len(test) < 25:
                continue
            for feature_set_name in EARLY_FEATURE_SETS:
                feature_cols = clean_feature_cols(feature_sets[feature_set_name], scoped)
                output = fit_predict_tuned(train, val, test, feature_cols, spec["target"], refit_with_validation=True, include_slow_models=include_slow_models)
                if output is None:
                    continue
                pred, selected = output
                baseline_value = float(pd.concat([train, val], ignore_index=True)[spec["target"]].mean())
                rows.append(
                    {
                        "experiment_family": "early_fusion_or_single_modality",
                        "scope": scope,
                        **spec,
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "train_plus_validation_rows": len(train) + len(val),
                        "test_rows": len(test),
                        **selected,
                        **regression_metrics(test[spec["target"]], pred, baseline_value),
                    }
                )
    return pd.DataFrame(rows)


def meta_feature_frame(
    price_pred: np.ndarray,
    news_pred: np.ndarray,
    combined_pred: np.ndarray | None = None,
    tickers: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "price_pred": price_pred,
            "news_pred": news_pred,
            "mean_price_news_pred": (price_pred + news_pred) / 2,
            "news_minus_price_pred": news_pred - price_pred,
            "abs_price_pred": np.abs(price_pred),
            "abs_news_pred": np.abs(news_pred),
        }
    )
    if combined_pred is not None:
        out["combined_pred"] = combined_pred
        out["combined_minus_mean_pred"] = combined_pred - out["mean_price_news_pred"]
    if tickers is not None:
        ticker_values = tickers.reset_index(drop=True)
        for ticker in TICKERS:
            out[f"ticker_is_{ticker.lower()}"] = (ticker_values == ticker).astype(int)
    return out


def best_weight_by_validation_rmse(y_val: pd.Series, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    best_weight = 0.5
    best_rmse = np.inf
    y = y_val.astype(float)
    for weight in BLEND_WEIGHTS:
        pred = weight * pred_a + (1 - weight) * pred_b
        rmse = float(np.sqrt(mean_squared_error(y, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight, best_rmse


def selective_rows(
    base_row: dict[str, Any],
    val_target: pd.Series,
    val_pred: np.ndarray,
    test_target: pd.Series,
    test_pred: np.ndarray,
    baseline_value: float,
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = SELECTIVE_COUNTS_POOLED if scope == "pooled_all_tickers" else SELECTIVE_COUNTS_SINGLE
    val_confidence = np.abs(val_pred)
    test_confidence = np.abs(test_pred)
    for requested_rows in counts:
        if len(val_confidence) < requested_rows:
            continue
        cutoff = float(np.sort(val_confidence)[::-1][requested_rows - 1])
        mask = test_confidence >= cutoff
        covered_rows = int(mask.sum())
        if covered_rows < 25:
            continue
        covered_target = test_target.iloc[np.where(mask)[0]]
        covered_pred = test_pred[mask]
        metric_values = regression_metrics(covered_target, covered_pred, baseline_value)
        rows.append(
            {
                **base_row,
                "selection_method": "validation_abs_predicted_return_cutoff",
                "validation_requested_rows": requested_rows,
                "confidence_cutoff_abs_predicted_return": cutoff,
                "test_covered_rows": covered_rows,
                "test_coverage": covered_rows / len(test_target),
                **{f"covered_{k}": v for k, v in metric_values.items()},
            }
        )
    return rows


def run_late_fusion_regression(df: pd.DataFrame, feature_sets: dict[str, list[str]], include_slow_models: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    selective_result_rows: list[dict[str, Any]] = []
    for spec in target_specs():
        for scope in SCOPES:
            scoped = scope_frame(df, scope).dropna(subset=[spec["target"]]).copy()
            train, val, test = split_holdout(scoped)
            if len(train) < 100 or len(val) < 50 or len(test) < 25:
                continue
            inner_train, inner_val = inner_chronological_split(train)
            price_cols = clean_feature_cols(feature_sets[FUSION_FEATURE_SETS["price"]], scoped)
            news_cols = clean_feature_cols(feature_sets[FUSION_FEATURE_SETS["news"]], scoped)
            combined_cols = clean_feature_cols(feature_sets[FUSION_FEATURE_SETS["combined"]], scoped)

            price_selected = tune_regressor(inner_train, inner_val, price_cols, spec["target"], include_slow_models)
            news_selected = tune_regressor(inner_train, inner_val, news_cols, spec["target"], include_slow_models)
            combined_selected = tune_regressor(inner_train, inner_val, combined_cols, spec["target"], include_slow_models)
            if price_selected is None or news_selected is None or combined_selected is None:
                continue

            price_model = fit_selected_regressor(train, price_cols, spec["target"], price_selected)
            news_model = fit_selected_regressor(train, news_cols, spec["target"], news_selected)
            combined_model = fit_selected_regressor(train, combined_cols, spec["target"], combined_selected)

            val_price = price_model.predict(val[price_cols])
            val_news = news_model.predict(val[news_cols])
            val_combined = combined_model.predict(val[combined_cols])
            test_price = price_model.predict(test[price_cols])
            test_news = news_model.predict(test[news_cols])
            test_combined = combined_model.predict(test[combined_cols])

            y_val = val[spec["target"]].astype(float)
            y_test = test[spec["target"]].astype(float)
            baseline_value = float(train[spec["target"]].mean())
            price_weight, weighted_val_rmse = best_weight_by_validation_rmse(y_val, val_price, val_news)
            val_weighted = price_weight * val_price + (1 - price_weight) * val_news
            test_weighted = price_weight * test_price + (1 - price_weight) * test_news

            meta_train = meta_feature_frame(val_price, val_news, val_combined, val["ticker"] if scope == "pooled_all_tickers" else None)
            meta_test = meta_feature_frame(test_price, test_news, test_combined, test["ticker"] if scope == "pooled_all_tickers" else None)
            meta_model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
            meta_model.fit(meta_train, y_val)
            val_meta = meta_model.predict(meta_train)
            test_meta = meta_model.predict(meta_test)

            candidates = [
                ("price_only_base", val_price, test_price, price_selected, len(price_cols), np.nan),
                ("news_only_base", val_news, test_news, news_selected, len(news_cols), np.nan),
                ("early_combined_base", val_combined, test_combined, combined_selected, len(combined_cols), np.nan),
                (
                    "late_fusion_validation_weighted_average",
                    val_weighted,
                    test_weighted,
                    {"selected_model": "weighted_average", "selected_params": {"price_weight": price_weight}, "validation_rmse": weighted_val_rmse},
                    2,
                    price_weight,
                ),
                (
                    "late_fusion_stacked_ridge",
                    val_meta,
                    test_meta,
                    {
                        "selected_model": "ridge_meta",
                        "selected_params": {"alpha": 1.0},
                        "validation_rmse": float(np.sqrt(mean_squared_error(y_val, val_meta))),
                    },
                    meta_train.shape[1],
                    np.nan,
                ),
            ]

            for model_label, val_pred, test_pred, selected, feature_count, price_weight_value in candidates:
                base_row = {
                    "experiment_family": "late_fusion_regression",
                    "scope": scope,
                    **spec,
                    "model_label": model_label,
                    "feature_count": feature_count,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                    "price_weight": price_weight_value,
                    "selected_model": selected.get("selected_model"),
                    "selected_params": selected.get("selected_params"),
                    "validation_rmse": selected.get("validation_rmse", np.nan),
                    "validation_correlation": selected.get("validation_correlation", np.nan),
                    "validation_r2": selected.get("validation_r2", np.nan),
                }
                metric_values = regression_metrics(y_test, test_pred, baseline_value)
                result_rows.append({**base_row, **metric_values})
                selective_result_rows.extend(
                    selective_rows(base_row, y_val, val_pred, y_test, test_pred, baseline_value, scope)
                )
    return pd.DataFrame(result_rows), pd.DataFrame(selective_result_rows)


def run_global_ticker_blend_regression(df: pd.DataFrame, feature_sets: dict[str, list[str]], include_slow_models: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    selective_result_rows: list[dict[str, Any]] = []
    for spec in target_specs():
        all_df = df.dropna(subset=[spec["target"]]).copy()
        train_all, val_all, test_all = split_holdout(all_df)
        if len(train_all) < 200 or len(val_all) < 100 or len(test_all) < 100:
            continue
        feature_cols = clean_feature_cols(feature_sets[GLOBAL_BLEND_FEATURE_SET], all_df)
        inner_train_all, inner_val_all = inner_chronological_split(train_all)
        global_selected = tune_regressor(inner_train_all, inner_val_all, feature_cols, spec["target"], include_slow_models)
        if global_selected is None:
            continue
        global_model = fit_selected_regressor(train_all, feature_cols, spec["target"], global_selected)
        global_val = val_all[["ticker", "trading_date", spec["target"]]].copy()
        global_val["global_pred"] = global_model.predict(val_all[feature_cols])
        global_test = test_all[["ticker", "trading_date", spec["target"]]].copy()
        global_test["global_pred"] = global_model.predict(test_all[feature_cols])

        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        model_diagnostics: list[dict[str, Any]] = []
        for ticker in TICKERS:
            ticker_df = all_df[all_df["ticker"] == ticker].copy()
            train_ticker, val_ticker, test_ticker = split_holdout(ticker_df)
            if len(train_ticker) < 80 or len(val_ticker) < 25 or len(test_ticker) < 25:
                continue
            inner_train_ticker, inner_val_ticker = inner_chronological_split(train_ticker)
            ticker_selected = tune_regressor(inner_train_ticker, inner_val_ticker, feature_cols, spec["target"], include_slow_models)
            if ticker_selected is None:
                continue
            ticker_model = fit_selected_regressor(train_ticker, feature_cols, spec["target"], ticker_selected)
            tv = val_ticker[["ticker", "trading_date", spec["target"]]].copy()
            tv["ticker_pred"] = ticker_model.predict(val_ticker[feature_cols])
            tt = test_ticker[["ticker", "trading_date", spec["target"]]].copy()
            tt["ticker_pred"] = ticker_model.predict(test_ticker[feature_cols])
            val_parts.append(tv)
            test_parts.append(tt)
            model_diagnostics.append(
                {
                    "ticker": ticker,
                    "ticker_selected_model": ticker_selected["selected_model"],
                    "ticker_selected_params": ticker_selected["selected_params"],
                    "ticker_validation_rmse": ticker_selected["validation_rmse"],
                }
            )

        if not val_parts or not test_parts:
            continue
        val_predictions = pd.concat(val_parts, ignore_index=True).merge(global_val, on=["ticker", "trading_date", spec["target"]], how="inner")
        test_predictions = pd.concat(test_parts, ignore_index=True).merge(global_test, on=["ticker", "trading_date", spec["target"]], how="inner")
        if len(val_predictions) < 100 or len(test_predictions) < 100:
            continue

        y_val = val_predictions[spec["target"]].astype(float)
        y_test = test_predictions[spec["target"]].astype(float)
        ticker_weight, weighted_val_rmse = best_weight_by_validation_rmse(y_val, val_predictions["ticker_pred"].to_numpy(), val_predictions["global_pred"].to_numpy())
        val_weighted = ticker_weight * val_predictions["ticker_pred"].to_numpy() + (1 - ticker_weight) * val_predictions["global_pred"].to_numpy()
        test_weighted = ticker_weight * test_predictions["ticker_pred"].to_numpy() + (1 - ticker_weight) * test_predictions["global_pred"].to_numpy()

        meta_train = meta_feature_frame(
            val_predictions["global_pred"].to_numpy(),
            val_predictions["ticker_pred"].to_numpy(),
            None,
            val_predictions["ticker"],
        )
        meta_test = meta_feature_frame(
            test_predictions["global_pred"].to_numpy(),
            test_predictions["ticker_pred"].to_numpy(),
            None,
            test_predictions["ticker"],
        )
        meta_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        meta_model.fit(meta_train, y_val)
        val_meta = meta_model.predict(meta_train)
        test_meta = meta_model.predict(meta_test)
        baseline_value = float(train_all[spec["target"]].mean())
        candidates = [
            ("global_only", val_predictions["global_pred"].to_numpy(), test_predictions["global_pred"].to_numpy(), 0.0),
            ("ticker_specific_only", val_predictions["ticker_pred"].to_numpy(), test_predictions["ticker_pred"].to_numpy(), 1.0),
            ("simple_average_50_50", 0.5 * val_predictions["ticker_pred"].to_numpy() + 0.5 * val_predictions["global_pred"].to_numpy(), 0.5 * test_predictions["ticker_pred"].to_numpy() + 0.5 * test_predictions["global_pred"].to_numpy(), 0.5),
            ("validation_weighted_average", val_weighted, test_weighted, ticker_weight),
            ("meta_stacked_ridge", val_meta, test_meta, np.nan),
        ]

        for blend_method, val_pred, test_pred, ticker_weight_value in candidates:
            base_row = {
                "experiment_family": "global_ticker_blend_regression",
                "scope": "pooled_all_tickers",
                **spec,
                "blend_method": blend_method,
                "feature_set": GLOBAL_BLEND_FEATURE_SET,
                "feature_count": len(feature_cols),
                "train_rows_global": len(train_all),
                "validation_rows": len(val_predictions),
                "test_rows": len(test_predictions),
                "ticker_weight": ticker_weight_value,
                "selected_model": global_selected["selected_model"] if blend_method == "global_only" else "mixed",
                "selected_params": global_selected["selected_params"] if blend_method == "global_only" else {},
                "validation_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
                "validation_correlation": float(np.corrcoef(y_val, val_pred)[0, 1]) if np.std(val_pred) > 0 and np.std(y_val) > 0 else np.nan,
                "validation_r2": float(r2_score(y_val, val_pred)),
                "ticker_model_diagnostics": model_diagnostics,
            }
            result_rows.append({**base_row, **regression_metrics(y_test, test_pred, baseline_value)})
            selective_result_rows.extend(
                selective_rows(base_row, y_val, val_pred, y_test, test_pred, baseline_value, "pooled_all_tickers")
            )

            ticker_eval = test_predictions[["ticker", spec["target"]]].copy()
            ticker_eval["pred"] = test_pred
            for ticker, part in ticker_eval.groupby("ticker"):
                if len(part) < 25:
                    continue
                ticker_base = {
                    **base_row,
                    "scope": ticker,
                    "test_rows": len(part),
                    "validation_rows": int((val_predictions["ticker"] == ticker).sum()),
                }
                result_rows.append(
                    {
                        **ticker_base,
                        **regression_metrics(part[spec["target"]], part["pred"].to_numpy(), baseline_value),
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(selective_result_rows)


def bootstrap_sample(train: pd.DataFrame, sample_fraction: float, random_state: int) -> pd.DataFrame:
    sample_size = max(1, int(round(len(train) * sample_fraction)))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(train.index.to_numpy(), size=sample_size, replace=True)
    return train.loc[idx].reset_index(drop=True)


def run_bootstrap_regression(df: pd.DataFrame, feature_sets: dict[str, list[str]], include_slow_models: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in target_specs():
        for scope in SCOPES:
            scoped = scope_frame(df, scope).dropna(subset=[spec["target"]]).copy()
            train, val, test = split_holdout(scoped)
            if len(train) < 80 or len(val) < 25 or len(test) < 25:
                continue
            feature_cols = clean_feature_cols(feature_sets[GLOBAL_BLEND_FEATURE_SET], scoped)
            selected = tune_regressor(train, val, feature_cols, spec["target"], include_slow_models)
            if selected is None:
                continue
            train_full = pd.concat([train, val], ignore_index=True)
            y_test = test[spec["target"]].astype(float)
            baseline_value = float(train_full[spec["target"]].mean())
            for sample_fraction in BOOTSTRAP_SAMPLE_FRACTIONS:
                bag_predictions: list[np.ndarray] = []
                for bag_idx in range(BOOTSTRAP_BAGS):
                    bag_train = bootstrap_sample(train_full, sample_fraction, 42 + bag_idx)
                    model = fit_selected_regressor(bag_train, feature_cols, spec["target"], selected)
                    bag_predictions.append(model.predict(test[feature_cols]))
                pred = np.vstack(bag_predictions).mean(axis=0)
                rows.append(
                    {
                        "experiment_family": "bootstrap_regression_ensemble",
                        "scope": scope,
                        **spec,
                        "feature_set": GLOBAL_BLEND_FEATURE_SET,
                        "feature_count": len(feature_cols),
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "train_plus_validation_rows": len(train_full),
                        "test_rows": len(test),
                        "bootstrap_bags": BOOTSTRAP_BAGS,
                        "bootstrap_sample_fraction": sample_fraction,
                        **selected,
                        **regression_metrics(y_test, pred, baseline_value),
                    }
                )
    return pd.DataFrame(rows)


def add_defensible_flags(df: pd.DataFrame, row_col: str = "test_rows", prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    acc_col = f"{prefix}directional_accuracy"
    ba_col = f"{prefix}directional_balanced_accuracy"
    delta_col = f"{prefix}directional_accuracy_minus_baseline"
    out["defensible_70_direction_candidate"] = (
        (out[row_col] >= 500)
        & (out[acc_col] >= 0.70)
        & (out[delta_col] >= 0.03)
        & (out[ba_col] >= 0.55)
    )
    return out


def build_summary(
    early: pd.DataFrame,
    late: pd.DataFrame,
    global_blend: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selective: pd.DataFrame,
) -> pd.DataFrame:
    all_rows = pd.concat([early, late, global_blend, bootstrap], ignore_index=True, sort=False)
    summary_rows: list[dict[str, Any]] = []
    if not all_rows.empty:
        large = all_rows[all_rows["test_rows"] >= 500].copy()
        if not large.empty:
            best_direction = large.sort_values(
                ["directional_accuracy", "directional_accuracy_minus_baseline", "directional_balanced_accuracy"],
                ascending=[False, False, False],
            ).iloc[0]
            best_r2 = large.sort_values(["r2", "correlation"], ascending=[False, False]).iloc[0]
            best_rmse = large.sort_values(["rmse_improvement_vs_train_mean", "r2"], ascending=[False, False]).iloc[0]
            for label, row in [
                ("best_large_directional_accuracy", best_direction),
                ("best_large_r2", best_r2),
                ("best_large_rmse_improvement", best_rmse),
            ]:
                summary_rows.append({"summary_group": label, **row.to_dict()})
        defensible = all_rows[all_rows.get("defensible_70_direction_candidate", False) == True]
        summary_rows.append(
            {
                "summary_group": "defensible_large_70_all_row_count",
                "candidate_count": int(len(defensible)),
                "note": "Requires >=500 test rows, directional accuracy >=0.70, +0.03 over majority-direction baseline, balanced accuracy >=0.55.",
            }
        )
    if not selective.empty:
        large_selective = selective[selective["test_covered_rows"] >= 500].copy()
        if not large_selective.empty:
            best_selective = large_selective.sort_values(
                ["covered_directional_accuracy", "covered_directional_accuracy_minus_baseline", "covered_directional_balanced_accuracy"],
                ascending=[False, False, False],
            ).iloc[0]
            summary_rows.append({"summary_group": "best_large_selective_directional_accuracy", **best_selective.to_dict()})
        defensible_selective = selective[selective.get("defensible_70_direction_candidate", False) == True]
        summary_rows.append(
            {
                "summary_group": "defensible_large_70_selective_count",
                "candidate_count": int(len(defensible_selective)),
                "note": "Selective cutoffs are chosen on 2023 validation absolute predicted return and applied to 2024 only.",
            }
        )
    return pd.DataFrame(summary_rows)


def read_table_if_exists(path: Any) -> pd.DataFrame:
    path = pd.io.common.stringify_path(path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def run_pipeline(
    project_root: str | None,
    dataset_name: str,
    scored_news_name: str,
    output_suffix: str,
    idea: str,
    include_slow_models: bool,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    suffix = f"_{output_suffix}" if output_suffix else ""

    if idea == "summary":
        early = read_table_if_exists(paths.tables_dir / f"portable_regression_early_results{suffix}.csv")
        late = read_table_if_exists(paths.tables_dir / f"portable_regression_late_fusion_results{suffix}.csv")
        global_blend = read_table_if_exists(paths.tables_dir / f"portable_regression_global_ticker_blend_results{suffix}.csv")
        bootstrap = read_table_if_exists(paths.tables_dir / f"portable_regression_bootstrap_results{suffix}.csv")
        selective = read_table_if_exists(paths.tables_dir / f"portable_regression_selective_results{suffix}.csv")
        overview = pd.DataFrame(
            [
                {
                    "dataset_name": dataset_name,
                    "scored_news_name": scored_news_name,
                    "target_count": len(target_specs()),
                    "horizons": ", ".join(str(h) for h in HORIZONS),
                    "return_kinds": ", ".join(RETURN_KINDS),
                    "idea_mode": idea,
                    "include_slow_models": include_slow_models,
                    "early_rows": len(early),
                    "late_fusion_rows": len(late),
                    "global_blend_rows": len(global_blend),
                    "bootstrap_rows": len(bootstrap),
                    "selective_rows": len(selective),
                }
            ]
        )
        summary = build_summary(early, late, global_blend, bootstrap, selective)
        metadata = pd.DataFrame(
            [
                {
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_name": dataset_name,
                    "scored_news_name": scored_news_name,
                    "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                    "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                    "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                    "targets": "continuous raw and excess future returns for 1, 2, 3, 5, 10, and 20 trading days",
                    "ported_ideas": "expanded features, price/news early fusion, price-only/news-only late fusion, pooled/ticker-specific weighted blending, validation-selected selective confidence filtering, bootstrap regression ensembling",
                    "idea_mode": idea,
                    "include_slow_models": include_slow_models,
                    "model_grid": ", ".join(name for name, _ in available_model_grid(include_slow_models)),
                    "selection_policy": "Combined from already completed phase outputs; each phase used train/validation only for selection and 2024 only for final test.",
                    "accuracy_policy": "Directional accuracy is computed from the sign of the predicted continuous return; continuous metrics remain the primary regression metrics.",
                    "defensible_70_rule": ">=500 test/covered rows, directional accuracy >=0.70, +0.03 over majority-direction baseline, balanced accuracy >=0.55",
                }
            ]
        )
        overview.to_csv(paths.tables_dir / f"portable_regression_overview{suffix}.csv", index=False)
        summary.to_csv(paths.tables_dir / f"portable_regression_best_summary{suffix}.csv", index=False)
        metadata.to_csv(paths.tables_dir / f"portable_regression_metadata{suffix}.csv", index=False)
        return {
            "early": early,
            "late": late,
            "global_blend": global_blend,
            "bootstrap": bootstrap,
            "selective": selective,
            "overview": overview,
            "summary": summary,
            "metadata": metadata,
        }

    df, feature_sets = load_experiment_frame(project_root, dataset_name, scored_news_name)

    early = pd.DataFrame()
    late = pd.DataFrame()
    late_selective = pd.DataFrame()
    global_blend = pd.DataFrame()
    global_selective = pd.DataFrame()
    bootstrap = pd.DataFrame()

    if idea in {"all", "early"}:
        early = add_defensible_flags(run_early_regression(df, feature_sets, include_slow_models))
        early.to_csv(paths.tables_dir / f"portable_regression_early_results{suffix}.csv", index=False)
    if idea in {"all", "late"}:
        late, late_selective = run_late_fusion_regression(df, feature_sets, include_slow_models)
        late = add_defensible_flags(late)
        late.to_csv(paths.tables_dir / f"portable_regression_late_fusion_results{suffix}.csv", index=False)
    if idea in {"all", "global_blend"}:
        global_blend, global_selective = run_global_ticker_blend_regression(df, feature_sets, include_slow_models)
        global_blend = add_defensible_flags(global_blend)
        global_blend.to_csv(paths.tables_dir / f"portable_regression_global_ticker_blend_results{suffix}.csv", index=False)
    if idea in {"all", "bootstrap"}:
        bootstrap = add_defensible_flags(run_bootstrap_regression(df, feature_sets, include_slow_models))
        bootstrap.to_csv(paths.tables_dir / f"portable_regression_bootstrap_results{suffix}.csv", index=False)

    selective = pd.concat([late_selective, global_selective], ignore_index=True, sort=False)
    if not selective.empty:
        selective = add_defensible_flags(selective, row_col="test_covered_rows", prefix="covered_")
        selective.to_csv(paths.tables_dir / f"portable_regression_selective_results{suffix}.csv", index=False)

    overview = pd.DataFrame(
        [
            {
                "rows_after_feature_target_build": len(df),
                "tickers": ", ".join(sorted(df["ticker"].dropna().unique())),
                "first_trading_date": df["trading_date"].min().date(),
                "last_trading_date": df["trading_date"].max().date(),
                "target_count": len(target_specs()),
                "horizons": ", ".join(str(h) for h in HORIZONS),
                "return_kinds": ", ".join(RETURN_KINDS),
                "idea_mode": idea,
                "include_slow_models": include_slow_models,
                "early_rows": len(early),
                "late_fusion_rows": len(late),
                "global_blend_rows": len(global_blend),
                "bootstrap_rows": len(bootstrap),
                "selective_rows": len(selective),
            }
        ]
    )
    summary = build_summary(early, late, global_blend, bootstrap, selective)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "targets": "continuous raw and excess future returns for 1, 2, 3, 5, 10, and 20 trading days",
                "ported_ideas": "expanded features, price/news early fusion, price-only/news-only late fusion, pooled/ticker-specific weighted blending, validation-selected selective confidence filtering, bootstrap regression ensembling",
                "idea_mode": idea,
                "include_slow_models": include_slow_models,
                "model_grid": ", ".join(name for name, _ in available_model_grid(include_slow_models)),
                "selection_policy": "Model choices and cutoffs use train/validation only; 2024 is used only as the final test period.",
                "accuracy_policy": "Directional accuracy is computed from the sign of the predicted continuous return; continuous metrics remain the primary regression metrics.",
                "defensible_70_rule": ">=500 test/covered rows, directional accuracy >=0.70, +0.03 over majority-direction baseline, balanced accuracy >=0.55",
            }
        ]
    )

    overview.to_csv(paths.tables_dir / f"portable_regression_overview{suffix}.csv", index=False)
    summary.to_csv(paths.tables_dir / f"portable_regression_best_summary{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"portable_regression_metadata{suffix}.csv", index=False)
    return {
        "early": early,
        "late": late,
        "global_blend": global_blend,
        "bootstrap": bootstrap,
        "selective": selective,
        "overview": overview,
        "summary": summary,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression versions of the portable classification-improvement ideas.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--scored-news-name", default=DEFAULT_SCORED_NEWS)
    parser.add_argument("--output-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--idea", default="all", choices=["all", "early", "late", "global_blend", "bootstrap", "summary"])
    parser.add_argument("--include-slow-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix, args.idea, args.include_slow_models)
    print("Metadata")
    print(outputs["metadata"].to_string(index=False))
    print("\nOverview")
    print(outputs["overview"].to_string(index=False))
    print("\nBest summary")
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
