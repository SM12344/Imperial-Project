from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    best_threshold,
    build_ablation_feature_sets,
    split_holdout,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import CATBOOST_GRID, LIGHTGBM_GRID, XGBOOST_GRID, build_model as build_boosting_model


TARGET = "target_5d_excess_gt_0"
SCOPES = ["AMZN", "pooled_all_tickers"]
MODEL_NAMES = ["logistic_balanced", "lightgbm", "xgboost", "catboost"]


def build_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )


def model_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "logistic_balanced":
        return [{}]
    if model_name == "lightgbm":
        return LIGHTGBM_GRID
    if model_name == "xgboost":
        return XGBOOST_GRID
    if model_name == "catboost":
        return CATBOOST_GRID
    raise ValueError(model_name)


def build_candidate_model(model_name: str, params: dict[str, Any], positive_rate: float) -> Pipeline:
    if model_name == "logistic_balanced":
        return build_logistic_model()
    return build_boosting_model(model_name, params, positive_rate)


def inner_chronological_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(train["trading_date"].drop_duplicates())
    cut = max(5, int(len(dates) * 0.8))
    if cut >= len(dates):
        cut = len(dates) - 1
    train_dates = dates[:cut]
    inner_val_dates = dates[cut:]
    return train[train["trading_date"].isin(train_dates)].copy(), train[train["trading_date"].isin(inner_val_dates)].copy()


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def tune_base_model(train: pd.DataFrame, feature_cols: list[str], target: str, modality: str) -> dict[str, Any]:
    inner_train, inner_val = inner_chronological_split(train)
    y_inner = inner_train[target].astype(int)
    y_val = inner_val[target].astype(int)
    if y_inner.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError(f"Cannot tune {modality}: only one class in inner split.")

    best: dict[str, Any] | None = None
    for model_name in MODEL_NAMES:
        for params in model_grid(model_name):
            model = build_candidate_model(model_name, params, float(y_inner.mean()))
            model.fit(inner_train[feature_cols], y_inner)
            proba = model.predict_proba(inner_val[feature_cols])[:, 1]
            threshold, val_ba = best_threshold(y_val, proba)
            row = {
                "modality": modality,
                "selected_model": model_name,
                "selected_params": params,
                "inner_validation_roc_auc": roc_auc_score(y_val, proba),
                "inner_validation_balanced_accuracy": val_ba,
                "inner_selected_threshold": threshold,
                "inner_train_rows": len(inner_train),
                "inner_validation_rows": len(inner_val),
            }
            if best is None or (row["inner_validation_roc_auc"], row["inner_validation_balanced_accuracy"]) > (
                best["inner_validation_roc_auc"],
                best["inner_validation_balanced_accuracy"],
            ):
                best = row
    if best is None:
        raise RuntimeError(f"No model selected for {modality}.")
    return best


def fit_selected_base(train: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]) -> Pipeline:
    y_train = train[target].astype(int)
    model = build_candidate_model(selected["selected_model"], selected["selected_params"], float(y_train.mean()))
    model.fit(train[feature_cols], y_train)
    return model


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def add_ticker_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    ticker_cols = []
    for ticker in sorted(out["ticker"].dropna().unique()):
        col = f"ticker_is_{ticker.lower()}"
        out[col] = (out["ticker"] == ticker).astype(int)
        ticker_cols.append(col)
    return out, ticker_cols


def meta_features(price_proba: np.ndarray, news_proba: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_proba": price_proba,
            "news_proba": news_proba,
            "mean_proba": (price_proba + news_proba) / 2,
            "proba_gap_news_minus_price": news_proba - price_proba,
            "max_proba": np.maximum(price_proba, news_proba),
            "min_proba": np.minimum(price_proba, news_proba),
        }
    )


def run_scope(df: pd.DataFrame, feature_sets: dict[str, list[str]], scope: str, include_ticker_flags: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    work = scope_frame(df, scope)
    ticker_cols: list[str] = []
    variant = "with_ticker_flags" if include_ticker_flags else "no_ticker_flags"
    if include_ticker_flags:
        work, ticker_cols = add_ticker_flags(work)

    train, val, test = split_holdout(work)
    y_val = val[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    price_features = [col for col in feature_sets["price_only"] + ticker_cols if col in work.columns]
    news_features = [col for col in feature_sets["news_all_only"] + ticker_cols if col in work.columns]
    combined_features = [col for col in feature_sets["price_news_quality"] + ticker_cols if col in work.columns]

    price_selected = tune_base_model(train, price_features, TARGET, "price_only")
    news_selected = tune_base_model(train, news_features, TARGET, "news_only")
    combined_selected = tune_base_model(train, combined_features, TARGET, "early_fusion_price_news")

    price_model = fit_selected_base(train, price_features, TARGET, price_selected)
    news_model = fit_selected_base(train, news_features, TARGET, news_selected)
    combined_model = fit_selected_base(train, combined_features, TARGET, combined_selected)

    val_price = price_model.predict_proba(val[price_features])[:, 1]
    val_news = news_model.predict_proba(val[news_features])[:, 1]
    val_combined = combined_model.predict_proba(val[combined_features])[:, 1]
    test_price = price_model.predict_proba(test[price_features])[:, 1]
    test_news = news_model.predict_proba(test[news_features])[:, 1]
    test_combined = combined_model.predict_proba(test[combined_features])[:, 1]

    meta_train = meta_features(val_price, val_news)
    meta_test = meta_features(test_price, test_news)
    meta_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    meta_model.fit(meta_train, y_val)
    val_meta = meta_model.predict_proba(meta_train)[:, 1]
    test_meta = meta_model.predict_proba(meta_test)[:, 1]
    meta_threshold, meta_val_ba = best_threshold(y_val, val_meta)

    average_val = (val_price + val_news) / 2
    average_test = (test_price + test_news) / 2
    average_threshold, average_val_ba = best_threshold(y_val, average_val)

    result_rows = []
    for model_label, proba, threshold, val_proba, val_ba, feature_count, selected in [
        ("price_only_base", test_price, price_selected["inner_selected_threshold"], val_price, np.nan, len(price_features), price_selected),
        ("news_only_base", test_news, news_selected["inner_selected_threshold"], val_news, np.nan, len(news_features), news_selected),
        ("early_fusion_price_news", test_combined, combined_selected["inner_selected_threshold"], val_combined, np.nan, len(combined_features), combined_selected),
        ("late_fusion_average", average_test, average_threshold, average_val, average_val_ba, 2, {"selected_model": "mean_probability", "selected_params": {}}),
        ("late_fusion_stacked_logistic", test_meta, meta_threshold, val_meta, meta_val_ba, meta_train.shape[1], {"selected_model": "logistic_meta", "selected_params": {}}),
    ]:
        result_rows.append(
            {
                "scope": scope,
                "variant": variant,
                "model_label": model_label,
                "target": TARGET,
                "threshold": threshold,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "feature_count": feature_count,
                "test_positive_rate": float(y_test.mean()),
                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                "validation_roc_auc": roc_auc_score(y_val, val_proba),
                "validation_balanced_accuracy_at_threshold": val_ba,
                "selected_model": selected["selected_model"],
                "selected_params": selected["selected_params"],
                **metrics(y_test, proba, threshold),
            }
        )

    diagnostic_rows = []
    for selected in [price_selected, news_selected, combined_selected]:
        diagnostic_rows.append({"scope": scope, "variant": variant, **selected})

    by_ticker_rows = []
    if scope == "pooled_all_tickers":
        test_eval = test[["ticker"]].copy()
        test_eval["y_true"] = y_test.to_numpy()
        test_eval["late_fusion_proba"] = test_meta
        test_eval["late_fusion_pred"] = (test_meta >= meta_threshold).astype(int)
        for ticker, part in test_eval.groupby("ticker"):
            if part["y_true"].nunique() < 2:
                continue
            by_ticker_rows.append(
                {
                    "scope": scope,
                    "variant": variant,
                    "ticker": ticker,
                    "rows": len(part),
                    "positive_rate": float(part["y_true"].mean()),
                    "majority_baseline_accuracy": float(max(part["y_true"].mean(), 1 - part["y_true"].mean())),
                    "accuracy": accuracy_score(part["y_true"], part["late_fusion_pred"]),
                    "balanced_accuracy": balanced_accuracy_score(part["y_true"], part["late_fusion_pred"]),
                    "roc_auc": roc_auc_score(part["y_true"], part["late_fusion_proba"]),
                    "precision": precision_score(part["y_true"], part["late_fusion_pred"], zero_division=0),
                    "recall": recall_score(part["y_true"], part["late_fusion_pred"], zero_division=0),
                    "f1": f1_score(part["y_true"], part["late_fusion_pred"], zero_division=0),
                }
            )
    return result_rows, diagnostic_rows, by_ticker_rows


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    result_rows = []
    diagnostic_rows = []
    by_ticker_rows = []
    for scope in SCOPES:
        variants = [False, True] if scope == "pooled_all_tickers" else [False]
        for include_ticker_flags in variants:
            rows, diagnostics, by_ticker = run_scope(df, feature_sets, scope, include_ticker_flags)
            result_rows.extend(rows)
            diagnostic_rows.extend(diagnostics)
            by_ticker_rows.extend(by_ticker)

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    by_ticker = pd.DataFrame(by_ticker_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "target": TARGET,
                "train_period": "2020-01-01 to 2022-12-31",
                "base_model_inner_tuning": "last 20 percent of 2020-2022 trading dates",
                "meta_model_training_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                "test_policy": "2024 is untouched and not used for base model, meta model, or threshold selection",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"late_fusion_results{suffix}.csv", index=False)
    diagnostics.to_csv(paths.tables_dir / f"late_fusion_base_model_diagnostics{suffix}.csv", index=False)
    by_ticker.to_csv(paths.tables_dir / f"late_fusion_by_ticker{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"late_fusion_metadata{suffix}.csv", index=False)
    return {"results": results, "diagnostics": diagnostics, "by_ticker": by_ticker, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run late-fusion price/news stacking experiment.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print("Metadata")
    print(outputs["metadata"].to_string(index=False))
    print("\nResults")
    print(outputs["results"].to_string(index=False))
    print("\nBase diagnostics")
    print(outputs["diagnostics"].to_string(index=False))
    print("\nBy ticker")
    print(outputs["by_ticker"].to_string(index=False))


if __name__ == "__main__":
    main()
