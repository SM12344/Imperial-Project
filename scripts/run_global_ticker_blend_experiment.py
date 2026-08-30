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
    build_logistic_model,
    split_holdout,
    tune_model,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
TARGETS = ["target_5d_excess_gt_0", "target_3d_excess_gt_0"]
FEATURE_SET_NAMES = ["price_only", "news_all_only", "price_news_quality"]
WEIGHT_GRID = np.linspace(0.0, 1.0, 21)


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
    }


def fit_selected_model(train: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]):
    y_train = train[target].astype(int)
    if selected["selected_model"] == "logistic_balanced":
        model = build_logistic_model()
    else:
        model = build_boosting_model(selected["selected_model"], selected["selected_params"], float(y_train.mean()))
    model.fit(train[feature_cols], y_train)
    return model


def tune_and_predict(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    selected = tune_model(train, val, feature_cols, target)
    if selected is None:
        raise ValueError("Could not tune model; insufficient rows or class variation.")
    model = fit_selected_model(train, feature_cols, target, selected)
    return model.predict_proba(val[feature_cols])[:, 1], model.predict_proba(test[feature_cols])[:, 1], selected


def best_weighted_blend(y_val: pd.Series, global_val: np.ndarray, ticker_val: np.ndarray) -> tuple[float, float, float]:
    best_weight = 0.5
    best_threshold_value = 0.5
    best_score = -np.inf
    for weight in WEIGHT_GRID:
        val_proba = weight * ticker_val + (1 - weight) * global_val
        threshold, score = best_threshold(y_val, val_proba)
        if score > best_score:
            best_weight = float(weight)
            best_threshold_value = float(threshold)
            best_score = float(score)
    return best_weight, best_threshold_value, best_score


def ticker_flag_frame(tickers: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({f"ticker_is_{ticker.lower()}": (tickers.reset_index(drop=True) == ticker).astype(int) for ticker in TICKERS})


def fit_meta_model(val_predictions: pd.DataFrame, target: str) -> tuple[Pipeline, float, float]:
    x_val = pd.concat(
        [
            val_predictions[["global_proba", "ticker_proba"]].reset_index(drop=True),
            ticker_flag_frame(val_predictions["ticker"]),
        ],
        axis=1,
    )
    y_val = val_predictions[target].astype(int)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_val, y_val)
    val_proba = model.predict_proba(x_val)[:, 1]
    threshold, score = best_threshold(y_val, val_proba)
    return model, float(threshold), float(score)


def predict_meta(model: Pipeline, test_predictions: pd.DataFrame) -> np.ndarray:
    x_test = pd.concat(
        [
            test_predictions[["global_proba", "ticker_proba"]].reset_index(drop=True),
            ticker_flag_frame(test_predictions["ticker"]),
        ],
        axis=1,
    )
    return model.predict_proba(x_test)[:, 1]


def run_target_feature(df: pd.DataFrame, target: str, feature_set_name: str, feature_cols: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_all, val_all, test_all = split_holdout(df)
    global_val, global_test, global_selected = tune_and_predict(train_all, val_all, test_all, feature_cols, target)

    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = [
        {
            "target": target,
            "feature_set": feature_set_name,
            "model_scope": "global_all_tickers",
            "ticker": "ALL",
            "train_rows": len(train_all),
            "validation_rows": len(val_all),
            "test_rows": len(test_all),
            **global_selected,
        }
    ]

    global_val_map = val_all[["ticker", "trading_date", target]].copy()
    global_val_map["global_proba"] = global_val
    global_test_map = test_all[["ticker", "trading_date", target]].copy()
    global_test_map["global_proba"] = global_test

    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker].copy()
        train_ticker, val_ticker, test_ticker = split_holdout(ticker_df)
        ticker_val, ticker_test, ticker_selected = tune_and_predict(train_ticker, val_ticker, test_ticker, feature_cols, target)
        model_rows.append(
            {
                "target": target,
                "feature_set": feature_set_name,
                "model_scope": "ticker_specific",
                "ticker": ticker,
                "train_rows": len(train_ticker),
                "validation_rows": len(val_ticker),
                "test_rows": len(test_ticker),
                **ticker_selected,
            }
        )

        tv = val_ticker[["ticker", "trading_date", target]].copy()
        tv["ticker_proba"] = ticker_val
        tt = test_ticker[["ticker", "trading_date", target]].copy()
        tt["ticker_proba"] = ticker_test
        val_parts.append(tv)
        test_parts.append(tt)

    val_predictions = pd.concat(val_parts, ignore_index=True).merge(global_val_map, on=["ticker", "trading_date", target], how="inner")
    test_predictions = pd.concat(test_parts, ignore_index=True).merge(global_test_map, on=["ticker", "trading_date", target], how="inner")

    y_val = val_predictions[target].astype(int)
    y_test = test_predictions[target].astype(int)
    weight, weighted_threshold, weighted_val_ba = best_weighted_blend(y_val, val_predictions["global_proba"].to_numpy(), val_predictions["ticker_proba"].to_numpy())
    meta_model, meta_threshold, meta_val_ba = fit_meta_model(val_predictions, target)

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    candidates = [
        ("global_only", test_predictions["global_proba"].to_numpy(), 0.0, 0.5, np.nan),
        ("ticker_only", test_predictions["ticker_proba"].to_numpy(), 1.0, 0.5, np.nan),
        ("simple_average_50_50", 0.5 * test_predictions["ticker_proba"].to_numpy() + 0.5 * test_predictions["global_proba"].to_numpy(), 0.5, 0.5, np.nan),
        (
            "validation_weighted_average",
            weight * test_predictions["ticker_proba"].to_numpy() + (1 - weight) * test_predictions["global_proba"].to_numpy(),
            weight,
            weighted_threshold,
            weighted_val_ba,
        ),
        ("meta_logistic", predict_meta(meta_model, test_predictions), np.nan, meta_threshold, meta_val_ba),
    ]

    for ensemble_method, proba, ticker_weight, threshold, validation_ba in candidates:
        metric_values = metrics(y_test, proba, threshold)
        result_rows.append(
            {
                "target": target,
                "feature_set": feature_set_name,
                "scope": "pooled_all_tickers",
                "ensemble_method": ensemble_method,
                "ticker_weight": ticker_weight,
                "threshold": threshold,
                "validation_blend_balanced_accuracy": validation_ba,
                "ticker_count": len(TICKERS),
                "train_rows_global": len(train_all),
                "validation_rows": len(val_predictions),
                "test_rows": len(test_predictions),
                "test_positive_rate": float(y_test.mean()),
                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                **metric_values,
            }
        )
        pred = (proba >= threshold).astype(int)
        prediction_rows.extend(
            {
                "ticker": row.ticker,
                "trading_date": row.trading_date,
                "target": target,
                "feature_set": feature_set_name,
                "ensemble_method": ensemble_method,
                "y_true": int(getattr(row, target)),
                "global_proba": float(row.global_proba),
                "ticker_proba": float(row.ticker_proba),
                "final_proba": float(final_proba),
                "pred": int(final_pred),
                "ticker_weight": ticker_weight,
                "threshold": threshold,
            }
            for row, final_proba, final_pred in zip(test_predictions.itertuples(index=False), proba, pred)
        )

        for ticker, part in pd.DataFrame({"ticker": test_predictions["ticker"], "y_true": y_test, "proba": proba, "pred": pred}).groupby("ticker"):
            if part["y_true"].nunique() < 2:
                continue
            result_rows.append(
                {
                    "target": target,
                    "feature_set": feature_set_name,
                    "scope": ticker,
                    "ensemble_method": ensemble_method,
                    "ticker_weight": ticker_weight,
                    "threshold": threshold,
                    "validation_blend_balanced_accuracy": validation_ba,
                    "ticker_count": 1,
                    "train_rows_global": len(train_all),
                    "validation_rows": int((val_predictions["ticker"] == ticker).sum()),
                    "test_rows": len(part),
                    "test_positive_rate": float(part["y_true"].mean()),
                    "majority_baseline_accuracy": float(max(part["y_true"].mean(), 1 - part["y_true"].mean())),
                    **metrics(part["y_true"], part["proba"].to_numpy(), threshold),
                }
            )

    return result_rows, model_rows, prediction_rows


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    all_results: list[dict[str, Any]] = []
    all_model_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for feature_set_name in FEATURE_SET_NAMES:
            feature_cols = [col for col in feature_sets[feature_set_name] if col in df.columns]
            result_rows, model_rows, prediction_rows = run_target_feature(df, target, feature_set_name, feature_cols)
            all_results.extend(result_rows)
            all_model_rows.extend(model_rows)
            all_prediction_rows.extend(prediction_rows)

    results = pd.DataFrame(all_results)
    models = pd.DataFrame(all_model_rows)
    predictions = pd.DataFrame(all_prediction_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "tickers": ", ".join(TICKERS),
                "targets": ", ".join(TARGETS),
                "feature_sets": ", ".join(FEATURE_SET_NAMES),
                "design": "Global model and ticker-specific models are trained on 2020-2022. Their 2023 probabilities tune the blend. Final metrics use 2024 only.",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"global_ticker_blend_results{suffix}.csv", index=False)
    models.to_csv(paths.tables_dir / f"global_ticker_blend_base_models{suffix}.csv", index=False)
    predictions.to_csv(paths.tables_dir / f"global_ticker_blend_predictions{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"global_ticker_blend_metadata{suffix}.csv", index=False)
    return {"results": results, "models": models, "predictions": predictions, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global + ticker-specific blend experiment.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    pooled = outputs["results"][outputs["results"]["scope"] == "pooled_all_tickers"].copy()
    print("\nPooled all-ticker metrics")
    print(pooled.sort_values(["target", "feature_set", "roc_auc"], ascending=[True, True, False]).to_string(index=False))
    print("\nBest pooled rows")
    print(pooled.sort_values(["target", "roc_auc", "balanced_accuracy"], ascending=[True, False, False]).groupby("target").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
