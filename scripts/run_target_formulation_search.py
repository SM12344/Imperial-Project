from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    build_logistic_model,
    split_holdout,
    tune_model,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
SCOPES = ["pooled_all_tickers", *TICKERS]
HORIZONS = [1, 2, 3, 5, 10, 20]
FEATURE_SET_NAMES = ["price_base", "price_expanded", "news_expanded", "price_news_base", "price_news_expanded"]
CONFIDENCE_ROWS = [300, 500, 800]


def add_extended_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    g = out.groupby("ticker", group_keys=False)
    for horizon in HORIZONS:
        raw_col = f"fwd_{horizon}d_return_audit"
        spy_col = f"fwd_{horizon}d_spy_return_audit"
        excess_col = f"fwd_{horizon}d_excess_return_audit"
        out[raw_col] = g["adj_close"].shift(-horizon) / out["adj_close"] - 1
        out[spy_col] = g["spy_return_1d"].transform(
            lambda s, h=horizon: (1 + s.shift(-1)).rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1
        )
        out[excess_col] = out[raw_col] - out[spy_col]

        for threshold in [0.0, 0.01, 0.02]:
            suffix = "gt0" if threshold == 0.0 else f"gt{int(threshold * 100)}pct"
            out[f"target_raw_{horizon}d_{suffix}"] = (out[raw_col] > threshold).astype("Int64")
            out[f"target_excess_{horizon}d_{suffix}"] = (out[excess_col] > threshold).astype("Int64")
        out[f"target_raw_abs_{horizon}d_gt2pct"] = (out[raw_col].abs() > 0.02).astype("Int64")
        out[f"target_excess_abs_{horizon}d_gt1pct"] = (out[excess_col].abs() > 0.01).astype("Int64")

        missing = out[raw_col].isna() | out[spy_col].isna() | out[excess_col].isna()
        target_cols = [col for col in out.columns if col.startswith(f"target_raw_{horizon}d_") or col.startswith(f"target_excess_{horizon}d_")]
        target_cols.extend([f"target_raw_abs_{horizon}d_gt2pct", f"target_excess_abs_{horizon}d_gt1pct"])
        for col in target_cols:
            out.loc[missing, col] = pd.NA
    return out


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
        "predicted_positive_rate": float(pred.mean()),
    }


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[np.ndarray, dict[str, Any]] | None:
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
    return model.predict_proba(test[feature_cols])[:, 1], selected


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_extended_targets(add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))))
    feature_sets = build_feature_sets(df)

    result_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    target_names = []
    for horizon in HORIZONS:
        target_names.extend(
            [
                f"target_raw_{horizon}d_gt0",
                f"target_raw_{horizon}d_gt1pct",
                f"target_raw_{horizon}d_gt2pct",
                f"target_excess_{horizon}d_gt0",
                f"target_excess_{horizon}d_gt1pct",
            ]
        )

    for scope in SCOPES:
        scoped = scope_frame(df, scope)
        for target in target_names:
            horizon = int(target.split("_")[2].replace("d", ""))
            usable = scoped.dropna(subset=[target]).copy()
            train, val, test = split_holdout(usable)
            if len(train) < 80 or len(val) < 25 or len(test) < 100:
                continue
            y_test = test[target].astype(int)
            if y_test.nunique() < 2:
                continue
            distribution_rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "target": target,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                    "test_positive_rate": float(y_test.mean()),
                    "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                }
            )
            for feature_set_name in FEATURE_SET_NAMES:
                feature_cols = [col for col in feature_sets[feature_set_name] if col in usable.columns]
                output = fit_predict(train, val, test, feature_cols, target)
                if output is None:
                    continue
                proba, selected = output
                threshold = selected["selected_threshold"]
                base_row = {
                    "scope": scope,
                    "horizon": horizon,
                    "target": target,
                    "feature_set": feature_set_name,
                    "feature_count": len(feature_cols),
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                    "test_positive_rate": float(y_test.mean()),
                    "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                    "selected_model": selected["selected_model"],
                    "selected_params": selected["selected_params"],
                    "selected_threshold": threshold,
                    "validation_roc_auc": selected["validation_roc_auc"],
                    "validation_balanced_accuracy_at_threshold": selected["validation_balanced_accuracy_at_threshold"],
                }
                result_rows.append({**base_row, **metrics(y_test, proba, threshold)})

                if scope == "pooled_all_tickers":
                    confidence = np.abs(proba - 0.5)
                    order = np.argsort(-confidence)
                    for covered_count in CONFIDENCE_ROWS:
                        if len(order) < covered_count:
                            continue
                        idx = order[:covered_count]
                        covered_y = y_test.iloc[idx]
                        if covered_y.nunique() < 2:
                            continue
                        confidence_rows.append(
                            {
                                **base_row,
                                "covered_rows": covered_count,
                                "coverage": covered_count / len(test),
                                "covered_positive_rate": float(covered_y.mean()),
                                "covered_majority_baseline_accuracy": float(max(covered_y.mean(), 1 - covered_y.mean())),
                                "confidence_cutoff": float(confidence[idx].min()),
                                **metrics(covered_y, proba[idx], 0.5),
                            }
                        )

    results = pd.DataFrame(result_rows)
    confidence = pd.DataFrame(confidence_rows)
    distributions = pd.DataFrame(distribution_rows)
    if len(results):
        results["accuracy_minus_baseline"] = results["accuracy"] - results["majority_baseline_accuracy"]
        results["defensible_70_candidate"] = (
            (results["test_rows"] >= 500)
            & (results["accuracy"] >= 0.70)
            & (results["accuracy_minus_baseline"] >= 0.03)
            & (results["balanced_accuracy"] >= 0.55)
        )
    if len(confidence):
        confidence["accuracy_minus_covered_baseline"] = confidence["accuracy"] - confidence["covered_majority_baseline_accuracy"]
        confidence["defensible_70_candidate"] = (
            (confidence["covered_rows"] >= 500)
            & (confidence["accuracy"] >= 0.70)
            & (confidence["accuracy_minus_covered_baseline"] >= 0.03)
            & (confidence["balanced_accuracy"] >= 0.55)
        )

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
                "targets": ", ".join(target_names),
                "defensible_70_rule": "accuracy >= 0.70, test/covered rows >= 500, accuracy at least 0.03 above majority baseline, balanced accuracy >= 0.55",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"target_formulation_search_results{suffix}.csv", index=False)
    confidence.to_csv(paths.tables_dir / f"target_formulation_confidence_results{suffix}.csv", index=False)
    distributions.to_csv(paths.tables_dir / f"target_formulation_distributions{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"target_formulation_search_metadata{suffix}.csv", index=False)
    return {"results": results, "confidence": confidence, "distributions": distributions, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search non-leaky target formulations for large-sample accuracy.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nBest large all-row results")
    results = outputs["results"].copy()
    large = results[results["test_rows"] >= 500]
    print(
        large.sort_values(["accuracy", "accuracy_minus_baseline", "balanced_accuracy"], ascending=[False, False, False])
        .head(40)
        .to_string(index=False)
    )
    print("\nBest confidence-filtered large results")
    confidence = outputs["confidence"].copy()
    large_conf = confidence[confidence["covered_rows"] >= 500]
    print(
        large_conf.sort_values(["accuracy", "accuracy_minus_covered_baseline", "balanced_accuracy"], ascending=[False, False, False])
        .head(40)
        .to_string(index=False)
    )
    print("\nDefensible 0.70 candidates")
    print(results[results.get("defensible_70_candidate", False)].to_string(index=False))
    print(confidence[confidence.get("defensible_70_candidate", False)].to_string(index=False))


if __name__ == "__main__":
    main()
