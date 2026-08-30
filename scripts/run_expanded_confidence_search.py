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
from run_expanded_feature_search import HORIZONS, add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model


MIN_COVERED_ROWS = [300, 500, 800]
FEATURE_SET_NAMES = ["price_base", "price_expanded", "news_base", "news_expanded", "price_news_base", "price_news_expanded", "price_quality_expanded"]


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


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news))))
    feature_sets = build_feature_sets(df)
    train, val, test = split_holdout(df)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        target = f"target_{horizon}d_excess_gt_0"
        y_test = test[target].astype(int)
        for feature_set_name in FEATURE_SET_NAMES:
            feature_cols = [col for col in feature_sets[feature_set_name] if col in df.columns]
            output = fit_predict(train, val, test, feature_cols, target)
            if output is None:
                continue
            proba, selected = output
            confidence = np.abs(proba - 0.5)
            order = np.argsort(-confidence)
            for min_rows in MIN_COVERED_ROWS:
                if len(order) < min_rows:
                    continue
                idx = order[:min_rows]
                covered_y = y_test.iloc[idx]
                covered_proba = proba[idx]
                threshold = 0.5
                rows.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "feature_set": feature_set_name,
                        "min_covered_rows": min_rows,
                        "covered_rows": len(idx),
                        "coverage": len(idx) / len(test),
                        "test_rows": len(test),
                        "test_positive_rate": float(y_test.mean()),
                        "covered_positive_rate": float(covered_y.mean()),
                        "covered_majority_baseline": float(max(covered_y.mean(), 1 - covered_y.mean())),
                        "confidence_cutoff": float(confidence[idx].min()),
                        "selected_model": selected["selected_model"],
                        "selected_params": selected["selected_params"],
                        "selected_threshold": selected["selected_threshold"],
                        **metrics(covered_y, covered_proba, threshold),
                    }
                )

    results = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "coverage_rule": "Pooled all-ticker 2024 predictions only; keep top-N most confident predictions with N in 300, 500, 800.",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"expanded_confidence_search_results{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"expanded_confidence_search_metadata{suffix}.csv", index=False)
    return {"results": results, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-coverage confidence search with expanded features.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print(outputs["results"].sort_values(["min_covered_rows", "accuracy", "balanced_accuracy"], ascending=[True, False, False]).groupby("min_covered_rows").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
