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
    add_holdout_filter_columns,
    best_threshold,
    build_ablation_feature_sets,
    build_logistic_model,
    metric_row,
    split_holdout,
    tune_model,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model as build_boosting_model


TARGET = "target_5d_excess_gt_0"
SCOPES = ["pooled_all_tickers", "AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
FILTERS = ["all_days", "top_10pct_news_volume", "strong_abs_sentiment"]
FEATURE_SETS = ["price_only", "news_all_only", "price_news_quality"]
CONFIDENCE_CUTOFFS = [0.55, 0.60, 0.65, 0.70]


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def fit_selected_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]) -> Any:
    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[target].astype(int)
    if selected["selected_model"] == "logistic_balanced":
        model = build_logistic_model()
    else:
        model = build_boosting_model(selected["selected_model"], selected["selected_params"], float(y_train_full.mean()))
    model.fit(train_full[feature_cols], y_train_full)
    return model


def safe_metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if y_true.nunique() > 1 else np.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() > 1 else np.nan,
    }
    return row


def confidence_slice(y_true: pd.Series, proba: np.ndarray, cutoff: float) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    low = 1.0 - cutoff
    mask = (proba <= low) | (proba >= cutoff)
    sliced_y = y_true.loc[mask]
    sliced_proba = proba[mask]
    sliced_pred = (sliced_proba >= cutoff).astype(int)
    return sliced_y, sliced_pred, sliced_proba


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    all_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    best_confidence_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for scope in SCOPES:
        scoped = scope_frame(df, scope)
        for filter_name in FILTERS:
            filtered = scoped[scoped[filter_name] == 1].copy()
            train, val, test = split_holdout(filtered)
            if len(test) == 0:
                skipped_rows.append({"scope": scope, "filter": filter_name, "reason": "no_test_rows"})
                continue
            y_test = test[TARGET].astype(int)
            if len(train) < 80 or len(val) < 25 or len(test) < 25 or y_test.nunique() < 2:
                skipped_rows.append(
                    {
                        "scope": scope,
                        "filter": filter_name,
                        "reason": "insufficient_rows_or_single_test_class",
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "test_rows": len(test),
                        "test_positive_rate": y_test.mean() if len(test) else np.nan,
                    }
                )
                continue

            for feature_set_name in FEATURE_SETS:
                feature_cols = [col for col in feature_sets[feature_set_name] if col in filtered.columns]
                selected = tune_model(train, val, feature_cols, TARGET)
                if selected is None:
                    skipped_rows.append({"scope": scope, "filter": filter_name, "feature_set": feature_set_name, "reason": "tuning_failed"})
                    continue

                model = fit_selected_model(train, val, feature_cols, TARGET, selected)
                test_proba = model.predict_proba(test[feature_cols])[:, 1]
                tuned_threshold = selected["selected_threshold"]
                tuned_pred = (test_proba >= tuned_threshold).astype(int)
                baseline = float(max(y_test.mean(), 1 - y_test.mean()))

                all_rows.append(
                    {
                        "scope": scope,
                        "filter": filter_name,
                        "target": TARGET,
                        "feature_set": feature_set_name,
                        "evaluation": "all_test_rows_validation_threshold",
                        "feature_count": len(feature_cols),
                        "threshold": tuned_threshold,
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "test_rows": len(test),
                        "coverage": 1.0,
                        "covered_rows": len(test),
                        "test_positive_rate": float(y_test.mean()),
                        "covered_positive_rate": float(y_test.mean()),
                        "majority_baseline_accuracy": baseline,
                        "covered_majority_baseline_accuracy": baseline,
                        **selected,
                        **metric_row(y_test, tuned_pred, test_proba),
                    }
                )

                for cutoff in CONFIDENCE_CUTOFFS:
                    sliced_y, sliced_pred, sliced_proba = confidence_slice(y_test.reset_index(drop=True), test_proba, cutoff)
                    covered_rows = len(sliced_y)
                    coverage = covered_rows / len(test)
                    if covered_rows < 20:
                        continue
                    covered_baseline = float(max(sliced_y.mean(), 1 - sliced_y.mean()))
                    row = {
                        "scope": scope,
                        "filter": filter_name,
                        "target": TARGET,
                        "feature_set": feature_set_name,
                        "evaluation": "confidence_filtered",
                        "confidence_cutoff": cutoff,
                        "low_probability_cutoff": 1.0 - cutoff,
                        "high_probability_cutoff": cutoff,
                        "feature_count": len(feature_cols),
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "test_rows": len(test),
                        "covered_rows": covered_rows,
                        "coverage": coverage,
                        "test_positive_rate": float(y_test.mean()),
                        "covered_positive_rate": float(sliced_y.mean()),
                        "majority_baseline_accuracy": baseline,
                        "covered_majority_baseline_accuracy": covered_baseline,
                        **selected,
                        **safe_metric_row(sliced_y, sliced_pred, sliced_proba),
                    }
                    confidence_rows.append(row)

    confidence = pd.DataFrame(confidence_rows)
    all_results = pd.DataFrame(all_rows)
    skipped = pd.DataFrame(skipped_rows)
    if len(confidence):
        confidence["accuracy_lift_over_covered_baseline"] = confidence["accuracy"] - confidence["covered_majority_baseline_accuracy"]
        confidence["balanced_accuracy_lift_over_0_5"] = confidence["balanced_accuracy"] - 0.5
        best_confidence = (
            confidence.sort_values(
                ["scope", "filter", "feature_set", "accuracy_lift_over_covered_baseline", "balanced_accuracy", "coverage"],
                ascending=[True, True, True, False, False, False],
            )
            .groupby(["scope", "filter", "feature_set"], as_index=False)
            .head(1)
        )
        headline = confidence[
            (confidence["coverage"] >= 0.20)
            & (confidence["balanced_accuracy"].fillna(0) >= 0.55)
            & (confidence["accuracy_lift_over_covered_baseline"] >= 0)
        ].sort_values(["accuracy", "balanced_accuracy", "coverage"], ascending=False)
    else:
        best_confidence = pd.DataFrame(best_confidence_rows)
        headline = pd.DataFrame()

    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "target": TARGET,
                "train_period": "2020-01-01 to 2022-12-31",
                "validation_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                "confidence_rule": "evaluate only rows where p <= 1-cutoff or p >= cutoff",
                "confidence_cutoffs": ", ".join(str(x) for x in CONFIDENCE_CUTOFFS),
                "scopes": ", ".join(SCOPES),
                "filters": ", ".join(FILTERS),
                "feature_sets": ", ".join(FEATURE_SETS),
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    all_results.to_csv(paths.tables_dir / f"confidence_filtered_all_rows{suffix}.csv", index=False)
    confidence.to_csv(paths.tables_dir / f"confidence_filtered_results{suffix}.csv", index=False)
    best_confidence.to_csv(paths.tables_dir / f"confidence_filtered_best{suffix}.csv", index=False)
    headline.to_csv(paths.tables_dir / f"confidence_filtered_headline_candidates{suffix}.csv", index=False)
    skipped.to_csv(paths.tables_dir / f"confidence_filtered_skipped{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"confidence_filtered_metadata{suffix}.csv", index=False)
    return {
        "all_results": all_results,
        "confidence": confidence,
        "best_confidence": best_confidence,
        "headline": headline,
        "skipped": skipped,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate confidence-filtered 2024 holdout predictions.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nAll rows")
    print(outputs["all_results"].sort_values(["scope", "filter", "feature_set"]).to_string(index=False))
    print("\nBest confidence-filtered rows")
    print(outputs["best_confidence"].sort_values(["scope", "filter", "feature_set"]).to_string(index=False))
    print("\nHeadline candidates")
    print(outputs["headline"].head(40).to_string(index=False))
    if len(outputs["skipped"]):
        print("\nSkipped")
        print(outputs["skipped"].to_string(index=False))


if __name__ == "__main__":
    main()
