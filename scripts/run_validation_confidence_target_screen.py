from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from run_2024_holdout_ablation import TEST_END, TEST_START, TRAIN_END, VALIDATION_START, add_holdout_filter_columns, split_holdout
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_target_formulation_search import add_extended_targets
from run_target_formulation_ultra_screen import FEATURE_SET_NAMES, add_context, target_specs, tune_fit_predict, models, best_threshold


VALIDATION_COVERED_ROWS = [300, 500, 800]


def metric_row(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
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


def fit_selected(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[Any, dict[str, Any]] | None:
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    best: dict[str, Any] | None = None
    for model_name, params, model in models(float(y_train.mean())):
        model.fit(train[feature_cols], y_train)
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        threshold, val_ba = best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "selected_threshold": threshold,
            "validation_roc_auc": float(val_auc),
            "validation_balanced_accuracy": float(val_ba),
        }
        if best is None or (row["validation_balanced_accuracy"], row["validation_roc_auc"]) > (
            best["validation_balanced_accuracy"],
            best["validation_roc_auc"],
        ):
            best = row
    if best is None:
        return None
    for model_name, params, model in models(float(y_train.mean())):
        if model_name == best["selected_model"] and params == best["selected_params"]:
            model.fit(train[feature_cols], y_train)
            return model, best
    return None


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_extended_targets(add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))))
    df, context_cols = add_context(df)
    feature_sets = build_feature_sets(df)
    feature_sets["price_base_context"] = feature_sets["price_base"] + context_cols
    feature_sets["price_expanded_context"] = feature_sets["price_expanded"] + context_cols
    feature_sets["news_expanded_context"] = feature_sets["news_expanded"] + context_cols
    feature_sets["price_news_expanded_context"] = feature_sets["price_news_expanded"] + context_cols

    rows: list[dict[str, Any]] = []
    for spec in target_specs():
        target = spec["target"]
        usable = df.dropna(subset=[target]).copy()
        train, val, test = split_holdout(usable)
        if len(train) < 200 or len(val) < 800 or len(test) < 500:
            continue
        y_val = val[target].astype(int)
        y_test = test[target].astype(int)
        if y_val.nunique() < 2 or y_test.nunique() < 2:
            continue
        for feature_set_name in FEATURE_SET_NAMES:
            feature_cols = feature_sets[feature_set_name]
            selected_output = fit_selected(train, val, feature_cols, target)
            if selected_output is None:
                continue
            model, selected = selected_output
            val_proba = model.predict_proba(val[feature_cols])[:, 1]

            train_full = pd.concat([train, val], ignore_index=True)
            y_train_full = train_full[target].astype(int)
            final_model = None
            for model_name, params, candidate in models(float(y_train_full.mean())):
                if model_name == selected["selected_model"] and params == selected["selected_params"]:
                    final_model = candidate
                    break
            if final_model is None:
                continue
            final_model.fit(train_full[feature_cols], y_train_full)
            test_proba = final_model.predict_proba(test[feature_cols])[:, 1]

            val_confidence = np.abs(val_proba - 0.5)
            test_confidence = np.abs(test_proba - 0.5)
            order = np.argsort(-val_confidence)
            for validation_rows in VALIDATION_COVERED_ROWS:
                if len(order) < validation_rows:
                    continue
                val_idx = order[:validation_rows]
                confidence_cutoff = float(val_confidence[val_idx].min())
                test_mask = test_confidence >= confidence_cutoff
                if int(test_mask.sum()) < 300:
                    continue
                covered_val = y_val.iloc[val_idx]
                covered_test = y_test[test_mask]
                test_metrics = metric_row(covered_test, test_proba[test_mask], 0.5)
                covered_baseline = float(max(covered_test.mean(), 1 - covered_test.mean()))
                row = {
                    **spec,
                    "feature_set": feature_set_name,
                    "feature_count": len(feature_cols),
                    "selected_model": selected["selected_model"],
                    "selected_params": selected["selected_params"],
                    "validation_selected_rows": validation_rows,
                    "validation_accuracy": accuracy_score(covered_val, (val_proba[val_idx] >= 0.5).astype(int)),
                    "validation_balanced_accuracy": balanced_accuracy_score(covered_val, (val_proba[val_idx] >= 0.5).astype(int)),
                    "validation_positive_rate": float(covered_val.mean()),
                    "validation_majority_baseline": float(max(covered_val.mean(), 1 - covered_val.mean())),
                    "confidence_cutoff": confidence_cutoff,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "test_rows_total": len(test),
                    "test_covered_rows": int(test_mask.sum()),
                    "test_coverage": float(test_mask.mean()),
                    "test_positive_rate": float(covered_test.mean()),
                    "test_majority_baseline": covered_baseline,
                    **test_metrics,
                }
                row["accuracy_minus_test_baseline"] = row["accuracy"] - row["test_majority_baseline"]
                row["defensible_70_candidate"] = (
                    row["test_covered_rows"] >= 500
                    and row["accuracy"] >= 0.70
                    and row["accuracy_minus_test_baseline"] >= 0.03
                    and row["balanced_accuracy"] >= 0.55
                )
                rows.append(row)

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
                "design": "Confidence cutoff selected only on 2023 validation rows, then applied unchanged to 2024 test rows.",
            }
        ]
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"validation_confidence_target_results{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"validation_confidence_target_metadata{suffix}.csv", index=False)
    return {"results": results, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-selected confidence screen for target formulations.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print(outputs["results"].sort_values(["accuracy", "accuracy_minus_test_baseline", "balanced_accuracy"], ascending=[False, False, False]).head(40).to_string(index=False))
    print("\nDefensible 0.70 candidates")
    print(outputs["results"][outputs["results"]["defensible_70_candidate"]].to_string(index=False))


if __name__ == "__main__":
    main()
