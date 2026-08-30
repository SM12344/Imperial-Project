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


SCENARIOS = [
    {"scenario": "pooled_all_3d_abs_move", "scope": "pooled_all_tickers", "target": "target_3d_abs_excess_gt_1pct"},
    {"scenario": "pooled_all_5d_direction", "scope": "pooled_all_tickers", "target": "target_5d_excess_gt_0"},
    {"scenario": "aapl_all_5d_direction", "scope": "AAPL", "target": "target_5d_excess_gt_0"},
    {"scenario": "amzn_all_5d_direction", "scope": "AMZN", "target": "target_5d_excess_gt_0"},
    {"scenario": "msft_all_5d_direction", "scope": "MSFT", "target": "target_5d_excess_gt_0"},
    {"scenario": "nvda_all_5d_direction", "scope": "NVDA", "target": "target_5d_excess_gt_0"},
    {"scenario": "tsla_all_5d_direction", "scope": "TSLA", "target": "target_5d_excess_gt_0"},
]

FEATURE_SET_NAMES = ["price_only", "price_quality", "price_news_quality"]
AUGMENTATION_RATIOS = [None, 0.50, 0.75, 1.00, 1.25]
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


def resample_minority_to_ratio(df: pd.DataFrame, target: str, ratio: float | None, random_state: int = 42) -> pd.DataFrame:
    if ratio is None:
        return df.copy()
    counts = df[target].astype(int).value_counts()
    if len(counts) < 2:
        return df.copy()
    majority_label = int(counts.idxmax())
    minority_label = int(counts.idxmin())
    majority = df[df[target].astype(int) == majority_label]
    minority = df[df[target].astype(int) == minority_label]
    target_minority_count = int(round(len(majority) * ratio))
    target_minority_count = max(target_minority_count, len(minority))
    rng = np.random.default_rng(random_state)
    sampled_idx = rng.choice(minority.index.to_numpy(), size=target_minority_count, replace=target_minority_count > len(minority))
    out = pd.concat([majority, df.loc[sampled_idx]], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


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


def scenario_frame(df: pd.DataFrame, scenario: dict[str, str]) -> pd.DataFrame:
    if scenario["scope"] == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scenario["scope"]].copy()


def tune_one(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    augmentation_ratio: float | None,
) -> dict[str, Any] | None:
    y_val = val[target].astype(int)
    if len(train) < 80 or len(val) < 25 or train[target].nunique() < 2 or y_val.nunique() < 2:
        return None
    augmented = resample_minority_to_ratio(train, target, augmentation_ratio)
    y_aug = augmented[target].astype(int)
    best: dict[str, Any] | None = None
    for model_name in MODEL_NAMES:
        for params in model_grid(model_name):
            model = build_candidate_model(model_name, params, float(y_aug.mean()))
            model.fit(augmented[feature_cols], y_aug)
            val_proba = model.predict_proba(val[feature_cols])[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            threshold, val_balanced = best_threshold(y_val, val_proba)
            row = {
                "selected_model": model_name,
                "selected_params": params,
                "selected_threshold": threshold,
                "validation_roc_auc": float(val_auc),
                "validation_balanced_accuracy_at_threshold": val_balanced,
                "validation_accuracy_at_threshold": accuracy_score(y_val, (val_proba >= threshold).astype(int)),
                "tuning_train_rows_after_augmentation": len(augmented),
                "tuning_train_positive_rate_after_augmentation": float(y_aug.mean()),
            }
            if best is None or (row["validation_roc_auc"], row["validation_balanced_accuracy_at_threshold"]) > (
                best["validation_roc_auc"],
                best["validation_balanced_accuracy_at_threshold"],
            ):
                best = row
    return best


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        sdf = scenario_frame(df, scenario)
        train, val, test = split_holdout(sdf)
        target = scenario["target"]
        coverage_rows.append(
            {
                **scenario,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "train_positive_rate": float(train[target].mean()) if len(train) else np.nan,
                "validation_positive_rate": float(val[target].mean()) if len(val) else np.nan,
                "test_positive_rate": float(test[target].mean()) if len(test) else np.nan,
                "test_majority_baseline_accuracy": float(max(test[target].mean(), 1 - test[target].mean())) if len(test) else np.nan,
            }
        )
        y_test = test[target].astype(int)
        if len(test) < 25 or y_test.nunique() < 2:
            continue
        for feature_set_name in FEATURE_SET_NAMES:
            feature_cols = [col for col in feature_sets[feature_set_name] if col in sdf.columns]
            if not feature_cols:
                continue
            for augmentation_ratio in AUGMENTATION_RATIOS:
                selected = tune_one(train, val, feature_cols, target, augmentation_ratio)
                if selected is None:
                    continue
                train_full = pd.concat([train, val], ignore_index=True)
                final_train = resample_minority_to_ratio(train_full, target, augmentation_ratio)
                y_final = final_train[target].astype(int)
                model = build_candidate_model(selected["selected_model"], selected["selected_params"], float(y_final.mean()))
                model.fit(final_train[feature_cols], y_final)
                test_proba = model.predict_proba(test[feature_cols])[:, 1]
                for threshold_mode, threshold in [("validation_tuned", selected["selected_threshold"]), ("default_0.5", 0.5)]:
                    rows.append(
                        {
                            **scenario,
                            "feature_set": feature_set_name,
                            "feature_count": len(feature_cols),
                            "augmentation_ratio_minority_to_majority": "none" if augmentation_ratio is None else augmentation_ratio,
                            "final_train_rows_after_augmentation": len(final_train),
                            "final_train_positive_rate_after_augmentation": float(y_final.mean()),
                            "threshold_mode": threshold_mode,
                            "threshold": threshold,
                            "train_rows": len(train),
                            "validation_rows": len(val),
                            "test_rows": len(test),
                            "test_positive_rate": float(y_test.mean()),
                            "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                            **selected,
                            **metrics(y_test, test_proba, threshold),
                        }
                    )

    results = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    best = (
        results.sort_values(["scenario", "roc_auc", "balanced_accuracy"], ascending=[True, False, False])
        .groupby("scenario", as_index=False)
        .head(10)
    )
    ratio_summary = (
        results[results["threshold_mode"] == "validation_tuned"]
        .groupby(["augmentation_ratio_minority_to_majority"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_accuracy=("accuracy", "mean"),
            rows=("scenario", "count"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": "2020-01-01 to 2022-12-31",
                "validation_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                "models_grid_searched": ", ".join(MODEL_NAMES),
                "feature_sets": ", ".join(FEATURE_SET_NAMES),
                "augmentation_ratios": ", ".join("none" if ratio is None else str(ratio) for ratio in AUGMENTATION_RATIOS),
                "augmentation_method": "training-only random oversampling of the minority class to a target minority/majority ratio",
                "test_policy": "2024 is never resampled and is not used for model, parameter, or threshold selection",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"full_data_augmentation_grid_results{suffix}.csv", index=False)
    best.to_csv(paths.tables_dir / f"full_data_augmentation_grid_best{suffix}.csv", index=False)
    ratio_summary.to_csv(paths.tables_dir / f"full_data_augmentation_grid_ratio_summary{suffix}.csv", index=False)
    coverage.to_csv(paths.tables_dir / f"full_data_augmentation_grid_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"full_data_augmentation_grid_metadata{suffix}.csv", index=False)
    return {"results": results, "best": best, "ratio_summary": ratio_summary, "coverage": coverage, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-data augmentation proportion and model grid-search experiments.")
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
    print("\nCoverage")
    print(outputs["coverage"].to_string(index=False))
    print("\nBest rows")
    print(outputs["best"].head(80).to_string(index=False))
    print("\nRatio summary")
    print(outputs["ratio_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
