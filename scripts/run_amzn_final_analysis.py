from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    build_ablation_feature_sets,
    split_holdout,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_tuned_boosting_experiments import build_model


TARGET = "target_5d_excess_gt_0"
TICKER = "AMZN"
BEST_FEATURE_SET = "price_quality"
BEST_MODEL_NAME = "lightgbm"
BEST_PARAMS = {
    "n_estimators": 80,
    "max_depth": 1,
    "num_leaves": 3,
    "learning_rate": 0.03,
    "min_child_samples": 25,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
}
BEST_THRESHOLD = 0.425


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "positive_rate": float(y_true.mean()),
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
    }


def model_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "model_importance"])
    return pd.DataFrame({"feature": feature_cols, "model_importance": values}).sort_values("model_importance", ascending=False)


def segment_rows(test: pd.DataFrame, y_true: pd.Series, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred = (proba >= threshold).astype(int)
    segment_masks = {
        "all_2024": pd.Series(True, index=test.index),
        "high_news_volume_top_quartile": test["news_count"] >= test["news_count"].quantile(0.75),
        "low_news_volume_bottom_quartile": test["news_count"] <= test["news_count"].quantile(0.25),
        "strong_abs_sentiment_top_quartile": test["abs_sentiment_score"] >= test["abs_sentiment_score"].quantile(0.75),
        "positive_stock_sentiment": test["finbert_sentiment_score_mean"] > 0,
        "negative_stock_sentiment": test["finbert_sentiment_score_mean"] < 0,
        "market_news_available": test["market_news_count"] > 0,
        "earnings_news_day": test.get("earnings_count", pd.Series(0, index=test.index)) > 0,
        "analyst_news_day": test.get("analyst_count", pd.Series(0, index=test.index)) > 0,
    }
    for segment, mask in segment_masks.items():
        idx = mask.fillna(False).to_numpy()
        if idx.sum() < 15 or y_true.iloc[idx].nunique() < 2:
            continue
        rows.append(
            {
                "segment": segment,
                "rows": int(idx.sum()),
                **metrics(y_true.iloc[idx], proba[idx], threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def monthly_rows(test: pd.DataFrame, y_true: pd.Series, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = test[["trading_date"]].copy()
    frame["month"] = frame["trading_date"].dt.to_period("M").astype(str)
    frame["y_true"] = y_true.to_numpy()
    frame["proba"] = proba
    frame["pred"] = (proba >= threshold).astype(int)
    rows = []
    for month, part in frame.groupby("month", sort=True):
        if part["y_true"].nunique() < 2:
            roc = np.nan
        else:
            roc = roc_auc_score(part["y_true"], part["proba"])
        rows.append(
            {
                "month": month,
                "rows": len(part),
                "accuracy": accuracy_score(part["y_true"], part["pred"]),
                "balanced_accuracy": balanced_accuracy_score(part["y_true"], part["pred"]),
                "roc_auc": roc,
                "positive_rate": part["y_true"].mean(),
                "mean_predicted_probability": part["proba"].mean(),
            }
        )
    return pd.DataFrame(rows)


def resampled_train(train_full: pd.DataFrame, target: str, random_state: int = 42) -> pd.DataFrame:
    counts = train_full[target].value_counts()
    if len(counts) < 2:
        return train_full.copy()
    target_count = counts.max()
    pieces = []
    rng = np.random.default_rng(random_state)
    for label, count in counts.items():
        part = train_full[train_full[target] == label]
        replace = count < target_count
        indices = rng.choice(part.index.to_numpy(), size=target_count, replace=replace)
        pieces.append(train_full.loc[indices])
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    df = df[df["ticker"] == TICKER].copy()
    train, val, test = split_holdout(df)
    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    feature_sets = build_ablation_feature_sets(df)
    feature_cols = feature_sets[BEST_FEATURE_SET]
    model = build_model(BEST_MODEL_NAME, BEST_PARAMS, float(y_train_full.mean()))
    model.fit(train_full[feature_cols], y_train_full)
    proba = model.predict_proba(test[feature_cols])[:, 1]

    summary = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "ticker": TICKER,
                "target": TARGET,
                "feature_set": BEST_FEATURE_SET,
                "model": BEST_MODEL_NAME,
                "threshold": BEST_THRESHOLD,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "train_period": "2020-01-01 to 2022-12-31",
                "validation_period": "2023-01-01 to 2023-12-31",
                "test_period": "2024-01-01 to 2024-12-31",
                **metrics(y_test, proba, BEST_THRESHOLD),
            }
        ]
    )

    importances = model_feature_importance(model, feature_cols)
    permutation = permutation_importance(
        model,
        test[feature_cols],
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_importance_mean_roc_auc": permutation.importances_mean,
            "permutation_importance_std_roc_auc": permutation.importances_std,
        }
    ).sort_values("permutation_importance_mean_roc_auc", ascending=False)
    feature_importance = permutation_df.merge(importances, on="feature", how="left")

    segments = segment_rows(test, y_test, proba, BEST_THRESHOLD)
    monthly = monthly_rows(test, y_test, proba, BEST_THRESHOLD)

    augmentation_rows = []
    for experiment_name, train_variant in [
        ("no_resampling_selected_model", train_full),
        ("training_only_random_oversampling", resampled_train(train_full, TARGET)),
    ]:
        y_variant = train_variant[TARGET].astype(int)
        variant_model = build_model(BEST_MODEL_NAME, BEST_PARAMS, float(y_variant.mean()))
        variant_model.fit(train_variant[feature_cols], y_variant)
        variant_proba = variant_model.predict_proba(test[feature_cols])[:, 1]
        augmentation_rows.append(
            {
                "experiment": experiment_name,
                "train_rows": len(train_variant),
                "train_positive_rate": float(y_variant.mean()),
                **metrics(y_test, variant_proba, BEST_THRESHOLD),
            }
        )
    augmentation = pd.DataFrame(augmentation_rows)

    suffix = f"_{output_suffix}" if output_suffix else ""
    summary.to_csv(paths.tables_dir / f"amzn_final_summary{suffix}.csv", index=False)
    feature_importance.to_csv(paths.tables_dir / f"amzn_final_feature_importance{suffix}.csv", index=False)
    segments.to_csv(paths.tables_dir / f"amzn_final_segment_performance{suffix}.csv", index=False)
    monthly.to_csv(paths.tables_dir / f"amzn_final_monthly_error_analysis{suffix}.csv", index=False)
    augmentation.to_csv(paths.tables_dir / f"amzn_final_training_resampling_check{suffix}.csv", index=False)
    return {
        "summary": summary,
        "feature_importance": feature_importance,
        "segments": segments,
        "monthly": monthly,
        "augmentation": augmentation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final AMZN holdout interpretation analysis.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print("Summary")
    print(outputs["summary"].to_string(index=False))
    print("\nTop feature importance")
    print(outputs["feature_importance"].head(20).to_string(index=False))
    print("\nSegments")
    print(outputs["segments"].to_string(index=False))
    print("\nTraining resampling check")
    print(outputs["augmentation"].to_string(index=False))


if __name__ == "__main__":
    main()
