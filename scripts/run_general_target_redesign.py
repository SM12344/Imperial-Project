from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from run_general_pooled_improvements import (
    FEATURE_SPECS,
    MODEL_NAMES,
    build_paths,
    build_pipeline,
    build_walk_forward_splits,
    evaluate_predictions,
    prepare_feature_frames,
    time_split,
)
from run_modelling_baselines import load_dataset


TARGET_DEFINITIONS = {
    "next_day_up": ("target_next_day_up", None),
    "next_day_excess_gt_0": ("next_day_excess_return", 0.0),
    "next_day_excess_gt_0_5pct": ("next_day_excess_return", 0.005),
}


def build_pooled_news_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    out["next_day_stock_return"] = out.groupby("ticker")["return_1d"].shift(-1)
    out["next_day_spy_return"] = out.groupby("ticker")["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    out = out[(out["has_news"] == 1) & out["next_day_stock_return"].notna() & out["next_day_spy_return"].notna()].copy()
    return out.reset_index(drop=True)


def assign_target(df: pd.DataFrame, source_col: str, threshold: float | None) -> pd.Series:
    if source_col == "target_next_day_up":
        return df[source_col].astype(int)
    if threshold is None:
        raise ValueError(f"Threshold required for target source {source_col}")
    return (df[source_col] > threshold).astype(int)


def run_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    target_name: str,
) -> dict[str, Any]:
    source_col, threshold = TARGET_DEFINITIONS[target_name]
    feature_spec = FEATURE_SPECS[feature_set_name]
    train_prepared, test_prepared, feature_cols = prepare_feature_frames(train_df, test_df, feature_spec)
    pipeline = build_pipeline(model_name, feature_spec["numeric"], feature_spec["categorical"])

    y_train = assign_target(train_prepared, source_col, threshold)
    y_test = assign_target(test_prepared, source_col, threshold)
    pipeline.fit(train_prepared[feature_cols], y_train)
    pred = pipeline.predict(test_prepared[feature_cols])
    proba = pipeline.predict_proba(test_prepared[feature_cols])[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)

    return {
        "target_name": target_name,
        "target_source": source_col,
        "target_threshold": threshold if threshold is not None else "existing",
        "feature_set": feature_set_name,
        "model_name": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["trading_date"].min().date().isoformat(),
        "train_end": train_df["trading_date"].max().date().isoformat(),
        "test_start": test_df["trading_date"].min().date().isoformat(),
        "test_end": test_df["trading_date"].max().date().isoformat(),
        "test_positive_rate": float(y_test.mean()),
        "always_positive_accuracy": float(y_test.mean()),
        **metrics,
    }


def run_holdout(df: pd.DataFrame, test_fraction: float) -> list[dict[str, Any]]:
    train_df, test_df, cutoff_date = time_split(df, test_fraction)
    rows: list[dict[str, Any]] = []
    for target_name in TARGET_DEFINITIONS:
        for feature_set_name in FEATURE_SPECS:
            for model_name in MODEL_NAMES:
                row = run_candidate(train_df, test_df, feature_set_name, model_name, target_name)
                row["split_type"] = "single_holdout"
                row["cutoff_date"] = cutoff_date.date().isoformat()
                rows.append(row)
    return rows


def run_walkforward(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_df, test_df, fold in build_walk_forward_splits(df):
        for target_name in TARGET_DEFINITIONS:
            for feature_set_name in FEATURE_SPECS:
                for model_name in MODEL_NAMES:
                    row = run_candidate(train_df, test_df, feature_set_name, model_name, target_name)
                    row["split_type"] = "walk_forward"
                    row["fold"] = fold
                    rows.append(row)
    return rows


def summarize_walkforward(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "always_positive_accuracy",
        "test_positive_rate",
    ]
    summary = (
        results_df.groupby(["target_name", "feature_set", "model_name"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metric_cols})
    )
    fold_counts = (
        results_df.groupby(["target_name", "feature_set", "model_name"], as_index=False)["fold"]
        .count()
        .rename(columns={"fold": "num_folds"})
    )
    return summary.merge(fold_counts, on=["target_name", "feature_set", "model_name"], how="left")


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fraction: float = 0.2,
    output_suffix: str = "",
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    pooled_news = build_pooled_news_dataset(df)

    holdout_df = pd.DataFrame(run_holdout(pooled_news, test_fraction)).sort_values(
        ["target_name", "roc_auc", "balanced_accuracy", "f1", "accuracy"],
        ascending=[True, False, False, False, False],
    )
    walkforward_df = pd.DataFrame(run_walkforward(pooled_news)).sort_values(
        ["target_name", "fold", "roc_auc"], ascending=[True, True, False]
    )
    walkforward_summary_df = summarize_walkforward(walkforward_df)
    metadata_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "rows": len(pooled_news),
                "unique_dates": pooled_news["trading_date"].nunique(),
                "test_fraction": test_fraction,
                "target_names": ", ".join(TARGET_DEFINITIONS.keys()),
                "feature_sets": ", ".join(FEATURE_SPECS.keys()),
                "models": ", ".join(MODEL_NAMES),
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    holdout_df.to_csv(paths.tables_dir / f"general_target_redesign_holdout_results{suffix}.csv", index=False)
    walkforward_df.to_csv(paths.tables_dir / f"general_target_redesign_walkforward_results{suffix}.csv", index=False)
    walkforward_summary_df.to_csv(paths.tables_dir / f"general_target_redesign_walkforward_summary{suffix}.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / f"general_target_redesign_metadata{suffix}.csv", index=False)

    return {
        "holdout_df": holdout_df,
        "walkforward_df": walkforward_df,
        "walkforward_summary_df": walkforward_summary_df,
        "metadata_df": metadata_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pooled news-days target redesign experiments using market-relative labels."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.test_fraction, args.output_suffix)
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["holdout_df"].to_string(index=False))
    print()
    print(outputs["walkforward_summary_df"].to_string(index=False))


if __name__ == "__main__":
    main()
