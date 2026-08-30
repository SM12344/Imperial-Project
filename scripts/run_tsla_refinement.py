from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from run_modelling_baselines import (
    FINBERT_FEATURES,
    PRICE_ONLY_FEATURES,
    build_logistic_pipeline,
    build_paths,
    build_random_forest_pipeline,
    evaluate_predictions,
    load_dataset,
)


TSLA_FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "finbert_only": FINBERT_FEATURES,
    "price_plus_finbert": PRICE_ONLY_FEATURES + FINBERT_FEATURES,
}


def build_tsla_news_days_slice(df: pd.DataFrame) -> pd.DataFrame:
    tsla = df[(df["ticker"] == "TSLA") & (df["has_news"] == 1)].copy()
    if tsla.empty:
        raise ValueError("No TSLA news-day rows found.")
    return tsla.sort_values("trading_date").reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_idx]
    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train_df, test_df, cutoff_date


def build_random_forest_variant(feature_cols: list[str], variant: str) -> Pipeline:
    pipeline = build_random_forest_pipeline(feature_cols)
    model = pipeline.named_steps["model"]
    if variant == "rf_shallow":
        model.set_params(n_estimators=300, max_depth=3, min_samples_leaf=5)
    elif variant == "rf_deeper":
        model.set_params(n_estimators=500, max_depth=None, min_samples_leaf=3)
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return pipeline


def run_one(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
) -> dict[str, Any]:
    feature_cols = TSLA_FEATURE_SETS[feature_set_name]
    if model_name == "logistic_regression":
        pipeline = build_logistic_pipeline(feature_cols)
    elif model_name in {"rf_shallow", "rf_deeper"}:
        pipeline = build_random_forest_variant(feature_cols, model_name)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    X_train = train_df[feature_cols]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_next_day_up"]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)
    always_up_accuracy = float(y_test.mean())

    return {
        "feature_set": feature_set_name,
        "model_name": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["trading_date"].min().date().isoformat(),
        "train_end": train_df["trading_date"].max().date().isoformat(),
        "test_start": test_df["trading_date"].min().date().isoformat(),
        "test_end": test_df["trading_date"].max().date().isoformat(),
        "always_up_accuracy": always_up_accuracy,
        **metrics,
    }


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    tsla = build_tsla_news_days_slice(df)
    train_df, test_df, cutoff_date = time_split(tsla, test_fraction)

    rows = []
    for feature_set_name in TSLA_FEATURE_SETS:
        for model_name in ["logistic_regression", "rf_shallow", "rf_deeper"]:
            row = run_one(train_df, test_df, feature_set_name, model_name)
            rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        ["roc_auc", "f1", "accuracy"], ascending=[False, False, False]
    )
    split_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cutoff_date": cutoff_date.date().isoformat(),
                "rows": len(tsla),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "unique_dates": tsla["trading_date"].nunique(),
                "test_fraction": test_fraction,
                "test_up_rate": float(test_df["target_next_day_up"].mean()),
            }
        ]
    )

    results_df.to_csv(paths.tables_dir / "tsla_refinement_results.csv", index=False)
    split_df.to_csv(paths.tables_dir / "tsla_refinement_split_metadata.csv", index=False)
    return {"results_df": results_df, "split_df": split_df}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TSLA refinement comparisons.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.test_fraction)
    print(outputs["split_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
