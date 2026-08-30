from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_modelling_baselines import (
    FEATURE_SETS,
    build_paths,
    build_random_forest_pipeline,
    evaluate_predictions,
    load_dataset,
)


DEFAULT_TEST_FRACTIONS = [0.15, 0.20, 0.25]


def build_tsla_news_days_slice(df: pd.DataFrame) -> pd.DataFrame:
    tsla = df[(df["ticker"] == "TSLA") & (df["has_news"] == 1)].copy()
    if tsla.empty:
        raise ValueError("No TSLA news-day rows found in the modelling dataset.")
    return tsla.sort_values("trading_date").reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for a stable time-based split.")

    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_idx]

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train_df, test_df, cutoff_date


def compute_naive_benchmarks(y_test: pd.Series) -> dict[str, float]:
    up_rate = float(y_test.mean())
    return {
        "always_up_accuracy": up_rate,
        "always_down_accuracy": 1.0 - up_rate,
    }


def extract_feature_importance_table(pipeline, feature_set_name: str) -> pd.DataFrame:
    feature_cols = FEATURE_SETS[feature_set_name]
    model = pipeline.named_steps["model"]
    importances = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "feature_name": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importances


def run_random_forest_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    split_label: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set_name]
    pipeline = build_random_forest_pipeline(feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_next_day_up"]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)
    naive = compute_naive_benchmarks(y_test)

    result_row = {
        "split_label": split_label,
        "feature_set": feature_set_name,
        "model_name": "random_forest",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["trading_date"].min().date().isoformat(),
        "train_end": train_df["trading_date"].max().date().isoformat(),
        "test_start": test_df["trading_date"].min().date().isoformat(),
        "test_end": test_df["trading_date"].max().date().isoformat(),
        **metrics,
        **naive,
    }

    predictions = test_df[
        [
            "ticker",
            "trading_date",
            "target_next_day_up",
            "has_news",
            "finbert_sentiment_score_mean",
        ]
    ].copy()
    predictions["split_label"] = split_label
    predictions["feature_set"] = feature_set_name
    predictions["model_name"] = "random_forest"
    predictions["predicted_up"] = pred
    predictions["predicted_probability_up"] = proba
    predictions["correct_prediction"] = (
        predictions["predicted_up"] == predictions["target_next_day_up"]
    ).astype(int)

    feature_importance_df = extract_feature_importance_table(pipeline, feature_set_name)
    feature_importance_df["split_label"] = split_label
    return result_row, predictions, feature_importance_df


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fractions: list[float] | None = None,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    tsla_news_df = build_tsla_news_days_slice(df)
    fractions = test_fractions or DEFAULT_TEST_FRACTIONS

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []

    for fraction in fractions:
        train_df, test_df, cutoff_date = time_split(tsla_news_df, test_fraction=fraction)
        split_label = f"test_fraction_{fraction:.2f}"
        split_rows.append(
            {
                "split_label": split_label,
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cutoff_date": cutoff_date.date().isoformat(),
                "rows": len(tsla_news_df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "unique_dates": tsla_news_df["trading_date"].nunique(),
                "train_unique_dates": train_df["trading_date"].nunique(),
                "test_unique_dates": test_df["trading_date"].nunique(),
                "up_rate_full_slice": float(tsla_news_df["target_next_day_up"].mean()),
                "up_rate_test_slice": float(test_df["target_next_day_up"].mean()),
            }
        )

        for feature_set_name in FEATURE_SETS:
            result_row, pred_df, importance_df = run_random_forest_experiment(
                train_df=train_df,
                test_df=test_df,
                feature_set_name=feature_set_name,
                split_label=split_label,
            )
            results.append(result_row)
            predictions.append(pred_df)
            importances.append(importance_df)

    results_df = pd.DataFrame(results).sort_values(
        ["split_label", "roc_auc", "f1", "accuracy"], ascending=[True, False, False, False]
    )
    predictions_df = pd.concat(predictions, ignore_index=True)
    feature_importance_df = pd.concat(importances, ignore_index=True).sort_values(
        ["split_label", "feature_set", "importance"], ascending=[True, True, False]
    )
    split_df = pd.DataFrame(split_rows).sort_values("split_label")

    results_df.to_csv(paths.tables_dir / "tsla_focus_results.csv", index=False)
    predictions_df.to_csv(paths.processed_dir / "tsla_focus_predictions.csv", index=False)
    feature_importance_df.to_csv(paths.tables_dir / "tsla_feature_importance.csv", index=False)
    split_df.to_csv(paths.tables_dir / "tsla_focus_split_metadata.csv", index=False)

    return {
        "dataset": tsla_news_df,
        "results_df": results_df,
        "predictions_df": predictions_df,
        "feature_importance_df": feature_importance_df,
        "split_df": split_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TSLA news-days focused random-forest robustness analysis."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project folder containing data/processed and outputs/tables. Defaults to the current directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default="model_dataset_finbert_complete.csv",
        help="Input dataset filename under data/processed.",
    )
    parser.add_argument(
        "--test-fractions",
        nargs="*",
        type=float,
        default=DEFAULT_TEST_FRACTIONS,
        help="One or more time-based holdout fractions to test, such as 0.15 0.20 0.25.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        test_fractions=args.test_fractions,
    )
    print(outputs["split_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
