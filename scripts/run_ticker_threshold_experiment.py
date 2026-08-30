from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_modelling_baselines import FEATURE_SETS, load_dataset


DEFAULT_TICKERS = ["TSLA", "AMZN", "MSFT"]
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class ProjectPaths:
    root: Path
    processed_dir: Path
    tables_dir: Path


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def build_paths(project_root: str | None = None) -> ProjectPaths:
    root = resolve_project_root(project_root)
    processed_dir = root / "data" / "processed"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, processed_dir=processed_dir, tables_dir=tables_dir)


def build_logistic_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, feature_cols)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, feature_cols)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def time_split_three_way(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    if len(unique_dates) < 15:
        raise ValueError("Not enough unique dates for train/validation/test splitting.")

    train_end_idx = max(1, int(len(unique_dates) * train_fraction))
    val_end_idx = max(train_end_idx + 1, int(len(unique_dates) * (train_fraction + val_fraction)))
    train_end_idx = min(train_end_idx, len(unique_dates) - 2)
    val_end_idx = min(val_end_idx, len(unique_dates) - 1)

    val_start_date = unique_dates[train_end_idx]
    test_start_date = unique_dates[val_end_idx]

    train_df = df[df["trading_date"] < val_start_date].copy()
    val_df = df[(df["trading_date"] >= val_start_date) & (df["trading_date"] < test_start_date)].copy()
    test_df = df[df["trading_date"] >= test_start_date].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Three-way split produced an empty partition.")
    return train_df, val_df, test_df, val_start_date, test_start_date


def choose_threshold(y_true: pd.Series, y_proba: pd.Series, thresholds: list[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def build_model(model_name: str, feature_cols: list[str]) -> Pipeline:
    if model_name == "logistic_regression_balanced":
        return build_logistic_pipeline(feature_cols)
    if model_name == "random_forest_balanced":
        return build_random_forest_pipeline(feature_cols)
    raise ValueError(f"Unsupported model: {model_name}")


def run_slice_experiment(
    df: pd.DataFrame,
    slice_name: str,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    train_df, val_df, test_df, val_start_date, test_start_date = time_split_three_way(df)
    rows: list[dict[str, Any]] = []

    for feature_set_name, feature_cols in FEATURE_SETS.items():
        for model_name in ["logistic_regression_balanced", "random_forest_balanced"]:
            pipeline = build_model(model_name, feature_cols)
            pipeline.fit(train_df[feature_cols], train_df["target_next_day_up"])

            val_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
            chosen_threshold, val_balanced_accuracy = choose_threshold(
                val_df["target_next_day_up"], val_proba, thresholds
            )

            test_proba = pipeline.predict_proba(test_df[feature_cols])[:, 1]
            test_pred = (test_proba >= chosen_threshold).astype(int)
            metrics = evaluate_predictions(test_df["target_next_day_up"], test_pred, test_proba)

            rows.append(
                {
                    "slice_name": slice_name,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "threshold": chosen_threshold,
                    "validation_balanced_accuracy": val_balanced_accuracy,
                    "train_rows": len(train_df),
                    "validation_rows": len(val_df),
                    "test_rows": len(test_df),
                    "train_start": train_df["trading_date"].min().date().isoformat(),
                    "train_end": train_df["trading_date"].max().date().isoformat(),
                    "validation_start": val_start_date.date().isoformat(),
                    "validation_end": val_df["trading_date"].max().date().isoformat(),
                    "test_start": test_start_date.date().isoformat(),
                    "test_end": test_df["trading_date"].max().date().isoformat(),
                    "test_up_rate": float(test_df["target_next_day_up"].mean()),
                    "always_up_accuracy": float(test_df["target_next_day_up"].mean()),
                    **metrics,
                }
            )
    return rows


def build_slices(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    slices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].copy()
        slices[f"{ticker}_all_days"] = ticker_df
        slices[f"{ticker}_news_days"] = ticker_df[ticker_df["has_news"] == 1].copy()
    return slices


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    tickers: list[str] | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    selected_tickers = tickers or DEFAULT_TICKERS
    threshold_grid = thresholds or DEFAULT_THRESHOLDS

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for slice_name, slice_df in build_slices(df, selected_tickers).items():
        try:
            results.extend(run_slice_experiment(slice_df, slice_name, threshold_grid))
        except ValueError as exc:
            skipped.append(
                {
                    "slice_name": slice_name,
                    "rows": len(slice_df),
                    "unique_dates": int(slice_df["trading_date"].nunique()),
                    "reason": str(exc),
                }
            )

    results_df = pd.DataFrame(results).sort_values(
        ["slice_name", "roc_auc", "balanced_accuracy", "f1"],
        ascending=[True, False, False, False],
    )
    metadata_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "tickers": ", ".join(selected_tickers),
                "thresholds": ", ".join(str(x) for x in threshold_grid),
                "split_type": "time_based_train_validation_test",
            }
        ]
    )
    skipped_df = pd.DataFrame(skipped).sort_values("slice_name") if skipped else pd.DataFrame()

    results_df.to_csv(paths.tables_dir / "ticker_threshold_results.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "ticker_threshold_metadata.csv", index=False)
    skipped_df.to_csv(paths.tables_dir / "ticker_threshold_skipped.csv", index=False)

    return {
        "results_df": results_df,
        "metadata_df": metadata_df,
        "skipped_df": skipped_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-ticker threshold-tuned balanced models for ticker improvement analysis."
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
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers to evaluate individually.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Probability thresholds to evaluate on the validation slice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        tickers=args.tickers,
        thresholds=args.thresholds,
    )
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))
    if not outputs["skipped_df"].empty:
        print()
        print(outputs["skipped_df"].to_string(index=False))


if __name__ == "__main__":
    main()
