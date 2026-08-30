from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_modelling_baselines import ALL_FINBERT_FEATURES, PRICE_ONLY_FEATURES, load_dataset


FEATURE_SPECS = {
    "price_only": {
        "numeric": PRICE_ONLY_FEATURES,
        "categorical": [],
        "normalize_by_ticker": False,
    },
    "price_plus_finbert": {
        "numeric": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
        "categorical": [],
        "normalize_by_ticker": False,
    },
    "price_plus_finbert_ticker": {
        "numeric": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
        "categorical": ["ticker"],
        "normalize_by_ticker": False,
    },
    "normalized_price_plus_finbert_ticker": {
        "numeric": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
        "categorical": ["ticker"],
        "normalize_by_ticker": True,
    },
}

MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


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


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric_features),
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(model_name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, random_state=42)
        scale_numeric = True
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        scale_numeric = False
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            l2_regularization=0.1,
            random_state=42,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features, scale_numeric)),
            ("model", model),
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


def build_slices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "pooled_all_days": df.copy(),
        "pooled_news_days": df[df["has_news"] == 1].copy(),
    }


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
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


def build_walk_forward_splits(
    df: pd.DataFrame,
    start_fraction: float = 0.5,
    test_fraction: float = 0.15,
    step_fraction: float = 0.15,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    n_dates = len(unique_dates)
    if n_dates < 30:
        raise ValueError("Not enough unique dates for walk-forward validation.")

    test_size = max(5, int(n_dates * test_fraction))
    step_size = max(5, int(n_dates * step_fraction))
    train_end_idx = max(10, int(n_dates * start_fraction))

    splits: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    fold_idx = 1
    while train_end_idx < n_dates - test_size:
        test_end_idx = min(train_end_idx + test_size, n_dates)
        train_end_date = unique_dates[train_end_idx]
        test_start_date = unique_dates[train_end_idx]
        test_end_date = unique_dates[test_end_idx - 1]

        train_df = df[df["trading_date"] < train_end_date].copy()
        test_df = df[(df["trading_date"] >= test_start_date) & (df["trading_date"] <= test_end_date)].copy()
        if not train_df.empty and not test_df.empty:
            splits.append((train_df, test_df, fold_idx))

        train_end_idx += step_size
        fold_idx += 1

    if not splits:
        raise ValueError("Walk-forward settings produced no valid folds.")
    return splits


def normalize_by_ticker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[numeric_features] = train_out[numeric_features].astype(float)
    test_out[numeric_features] = test_out[numeric_features].astype(float)

    stats = (
        train_df.groupby("ticker")[numeric_features]
        .agg(["mean", "std"])
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    global_mean = train_df[numeric_features].mean()
    global_std = train_df[numeric_features].std().replace(0, 1.0).fillna(1.0)

    for ticker in train_df["ticker"].drop_duplicates():
        ticker_mask_train = train_out["ticker"] == ticker
        ticker_mask_test = test_out["ticker"] == ticker
        if ticker not in stats.index:
            means = global_mean
            stds = global_std
        else:
            means = stats.loc[ticker, ("mean", slice(None))]
            means.index = means.index.get_level_values(1)
            stds = stats.loc[ticker, ("std", slice(None))]
            stds.index = stds.index.get_level_values(1)
            stds = stds.replace(0, 1.0).fillna(1.0)

        train_out.loc[ticker_mask_train, numeric_features] = (
            train_out.loc[ticker_mask_train, numeric_features] - means
        ) / stds
        test_out.loc[ticker_mask_test, numeric_features] = (
            test_out.loc[ticker_mask_test, numeric_features] - means
        ) / stds

    return train_out, test_out


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]
    feature_cols = numeric_features + categorical_features

    train_prepared = train_df.copy()
    test_prepared = test_df.copy()
    if feature_spec["normalize_by_ticker"]:
        train_prepared, test_prepared = normalize_by_ticker(train_prepared, test_prepared, numeric_features)

    return train_prepared, test_prepared, feature_cols


def run_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
) -> dict[str, Any]:
    feature_spec = FEATURE_SPECS[feature_set_name]
    train_prepared, test_prepared, feature_cols = prepare_feature_frames(train_df, test_df, feature_spec)
    pipeline = build_pipeline(model_name, feature_spec["numeric"], feature_spec["categorical"])

    X_train = train_prepared[feature_cols]
    y_train = train_prepared["target_next_day_up"]
    X_test = test_prepared[feature_cols]
    y_test = test_prepared["target_next_day_up"]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)

    return {
        "feature_set": feature_set_name,
        "model_name": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["trading_date"].min().date().isoformat(),
        "train_end": train_df["trading_date"].max().date().isoformat(),
        "test_start": test_df["trading_date"].min().date().isoformat(),
        "test_end": test_df["trading_date"].max().date().isoformat(),
        **metrics,
    }


def run_holdout_suite(
    df: pd.DataFrame,
    slice_name: str,
    test_fraction: float,
) -> list[dict[str, Any]]:
    train_df, test_df, cutoff_date = time_split(df, test_fraction)
    rows: list[dict[str, Any]] = []
    for feature_set_name in FEATURE_SPECS:
        for model_name in MODEL_NAMES:
            row = run_candidate(train_df, test_df, feature_set_name, model_name)
            row["slice_name"] = slice_name
            row["split_type"] = "single_holdout"
            row["cutoff_date"] = cutoff_date.date().isoformat()
            row["test_up_rate"] = float(test_df["target_next_day_up"].mean())
            row["always_up_accuracy"] = float(test_df["target_next_day_up"].mean())
            rows.append(row)
    return rows


def run_walk_forward_suite(df: pd.DataFrame, slice_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = build_walk_forward_splits(df)
    for train_df, test_df, fold_idx in splits:
        for feature_set_name in FEATURE_SPECS:
            for model_name in MODEL_NAMES:
                row = run_candidate(train_df, test_df, feature_set_name, model_name)
                row["slice_name"] = slice_name
                row["split_type"] = "walk_forward"
                row["fold"] = fold_idx
                row["test_up_rate"] = float(test_df["target_next_day_up"].mean())
                row["always_up_accuracy"] = float(test_df["target_next_day_up"].mean())
                rows.append(row)
    return rows


def summarize_walk_forward(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "always_up_accuracy",
        "test_up_rate",
    ]
    summary = (
        results_df.groupby(["slice_name", "feature_set", "model_name"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metric_cols})
    )
    fold_counts = (
        results_df.groupby(["slice_name", "feature_set", "model_name"], as_index=False)["fold"]
        .count()
        .rename(columns={"fold": "num_folds"})
    )
    return summary.merge(fold_counts, on=["slice_name", "feature_set", "model_name"], how="left")


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    slices = build_slices(df)

    holdout_rows: list[dict[str, Any]] = []
    walk_forward_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for slice_name, slice_df in slices.items():
        try:
            holdout_rows.extend(run_holdout_suite(slice_df, slice_name, test_fraction))
        except ValueError as exc:
            skipped_rows.append(
                {
                    "slice_name": slice_name,
                    "stage": "single_holdout",
                    "rows": len(slice_df),
                    "unique_dates": int(slice_df["trading_date"].nunique()),
                    "reason": str(exc),
                }
            )

        try:
            walk_forward_rows.extend(run_walk_forward_suite(slice_df, slice_name))
        except ValueError as exc:
            skipped_rows.append(
                {
                    "slice_name": slice_name,
                    "stage": "walk_forward",
                    "rows": len(slice_df),
                    "unique_dates": int(slice_df["trading_date"].nunique()),
                    "reason": str(exc),
                }
            )

    holdout_df = pd.DataFrame(holdout_rows).sort_values(
        ["slice_name", "roc_auc", "balanced_accuracy", "f1", "accuracy"],
        ascending=[True, False, False, False, False],
    )
    walk_forward_df = pd.DataFrame(walk_forward_rows).sort_values(
        ["slice_name", "fold", "roc_auc"], ascending=[True, True, False]
    )
    walk_forward_summary_df = summarize_walk_forward(walk_forward_df)
    skipped_df = pd.DataFrame(skipped_rows).sort_values(["slice_name", "stage"]) if skipped_rows else pd.DataFrame()
    metadata_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "slices": ", ".join(slices.keys()),
                "feature_sets": ", ".join(FEATURE_SPECS.keys()),
                "models": ", ".join(MODEL_NAMES),
                "single_holdout_test_fraction": test_fraction,
                "walk_forward_start_fraction": 0.5,
                "walk_forward_test_fraction": 0.15,
                "walk_forward_step_fraction": 0.15,
            }
        ]
    )

    holdout_df.to_csv(paths.tables_dir / "general_pooled_holdout_results.csv", index=False)
    walk_forward_df.to_csv(paths.tables_dir / "general_pooled_walkforward_results.csv", index=False)
    walk_forward_summary_df.to_csv(paths.tables_dir / "general_pooled_walkforward_summary.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "general_pooled_metadata.csv", index=False)
    skipped_df.to_csv(paths.tables_dir / "general_pooled_skipped.csv", index=False)

    return {
        "holdout_df": holdout_df,
        "walk_forward_df": walk_forward_df,
        "walk_forward_summary_df": walk_forward_summary_df,
        "metadata_df": metadata_df,
        "skipped_df": skipped_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run general pooled-model improvements with ticker awareness, normalization, and walk-forward validation."
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
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique trading dates reserved for the single holdout test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        test_fraction=args.test_fraction,
    )
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["holdout_df"].to_string(index=False))
    print()
    print(outputs["walk_forward_summary_df"].to_string(index=False))
    if not outputs["skipped_df"].empty:
        print()
        print(outputs["skipped_df"].to_string(index=False))


if __name__ == "__main__":
    main()
