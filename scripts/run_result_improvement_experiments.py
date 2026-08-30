from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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

from run_modelling_baselines import FINBERT_FEATURES, PRICE_ONLY_FEATURES, load_dataset


BASE_SENTIMENT_COLS = [
    "news_count",
    "has_news",
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score_mean",
    "finbert_sentiment_score_surprise",
]


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
    return ProjectPaths(
        root=root,
        processed_dir=root / "data" / "processed",
        tables_dir=root / "outputs" / "tables",
    )


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    out = out.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    out["next_day_stock_return"] = out.groupby("ticker")["return_1d"].shift(-1)
    out["next_day_spy_return"] = out.groupby("ticker")["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    out["target_next_day_excess_gt_0"] = (out["next_day_excess_return"] > 0).astype(int)

    for lag in range(0, 4):
        for col in BASE_SENTIMENT_COLS:
            out[f"{col}_causal_lag{lag}"] = out.groupby("ticker")[col].shift(lag)

    lag_score_cols = [f"finbert_sentiment_score_mean_causal_lag{lag}" for lag in range(0, 4)]
    out["sentiment_lag_mean_0_3"] = out[lag_score_cols].mean(axis=1)
    out["sentiment_lag_max_0_3"] = out[lag_score_cols].max(axis=1)
    out["sentiment_lag_min_0_3"] = out[lag_score_cols].min(axis=1)
    out["news_count_lag_sum_0_3"] = out[[f"news_count_causal_lag{lag}" for lag in range(0, 4)]].sum(axis=1)
    out["has_recent_news_0_3"] = (out[[f"has_news_causal_lag{lag}" for lag in range(0, 4)]].sum(axis=1) > 0).astype(int)
    return out.dropna(subset=["next_day_stock_return", "next_day_spy_return"]).reset_index(drop=True)


def build_walk_forward_splits(
    df: pd.DataFrame,
    start_fraction: float = 0.5,
    test_fraction: float = 0.15,
    step_fraction: float = 0.15,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    unique_dates = sorted(df["trading_date"].drop_duplicates())
    n_dates = len(unique_dates)
    test_size = max(5, int(n_dates * test_fraction))
    step_size = max(5, int(n_dates * step_fraction))
    train_end_idx = max(10, int(n_dates * start_fraction))
    splits: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    fold = 1
    while train_end_idx < n_dates - test_size:
        train_end_date = unique_dates[train_end_idx]
        test_end_idx = min(train_end_idx + test_size, n_dates)
        test_end_date = unique_dates[test_end_idx - 1]
        train_df = df[df["trading_date"] < train_end_date].copy()
        test_df = df[(df["trading_date"] >= train_end_date) & (df["trading_date"] <= test_end_date)].copy()
        if not train_df.empty and not test_df.empty:
            splits.append((train_df, test_df, fold))
        train_end_idx += step_size
        fold += 1
    return splits


def feature_sets() -> dict[str, dict[str, list[str]]]:
    causal_lag_cols = [
        f"{col}_causal_lag{lag}"
        for lag in range(0, 4)
        for col in BASE_SENTIMENT_COLS
    ]
    causal_summary_cols = [
        "sentiment_lag_mean_0_3",
        "sentiment_lag_max_0_3",
        "sentiment_lag_min_0_3",
        "news_count_lag_sum_0_3",
        "has_recent_news_0_3",
    ]
    return {
        "price_only": {"numeric": PRICE_ONLY_FEATURES, "categorical": []},
        "price_plus_current_finbert": {"numeric": PRICE_ONLY_FEATURES + FINBERT_FEATURES, "categorical": []},
        "price_plus_current_finbert_ticker": {
            "numeric": PRICE_ONLY_FEATURES + FINBERT_FEATURES,
            "categorical": ["ticker"],
        },
        "price_plus_causal_sentiment_lags_ticker": {
            "numeric": PRICE_ONLY_FEATURES + causal_lag_cols + causal_summary_cols,
            "categorical": ["ticker"],
        },
    }


def build_model(model_name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    scale_numeric = model_name in {"logistic_balanced", "logistic"}
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric),
    ]
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))

    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=3000, random_state=42)
    elif model_name == "logistic_balanced":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif model_name == "random_forest_balanced":
        estimator = RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "extra_trees_balanced":
        estimator = ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient":
        estimator = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.035,
            l2_regularization=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("preprocessor", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])


def split_train_validation(train_df: pd.DataFrame, validation_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(train_df["trading_date"].drop_duplicates())
    split_idx = min(max(1, int(len(unique_dates) * (1 - validation_fraction))), len(unique_dates) - 1)
    cutoff = unique_dates[split_idx]
    subtrain = train_df[train_df["trading_date"] < cutoff].copy()
    validation = train_df[train_df["trading_date"] >= cutoff].copy()
    return subtrain, validation


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def choose_threshold(y_true: pd.Series, proba: np.ndarray, objective: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.2, 0.8, 61):
        pred = (proba >= threshold).astype(int)
        if objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        elif objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def run_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    fold: int,
    target_col: str = "target_next_day_excess_gt_0",
) -> list[dict[str, Any]]:
    spec = feature_sets()[feature_set_name]
    feature_cols = spec["numeric"] + spec["categorical"]
    train_df = train_df.dropna(subset=[target_col]).copy()
    test_df = test_df.dropna(subset=[target_col]).copy()
    y_train = train_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return []

    model = build_model(model_name, spec["numeric"], spec["categorical"])
    model.fit(train_df[feature_cols], y_train)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]

    rows: list[dict[str, Any]] = []
    for threshold_name, threshold in [("default_0_5", 0.5)]:
        pred = (proba >= threshold).astype(int)
        rows.append(
            {
                "fold": fold,
                "feature_set": feature_set_name,
                "model_name": model_name,
                "threshold_strategy": threshold_name,
                "threshold": threshold,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_positive_rate": float(y_test.mean()),
                "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                **classification_metrics(y_test, pred, proba),
            }
        )

    subtrain_df, validation_df = split_train_validation(train_df)
    y_subtrain = subtrain_df[target_col].astype(int)
    y_validation = validation_df[target_col].astype(int)
    if y_subtrain.nunique() == 2 and y_validation.nunique() == 2:
        threshold_model = build_model(model_name, spec["numeric"], spec["categorical"])
        threshold_model.fit(subtrain_df[feature_cols], y_subtrain)
        validation_proba = threshold_model.predict_proba(validation_df[feature_cols])[:, 1]
        for objective in ["balanced_accuracy", "f1"]:
            tuned_threshold, validation_score = choose_threshold(y_validation, validation_proba, objective)
            pred = (proba >= tuned_threshold).astype(int)
            rows.append(
                {
                    "fold": fold,
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "threshold_strategy": f"train_tuned_{objective}",
                    "threshold": tuned_threshold,
                    "validation_objective_score": validation_score,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "test_positive_rate": float(y_test.mean()),
                    "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                    **classification_metrics(y_test, pred, proba),
                }
            )
    return rows


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    raw_df = load_dataset(paths, dataset_name)
    df = prepare_dataset(raw_df)
    news_or_recent_news = df[(df["has_news"] == 1) | (df["has_recent_news_0_3"] == 1)].copy()

    model_names = [
        "logistic",
        "logistic_balanced",
        "random_forest_balanced",
        "extra_trees_balanced",
        "hist_gradient",
    ]
    rows: list[dict[str, Any]] = []
    for slice_name, slice_df in {"all_days": df, "news_or_recent_news": news_or_recent_news}.items():
        for train_df, test_df, fold in build_walk_forward_splits(slice_df):
            for feature_set_name in feature_sets():
                for model_name in model_names:
                    candidate_rows = run_candidate(train_df, test_df, feature_set_name, model_name, fold)
                    for row in candidate_rows:
                        row["slice_name"] = slice_name
                    rows.extend(candidate_rows)

    results = pd.DataFrame(rows)
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "majority_baseline_accuracy",
        "test_positive_rate",
    ]
    summary = (
        results.groupby(["slice_name", "feature_set", "model_name", "threshold_strategy"], as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metric_cols})
    )
    fold_counts = (
        results.groupby(["slice_name", "feature_set", "model_name", "threshold_strategy"], as_index=False)["fold"]
        .count()
        .rename(columns={"fold": "num_folds"})
    )
    summary = summary.merge(fold_counts, on=["slice_name", "feature_set", "model_name", "threshold_strategy"])
    summary = summary.sort_values(["mean_roc_auc", "mean_balanced_accuracy"], ascending=False)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "rows": len(df),
                "target": "target_next_day_excess_gt_0",
                "description": "Causal feature and threshold-tuning experiments for result improvement",
            }
        ]
    )

    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths.tables_dir / "result_improvement_walkforward_results.csv", index=False)
    summary.to_csv(paths.tables_dir / "result_improvement_walkforward_summary.csv", index=False)
    metadata.to_csv(paths.tables_dir / "result_improvement_metadata.csv", index=False)
    return {"results": results, "summary": summary, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal result-improvement experiments.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name)
    print(outputs["metadata"].to_string(index=False))
    print()
    print(outputs["summary"].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
