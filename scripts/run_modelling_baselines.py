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


PRICE_ONLY_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volume_change_1d",
    "moving_avg_5d",
    "moving_avg_20d",
    "ma_5_20_gap",
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_5d",
]

FINBERT_FEATURES = [
    "news_count",
    "log_news_count",
    "news_count_above_1",
    "news_count_above_2",
    "has_news",
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score_mean",
    "finbert_positive_std",
    "finbert_negative_std",
    "finbert_neutral_std",
    "finbert_sentiment_score_std",
    "finbert_positive_max",
    "finbert_negative_max",
    "finbert_neutral_max",
    "finbert_sentiment_score_max",
    "finbert_sentiment_score_min",
    "news_count_lag1",
    "news_count_rolling3",
    "news_count_rolling5",
    "finbert_positive_mean_lag1",
    "finbert_negative_mean_lag1",
    "finbert_neutral_mean_lag1",
    "finbert_positive_mean_rolling5",
    "finbert_negative_mean_rolling5",
    "finbert_neutral_mean_rolling5",
    "finbert_sentiment_score_lag1",
    "finbert_sentiment_score_rolling3",
    "finbert_sentiment_score_rolling5",
    "finbert_sentiment_score_surprise",
    "finbert_positive_mean_surprise",
    "finbert_negative_mean_surprise",
    "sentiment_x_has_news",
    "sentiment_x_volatility_5d",
    "sentiment_x_abs_return_1d",
]

MARKET_FINBERT_FEATURES = [
    "has_market_news",
    "market_news_count",
    "market_log_news_count",
    "market_news_count_above_1",
    "market_news_count_above_2",
    "market_finbert_positive_mean",
    "market_finbert_negative_mean",
    "market_finbert_neutral_mean",
    "market_finbert_sentiment_score_mean",
    "market_finbert_positive_std",
    "market_finbert_negative_std",
    "market_finbert_neutral_std",
    "market_finbert_sentiment_score_std",
    "market_finbert_positive_max",
    "market_finbert_negative_max",
    "market_finbert_neutral_max",
    "market_finbert_sentiment_score_max",
    "market_finbert_sentiment_score_min",
    "market_finbert_sentiment_score_lag1",
    "market_finbert_sentiment_score_rolling3",
    "market_finbert_sentiment_score_rolling5",
    "market_news_count_lag1",
    "market_news_count_rolling5",
    "market_sentiment_x_spy_return_1d",
]

ALL_FINBERT_FEATURES = FINBERT_FEATURES + MARKET_FINBERT_FEATURES

FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "price_plus_finbert": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
}


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


def load_dataset(paths: ProjectPaths, dataset_name: str) -> pd.DataFrame:
    dataset_path = paths.processed_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run notebook 03 first to create the FinBERT modelling dataset."
        )

    df = pd.read_csv(dataset_path)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df["target_next_day_up"] = df["target_next_day_up"].astype(int)
    for col in MARKET_FINBERT_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
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
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
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
                    n_estimators=400,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
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


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_set_name]
    X_train = train_df[feature_cols]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_next_day_up"]

    if model_name == "logistic_regression":
        pipeline = build_logistic_pipeline(feature_cols)
    elif model_name == "random_forest":
        pipeline = build_random_forest_pipeline(feature_cols)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, pred, proba)
    result_row = {
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

    predictions = test_df[["ticker", "trading_date", "target_next_day_up"]].copy()
    predictions["feature_set"] = feature_set_name
    predictions["model_name"] = model_name
    predictions["predicted_up"] = pred
    predictions["predicted_probability_up"] = proba
    predictions["correct_prediction"] = (predictions["predicted_up"] == predictions["target_next_day_up"]).astype(int)
    return result_row, predictions


def build_split_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "split_type": "time_based",
                "cutoff_date": cutoff_date.date().isoformat(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_unique_dates": train_df["trading_date"].nunique(),
                "test_unique_dates": test_df["trading_date"].nunique(),
            }
        ]
    )


def build_feature_manifest() -> pd.DataFrame:
    rows = []
    for feature_set_name, cols in FEATURE_SETS.items():
        for col in cols:
            rows.append({"feature_set": feature_set_name, "feature_name": col})
    return pd.DataFrame(rows)


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(paths, dataset_name)
    train_df, test_df, cutoff_date = time_split(df, test_fraction=test_fraction)

    results = []
    prediction_frames = []
    for feature_set_name in FEATURE_SETS:
        for model_name in ["logistic_regression", "random_forest"]:
            result_row, predictions = run_experiment(train_df, test_df, feature_set_name, model_name)
            results.append(result_row)
            prediction_frames.append(predictions)

    results_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "f1", "accuracy"], ascending=[False, False, False]
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    split_df = build_split_metadata(train_df, test_df, cutoff_date)
    feature_manifest_df = build_feature_manifest()

    results_df.to_csv(paths.tables_dir / "modelling_baseline_results.csv", index=False)
    predictions_df.to_csv(paths.processed_dir / "modelling_baseline_predictions.csv", index=False)
    split_df.to_csv(paths.tables_dir / "modelling_split_metadata.csv", index=False)
    feature_manifest_df.to_csv(paths.tables_dir / "modelling_feature_manifest.csv", index=False)

    return {
        "paths": paths,
        "dataset": df,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "predictions_df": predictions_df,
        "split_df": split_df,
        "feature_manifest_df": feature_manifest_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run price-only and price-plus-FinBERT modelling baselines."
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
        help="Fraction of unique trading dates reserved for the time-ordered test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        test_fraction=args.test_fraction,
    )
    print(outputs["split_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
