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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTICLE_FINBERT_FEATURES = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "finbert_sentiment_score",
    "after_market_close",
    "shifted_to_future_trading_day",
]
CATEGORICAL_FEATURES = ["ticker", "finbert_predicted_label"]
MODEL_NAMES = ["logistic_regression", "random_forest"]


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


def load_inputs(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = paths.processed_dir / "news_target_tickers_finbert_scored.csv"
    model_path = paths.processed_dir / "model_dataset_finbert_complete.csv"
    missing = [str(path) for path in [news_path, model_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    news = pd.read_csv(news_path)
    model_df = pd.read_csv(model_path)
    news["aligned_trading_date"] = pd.to_datetime(news["aligned_trading_date"])
    model_df["trading_date"] = pd.to_datetime(model_df["trading_date"])
    model_df["target_next_day_up"] = model_df["target_next_day_up"].astype(int)
    return news, model_df


def build_article_transfer_dataset(news: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["ticker", "trading_date", "target_next_day_up"]
    article_cols = [
        "article_id",
        "ticker",
        "published_utc",
        "text",
        "aligned_trading_date",
        "after_market_close",
        "shifted_to_future_trading_day",
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "finbert_sentiment_score",
        "finbert_predicted_label",
    ]
    article_df = news[article_cols].dropna(subset=["aligned_trading_date"]).copy()
    labelled = article_df.merge(
        model_df[target_cols],
        left_on=["ticker", "aligned_trading_date"],
        right_on=["ticker", "trading_date"],
        how="inner",
        validate="many_to_one",
    )
    labelled = labelled.sort_values(["aligned_trading_date", "ticker", "article_id"]).reset_index(drop=True)
    return labelled


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = sorted(df["aligned_trading_date"].drop_duplicates())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique aligned trading dates for a stable time split.")

    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    cutoff_date = unique_dates[split_idx]
    train_df = df[df["aligned_trading_date"] < cutoff_date].copy()
    test_df = df[df["aligned_trading_date"] >= cutoff_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train_df, test_df, cutoff_date


def build_pipeline(model_name: str) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, ARTICLE_FINBERT_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


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
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = ARTICLE_FINBERT_FEATURES + CATEGORICAL_FEATURES
    pipeline = build_pipeline(model_name)
    pipeline.fit(train_df[feature_cols], train_df["target_next_day_up"])
    pred = pipeline.predict(test_df[feature_cols])
    proba = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    metrics = evaluate_predictions(test_df["target_next_day_up"], pred, proba)

    result_row = {
        "feature_source": "frozen_finbert_article_scores",
        "model_name": model_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["aligned_trading_date"].min().date().isoformat(),
        "train_end": train_df["aligned_trading_date"].max().date().isoformat(),
        "test_start": test_df["aligned_trading_date"].min().date().isoformat(),
        "test_end": test_df["aligned_trading_date"].max().date().isoformat(),
        **metrics,
    }

    predictions = test_df[
        ["article_id", "ticker", "aligned_trading_date", "target_next_day_up", "finbert_sentiment_score"]
    ].copy()
    predictions["model_name"] = model_name
    predictions["predicted_up"] = pred
    predictions["predicted_probability_up"] = proba
    predictions["correct_prediction"] = (predictions["predicted_up"] == predictions["target_next_day_up"]).astype(int)
    return result_row, predictions


def run_pipeline(
    project_root: str | None = None,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    news, model_df = load_inputs(paths)
    article_dataset = build_article_transfer_dataset(news, model_df)
    train_df, test_df, cutoff_date = time_split(article_dataset, test_fraction=test_fraction)

    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        result_row, predictions = run_experiment(train_df, test_df, model_name)
        results.append(result_row)
        prediction_frames.append(predictions)

    results_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "balanced_accuracy", "f1", "accuracy"],
        ascending=[False, False, False, False],
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metadata_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "split_type": "time_based_on_aligned_trading_date",
                "cutoff_date": cutoff_date.date().isoformat(),
                "rows": len(article_dataset),
                "unique_articles": article_dataset["article_id"].nunique(),
                "unique_ticker_days": article_dataset[["ticker", "aligned_trading_date"]].drop_duplicates().shape[0],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "target_up_rate": float(article_dataset["target_next_day_up"].mean()),
            }
        ]
    )

    article_dataset.to_csv(paths.processed_dir / "finbert_transfer_article_dataset.csv", index=False)
    predictions_df.to_csv(paths.processed_dir / "finbert_transfer_probe_predictions.csv", index=False)
    results_df.to_csv(paths.tables_dir / "finbert_transfer_probe_results.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "finbert_transfer_probe_metadata.csv", index=False)

    return {
        "article_dataset": article_dataset,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "predictions_df": predictions_df,
        "metadata_df": metadata_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe frozen FinBERT article-level transfer features.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project folder containing data/processed and outputs/tables. Defaults to the current directory.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique aligned trading dates reserved for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(project_root=args.project_root, test_fraction=args.test_fraction)
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
