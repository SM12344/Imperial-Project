from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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

from run_modelling_baselines import FINBERT_FEATURES, PRICE_ONLY_FEATURES, load_dataset, time_split


FEATURE_SETS = {
    "price_only": {
        "numeric": PRICE_ONLY_FEATURES,
        "categorical": [],
    },
    "price_plus_finbert": {
        "numeric": PRICE_ONLY_FEATURES + FINBERT_FEATURES,
        "categorical": [],
    },
    "price_plus_finbert_ticker": {
        "numeric": PRICE_ONLY_FEATURES + FINBERT_FEATURES,
        "categorical": ["ticker"],
    },
}

MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


@dataclass
class ProjectPaths:
    root: Path
    processed_dir: Path
    tables_dir: Path
    model_dir: Path


def resolve_project_root(project_root: str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    if root.name in {"notebooks", "scripts"}:
        root = root.parent
    return root


def build_paths(project_root: str | None = None) -> ProjectPaths:
    root = resolve_project_root(project_root)
    processed_dir = root / "data" / "processed"
    tables_dir = root / "outputs" / "tables"
    model_dir = root / "models" / "sklearn"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, processed_dir=processed_dir, tables_dir=tables_dir, model_dir=model_dir)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
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


def run_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    feature_spec = FEATURE_SETS[feature_set_name]
    feature_cols = feature_spec["numeric"] + feature_spec["categorical"]
    pipeline = build_pipeline(model_name, feature_spec["numeric"], feature_spec["categorical"])

    X_train = train_df[feature_cols]
    y_train = train_df["target_next_day_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_next_day_up"]

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

    predictions = test_df[["ticker", "trading_date", "target_next_day_up", "has_news"]].copy()
    predictions["feature_set"] = feature_set_name
    predictions["model_name"] = model_name
    predictions["predicted_up"] = pred
    predictions["predicted_probability_up"] = proba
    predictions["correct_prediction"] = (predictions["predicted_up"] == predictions["target_next_day_up"]).astype(int)
    return result_row, predictions, pipeline


def select_best_candidate(results_df: pd.DataFrame) -> pd.Series:
    ranked = results_df.sort_values(
        ["roc_auc", "balanced_accuracy", "f1", "accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def save_feature_schema(paths: ProjectPaths, metadata: dict[str, Any]) -> Path:
    schema_path = paths.model_dir / "best_direction_model_schema.json"
    schema_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return schema_path


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    test_fraction: float = 0.2,
    refit_on_all_data: bool = True,
) -> dict[str, Any]:
    paths = build_paths(project_root)
    df = load_dataset(
        paths=type("BaselinePaths", (), {"processed_dir": paths.processed_dir})(),
        dataset_name=dataset_name,
    )
    train_df, test_df, cutoff_date = time_split(df, test_fraction=test_fraction)

    results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_candidates: dict[tuple[str, str], Pipeline] = {}

    for feature_set_name in FEATURE_SETS:
        for model_name in MODEL_NAMES:
            result_row, predictions, pipeline = run_candidate(train_df, test_df, feature_set_name, model_name)
            results.append(result_row)
            prediction_frames.append(predictions)
            fitted_candidates[(feature_set_name, model_name)] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "balanced_accuracy", "f1", "accuracy"],
        ascending=[False, False, False, False],
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    best = select_best_candidate(results_df)
    best_feature_set = str(best["feature_set"])
    best_model_name = str(best["model_name"])
    best_feature_spec = FEATURE_SETS[best_feature_set]
    best_feature_cols = best_feature_spec["numeric"] + best_feature_spec["categorical"]

    if refit_on_all_data:
        best_pipeline = build_pipeline(best_model_name, best_feature_spec["numeric"], best_feature_spec["categorical"])
        best_pipeline.fit(df[best_feature_cols], df["target_next_day_up"])
        trained_rows = len(df)
        trained_start = df["trading_date"].min().date().isoformat()
        trained_end = df["trading_date"].max().date().isoformat()
    else:
        best_pipeline = fitted_candidates[(best_feature_set, best_model_name)]
        trained_rows = len(train_df)
        trained_start = train_df["trading_date"].min().date().isoformat()
        trained_end = train_df["trading_date"].max().date().isoformat()

    model_path = paths.model_dir / "best_direction_model.joblib"
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "target": "target_next_day_up",
        "split_type": "time_based",
        "cutoff_date": cutoff_date.date().isoformat(),
        "test_fraction": test_fraction,
        "selected_feature_set": best_feature_set,
        "selected_model_name": best_model_name,
        "selected_holdout_metrics": {
            key: float(best[key])
            for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
        },
        "numeric_features": best_feature_spec["numeric"],
        "categorical_features": best_feature_spec["categorical"],
        "refit_on_all_data": bool(refit_on_all_data),
        "trained_rows": trained_rows,
        "trained_start": trained_start,
        "trained_end": trained_end,
        "model_path": str(model_path),
    }
    schema_path = save_feature_schema(paths, metadata)
    metadata_df = pd.DataFrame([metadata | {"schema_path": str(schema_path)}])

    results_df.to_csv(paths.tables_dir / "model_persistence_candidate_results.csv", index=False)
    predictions_df.to_csv(paths.processed_dir / "model_persistence_holdout_predictions.csv", index=False)
    metadata_df.to_csv(paths.tables_dir / "model_persistence_metadata.csv", index=False)

    return {
        "dataset": df,
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "predictions_df": predictions_df,
        "metadata_df": metadata_df,
        "best_pipeline": best_pipeline,
        "model_path": model_path,
        "schema_path": schema_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, select, and persist the best stock-direction baseline model.")
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
        help="Fraction of unique trading dates reserved for model selection.",
    )
    parser.add_argument(
        "--no-refit-on-all-data",
        action="store_true",
        help="Persist the model fitted only on the training split instead of refitting the selected model on all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        test_fraction=args.test_fraction,
        refit_on_all_data=not args.no_refit_on_all_data,
    )
    print(outputs["metadata_df"].to_string(index=False))
    print()
    print(outputs["results_df"].to_string(index=False))


if __name__ == "__main__":
    main()
