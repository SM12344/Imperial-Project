from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from run_high_signal_event_experiments import (
    add_features_and_targets,
    add_filter_columns,
    build_feature_sets,
    build_splits,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_modelling_baselines import build_paths, load_dataset

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


SCENARIOS = [
    {
        "scenario": "pooled_all_3d_abs_move",
        "scope": "pooled_all_tickers",
        "filter": "all_days",
        "target": "target_3d_abs_excess_gt_1pct",
    },
    {
        "scenario": "pooled_high_news_3d_direction",
        "scope": "pooled_all_tickers",
        "filter": "top_10pct_news_volume",
        "target": "target_3d_excess_gt_0",
    },
    {
        "scenario": "pooled_strong_sentiment_5d_abs_move",
        "scope": "pooled_all_tickers",
        "filter": "strong_abs_sentiment",
        "target": "target_5d_abs_excess_gt_1pct",
    },
    {
        "scenario": "aapl_macro_3d_abs_move",
        "scope": "AAPL",
        "filter": "macro_high_market",
        "target": "target_3d_abs_excess_gt_1pct",
    },
    {
        "scenario": "amzn_all_5d_direction",
        "scope": "AMZN",
        "filter": "all_days",
        "target": "target_5d_excess_gt_0",
    },
    {
        "scenario": "msft_macro_5d_gt_2pct",
        "scope": "MSFT",
        "filter": "macro_high_market",
        "target": "target_5d_excess_gt_2pct",
    },
]

FEATURE_SET_NAMES = ["price_only", "price_news_events", "quality_news_only", "price_news_quality"]

LIGHTGBM_GRID = [
    {"n_estimators": 80, "max_depth": 1, "num_leaves": 3, "learning_rate": 0.03, "min_child_samples": 25, "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"n_estimators": 140, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.025, "min_child_samples": 30, "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"n_estimators": 220, "max_depth": 2, "num_leaves": 5, "learning_rate": 0.02, "min_child_samples": 40, "reg_lambda": 15.0, "reg_alpha": 0.2},
    {"n_estimators": 120, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.02, "min_child_samples": 35, "reg_lambda": 20.0, "reg_alpha": 0.5},
]

XGBOOST_GRID = [
    {"n_estimators": 80, "max_depth": 1, "learning_rate": 0.03, "min_child_weight": 5, "reg_lambda": 10.0, "reg_alpha": 0.0},
    {"n_estimators": 140, "max_depth": 2, "learning_rate": 0.025, "min_child_weight": 8, "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.02, "min_child_weight": 10, "reg_lambda": 15.0, "reg_alpha": 0.2},
    {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 12, "reg_lambda": 20.0, "reg_alpha": 0.5},
]

CATBOOST_GRID = [
    {"iterations": 80, "depth": 1, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
    {"iterations": 140, "depth": 2, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
    {"iterations": 220, "depth": 2, "learning_rate": 0.02, "l2_leaf_reg": 15.0},
    {"iterations": 120, "depth": 3, "learning_rate": 0.02, "l2_leaf_reg": 20.0},
]


def build_model(model_name: str, params: dict[str, Any], positive_rate: float) -> Pipeline:
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        scale_pos_weight = (1 - positive_rate) / positive_rate if 0 < positive_rate < 1 else 1.0
        model = LGBMClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        scale_pos_weight = (1 - positive_rate) / positive_rate if 0 < positive_rate < 1 else 1.0
        model = XGBClassifier(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")
        class_weights = [1.0, (1 - positive_rate) / positive_rate] if 0 < positive_rate < 1 else None
        model = CatBoostClassifier(
            **params,
            class_weights=class_weights,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
            thread_count=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])


def grid_for_model(model_name: str) -> list[dict[str, Any]]:
    if model_name == "lightgbm":
        return LIGHTGBM_GRID
    if model_name == "xgboost":
        return XGBOOST_GRID
    if model_name == "catboost":
        return CATBOOST_GRID
    raise ValueError(model_name)


def metric_row(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def chronological_train_validation_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(train["trading_date"].drop_duplicates())
    cut = max(5, int(len(dates) * 0.8))
    if cut >= len(dates):
        cut = len(dates) - 1
    train_dates = dates[:cut]
    val_dates = dates[cut:]
    return train[train["trading_date"].isin(train_dates)].copy(), train[train["trading_date"].isin(val_dates)].copy()


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_balanced = -np.inf
    for threshold in np.linspace(0.2, 0.8, 25):
        score = balanced_accuracy_score(y_true, (proba >= threshold).astype(int))
        if score > best_balanced:
            best_balanced = score
            best_t = float(threshold)
    return best_t, float(best_balanced)


def tune_on_validation(train: pd.DataFrame, feature_cols: list[str], target: str, model_name: str) -> dict[str, Any] | None:
    train_inner, val = chronological_train_validation_split(train)
    y_train = train_inner[target].astype(int)
    y_val = val[target].astype(int)
    if len(train_inner) < 50 or len(val) < 15 or y_train.nunique() < 2 or y_val.nunique() < 2:
        return None

    best: dict[str, Any] | None = None
    for params in grid_for_model(model_name):
        model = build_model(model_name, params, float(y_train.mean()))
        model.fit(train_inner[feature_cols], y_train)
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        threshold, threshold_balanced_accuracy = best_threshold(y_val, val_proba)
        row = {
            "selected_params": params,
            "selected_threshold": threshold,
            "validation_roc_auc": float(val_auc),
            "validation_balanced_accuracy_at_threshold": threshold_balanced_accuracy,
            "validation_rows": len(val),
            "validation_positive_rate": float(y_val.mean()),
        }
        if best is None or (row["validation_roc_auc"], row["validation_balanced_accuracy_at_threshold"]) > (
            best["validation_roc_auc"],
            best["validation_balanced_accuracy_at_threshold"],
        ):
            best = row
    return best


def scenario_frame(df: pd.DataFrame, scenario: dict[str, str]) -> pd.DataFrame:
    if scenario["scope"] == "pooled_all_tickers":
        scoped = df.copy()
    else:
        scoped = df[df["ticker"] == scenario["scope"]].copy()
    return scoped[scoped[scenario["filter"]] == 1].copy()


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_feature_sets(df)

    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        sdf = scenario_frame(df, scenario)
        splits = build_splits(sdf)
        coverage_rows.append(
            {
                **scenario,
                "rows": len(sdf),
                "unique_dates": sdf["trading_date"].nunique(),
                "num_folds": len(splits),
                "mean_news_count": sdf["news_count"].mean() if len(sdf) else 0.0,
                "positive_rate": sdf[scenario["target"]].mean() if len(sdf) else np.nan,
            }
        )
        for train, test, fold in splits:
            y_test = test[scenario["target"]].astype(int)
            if len(train) < 80 or len(test) < 25 or train[scenario["target"]].nunique() < 2 or y_test.nunique() < 2:
                continue
            for feature_set_name in FEATURE_SET_NAMES:
                if feature_set_name not in feature_sets:
                    continue
                feature_cols = [col for col in feature_sets[feature_set_name] if col in train.columns]
                if not feature_cols:
                    continue
                for model_name in ["lightgbm", "xgboost", "catboost"]:
                    selected = tune_on_validation(train, feature_cols, scenario["target"], model_name)
                    if selected is None:
                        continue
                    y_train = train[scenario["target"]].astype(int)
                    model = build_model(model_name, selected["selected_params"], float(y_train.mean()))
                    model.fit(train[feature_cols], y_train)
                    test_proba = model.predict_proba(test[feature_cols])[:, 1]
                    tuned_pred = (test_proba >= selected["selected_threshold"]).astype(int)
                    default_pred = (test_proba >= 0.5).astype(int)
                    rows.append(
                        {
                            **scenario,
                            "fold": fold,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "test_positive_rate": float(y_test.mean()),
                            "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                            **selected,
                            "threshold_mode": "validation_tuned",
                            **metric_row(y_test, tuned_pred, test_proba),
                        }
                    )
                    rows.append(
                        {
                            **scenario,
                            "fold": fold,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "feature_count": len(feature_cols),
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "test_positive_rate": float(y_test.mean()),
                            "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
                            **selected,
                            "threshold_mode": "default_0.5",
                            **metric_row(y_test, default_pred, test_proba),
                        }
                    )

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
        "train_rows",
        "test_rows",
        "validation_roc_auc",
        "validation_balanced_accuracy_at_threshold",
    ]
    group_cols = ["scenario", "scope", "filter", "target", "feature_set", "model_name", "threshold_mode"]
    summary = (
        results.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .rename(columns={col: f"mean_{col}" for col in metric_cols})
    )
    counts = results.groupby(group_cols, as_index=False)["fold"].count().rename(columns={"fold": "num_folds"})
    summary = summary.merge(counts, on=group_cols, how="left")
    best = summary.sort_values(["scenario", "mean_roc_auc", "mean_balanced_accuracy"], ascending=[True, False, False]).groupby(
        "scenario", as_index=False
    ).head(5)
    coverage = pd.DataFrame(coverage_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "rows": len(df),
                "scenarios": len(SCENARIOS),
                "models": "lightgbm, xgboost, catboost",
                "feature_sets": ", ".join(name for name in FEATURE_SET_NAMES if name in feature_sets),
                "tuning_method": "Nested chronological validation inside each walk-forward training fold",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"tuned_boosting_results{suffix}.csv", index=False)
    summary.to_csv(paths.tables_dir / f"tuned_boosting_summary{suffix}.csv", index=False)
    best.to_csv(paths.tables_dir / f"tuned_boosting_best{suffix}.csv", index=False)
    coverage.to_csv(paths.tables_dir / f"tuned_boosting_coverage{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"tuned_boosting_metadata{suffix}.csv", index=False)
    return {"results": results, "summary": summary, "best": best, "coverage": coverage, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested walk-forward tuning for shallow boosting models.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nScenario coverage")
    print(outputs["coverage"].to_string(index=False))
    print("\nBest tuned rows")
    print(outputs["best"].to_string(index=False))


if __name__ == "__main__":
    main()
