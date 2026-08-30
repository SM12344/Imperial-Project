from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_2024_holdout_ablation import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    VALIDATION_START,
    add_holdout_filter_columns,
    build_ablation_feature_sets,
    split_holdout,
)
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


SCOPES = ["pooled_all_tickers", "AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
TARGETS = ["fwd_3d_excess_return", "fwd_5d_excess_return"]
FEATURE_SET_NAMES = ["price_only", "news_all_only", "price_news_quality", "price_quality"]

RIDGE_GRID = [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 50.0}]
ELASTIC_NET_GRID = [
    {"alpha": 0.001, "l1_ratio": 0.1},
    {"alpha": 0.005, "l1_ratio": 0.1},
    {"alpha": 0.01, "l1_ratio": 0.2},
]
RANDOM_FOREST_GRID = [
    {"n_estimators": 160, "max_depth": 2, "min_samples_leaf": 30},
    {"n_estimators": 220, "max_depth": 3, "min_samples_leaf": 40},
]
LIGHTGBM_GRID = [
    {"n_estimators": 120, "max_depth": 1, "num_leaves": 3, "learning_rate": 0.025, "min_child_samples": 35, "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"n_estimators": 200, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.02, "min_child_samples": 45, "reg_lambda": 20.0, "reg_alpha": 0.2},
]
XGBOOST_GRID = [
    {"n_estimators": 120, "max_depth": 1, "learning_rate": 0.025, "min_child_weight": 8, "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.02, "min_child_weight": 12, "reg_lambda": 20.0, "reg_alpha": 0.2},
]
CATBOOST_GRID = [
    {"iterations": 120, "depth": 1, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
    {"iterations": 200, "depth": 2, "learning_rate": 0.02, "l2_leaf_reg": 20.0},
]


def scope_frame(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "pooled_all_tickers":
        return df.copy()
    return df[df["ticker"] == scope].copy()


def build_model(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "ridge":
        model = Ridge(**params)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "elastic_net":
        model = ElasticNet(**params, max_iter=10000, random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "random_forest":
        model = RandomForestRegressor(**params, random_state=42, n_jobs=1)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        model = LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbose=-1)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        model = XGBRegressor(**params, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=1)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostRegressor(**params, loss_function="RMSE", verbose=False, random_seed=42, thread_count=1)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    raise ValueError(model_name)


def candidate_models() -> list[tuple[str, dict[str, Any]]]:
    candidates = [("ridge", params) for params in RIDGE_GRID]
    candidates.extend(("elastic_net", params) for params in ELASTIC_NET_GRID)
    candidates.extend(("random_forest", params) for params in RANDOM_FOREST_GRID)
    if LGBMRegressor is not None:
        candidates.extend(("lightgbm", params) for params in LIGHTGBM_GRID)
    if XGBRegressor is not None:
        candidates.extend(("xgboost", params) for params in XGBOOST_GRID)
    if CatBoostRegressor is not None:
        candidates.extend(("catboost", params) for params in CATBOOST_GRID)
    return candidates


def regression_metrics(y_true: pd.Series, pred_return: np.ndarray) -> dict[str, float]:
    baseline_pred = np.repeat(float(y_true.mean()), len(y_true))
    actual_direction = (y_true > 0).astype(int)
    pred_direction = (pred_return > 0).astype(int)
    majority_baseline = float(max(actual_direction.mean(), 1 - actual_direction.mean()))
    mse = mean_squared_error(y_true, pred_return)
    baseline_mse = mean_squared_error(y_true, baseline_pred)
    return {
        "mae": mean_absolute_error(y_true, pred_return),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, pred_return),
        "correlation": float(np.corrcoef(y_true, pred_return)[0, 1]) if np.std(pred_return) > 0 and np.std(y_true) > 0 else np.nan,
        "baseline_mae_mean_return": mean_absolute_error(y_true, baseline_pred),
        "baseline_rmse_mean_return": float(np.sqrt(baseline_mse)),
        "directional_accuracy": accuracy_score(actual_direction, pred_direction),
        "directional_balanced_accuracy": balanced_accuracy_score(actual_direction, pred_direction),
        "directional_majority_baseline": majority_baseline,
        "actual_positive_rate": float(actual_direction.mean()),
        "predicted_positive_rate": float(pred_direction.mean()),
        "actual_return_mean": float(y_true.mean()),
        "predicted_return_mean": float(np.mean(pred_return)),
        "actual_return_std": float(y_true.std()),
        "predicted_return_std": float(np.std(pred_return)),
    }


def tune_regressor(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str) -> dict[str, Any] | None:
    if len(train) < 80 or len(val) < 25:
        return None
    y_train = train[target].astype(float)
    y_val = val[target].astype(float)
    best: dict[str, Any] | None = None
    for model_name, params in candidate_models():
        model = build_model(model_name, params)
        model.fit(train[feature_cols], y_train)
        val_pred = model.predict(val[feature_cols])
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        corr = float(np.corrcoef(y_val, val_pred)[0, 1]) if np.std(val_pred) > 0 and np.std(y_val) > 0 else np.nan
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "validation_rmse": float(rmse),
            "validation_mae": float(mean_absolute_error(y_val, val_pred)),
            "validation_correlation": corr,
            "validation_r2": float(r2_score(y_val, val_pred)),
        }
        if best is None or (row["validation_rmse"], -np.nan_to_num(row["validation_correlation"], nan=-999.0)) < (
            best["validation_rmse"],
            -np.nan_to_num(best["validation_correlation"], nan=-999.0),
        ):
            best = row
    return best


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))
    feature_sets = build_ablation_feature_sets(df)

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for scope in SCOPES:
            scoped = scope_frame(df, scope)
            train, val, test = split_holdout(scoped)
            if len(test) < 25:
                continue
            for feature_set_name in FEATURE_SET_NAMES:
                feature_cols = [col for col in feature_sets[feature_set_name] if col in scoped.columns]
                if not feature_cols:
                    continue
                selected = tune_regressor(train, val, feature_cols, target)
                if selected is None:
                    continue
                train_full = pd.concat([train, val], ignore_index=True)
                model = build_model(selected["selected_model"], selected["selected_params"])
                model.fit(train_full[feature_cols], train_full[target].astype(float))
                test_pred = model.predict(test[feature_cols])
                y_test = test[target].astype(float)
                result_rows.append(
                    {
                        "scope": scope,
                        "target": target,
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_cols),
                        "train_rows": len(train),
                        "validation_rows": len(val),
                        "train_plus_validation_rows": len(train_full),
                        "test_rows": len(test),
                        **selected,
                        **regression_metrics(y_test, test_pred),
                    }
                )
                prediction_rows.extend(
                    {
                        "scope": scope,
                        "ticker": row.ticker,
                        "trading_date": row.trading_date,
                        "target": target,
                        "feature_set": feature_set_name,
                        "actual_return": float(actual),
                        "predicted_return": float(pred),
                        "actual_direction": int(actual > 0),
                        "predicted_direction": int(pred > 0),
                        "selected_model": selected["selected_model"],
                    }
                    for row, actual, pred in zip(test.itertuples(index=False), y_test, test_pred)
                )

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "targets": ", ".join(TARGETS),
                "feature_sets": ", ".join(FEATURE_SET_NAMES),
                "models": "ridge, elastic_net, random_forest, lightgbm, xgboost, catboost when installed",
                "design": "Predict continuous future excess return, then derive directional accuracy from predicted sign.",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"return_regression_results{suffix}.csv", index=False)
    predictions.to_csv(paths.tables_dir / f"return_regression_predictions{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"return_regression_metadata{suffix}.csv", index=False)
    return {"results": results, "predictions": predictions, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict holdout regression experiments for future excess returns.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nBest pooled rows by RMSE")
    pooled = outputs["results"][outputs["results"]["scope"] == "pooled_all_tickers"].copy()
    print(pooled.sort_values(["target", "rmse", "mae"]).to_string(index=False))
    print("\nBest rows by directional balanced accuracy")
    print(
        outputs["results"]
        .sort_values(["target", "directional_balanced_accuracy", "correlation"], ascending=[True, False, False])
        .groupby(["target", "scope"], as_index=False)
        .head(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
