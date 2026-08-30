from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_2024_holdout_ablation import TEST_END, TEST_START, TRAIN_END, VALIDATION_START, add_holdout_filter_columns, split_holdout
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_target_formulation_search import HORIZONS, add_extended_targets

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


TICKERS = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
BASE_FEATURE_SETS = ["price_base", "price_expanded", "news_expanded", "price_news_expanded"]
CONFIDENCE_ROWS = [300, 500, 800, 1000]


def add_known_context_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["trading_date"] = pd.to_datetime(out["trading_date"])
    ticker_cols = []
    for ticker in TICKERS:
        col = f"ticker_is_{ticker.lower()}"
        out[col] = (out["ticker"] == ticker).astype(int)
        ticker_cols.append(col)
    out["month_sin"] = np.sin(2 * np.pi * out["trading_date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["trading_date"].dt.month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["trading_date"].dt.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["trading_date"].dt.dayofweek / 5)
    return out, ticker_cols + ["month_sin", "month_cos", "dow_sin", "dow_cos"]


def target_specs() -> list[dict[str, Any]]:
    specs = []
    for horizon in HORIZONS:
        for prefix in ["raw", "excess"]:
            for suffix in ["gt0", "gt1pct", "gt2pct"]:
                specs.append({"horizon": horizon, "target": f"target_{prefix}_{horizon}d_{suffix}", "target_family": f"{prefix}_{suffix}"})
        specs.append({"horizon": horizon, "target": f"target_raw_abs_{horizon}d_gt2pct", "target_family": "raw_abs_gt2pct"})
        specs.append({"horizon": horizon, "target": f"target_excess_abs_{horizon}d_gt1pct", "target_family": "excess_abs_gt1pct"})
    return specs


def best_threshold(y_true: pd.Series, proba: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.2, 0.8, 25):
        score = balanced_accuracy_score(y_true, (proba >= threshold).astype(int))
        if score > best_score:
            best_score = float(score)
            best_t = float(threshold)
    return best_t, best_score


def build_model(model_name: str, params: dict[str, Any], positive_rate: float) -> Pipeline:
    if model_name == "logistic_balanced":
        model = LogisticRegression(**params, max_iter=3000, class_weight="balanced", random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler()), ("model", model)])
    if model_name == "random_forest":
        model = RandomForestClassifier(**params, random_state=42, n_jobs=1, class_weight="balanced_subsample")
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
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
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        scale_pos_weight = (1 - positive_rate) / positive_rate if 0 < positive_rate < 1 else 1.0
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("model", model)])
    raise ValueError(model_name)


def candidates() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [
        ("logistic_balanced", {"C": 0.05}),
        ("logistic_balanced", {"C": 0.2}),
        ("logistic_balanced", {"C": 1.0}),
        ("random_forest", {"n_estimators": 180, "max_depth": 2, "min_samples_leaf": 35}),
    ]
    if LGBMClassifier is not None:
        out.extend(
            [
                ("lightgbm", {"n_estimators": 80, "max_depth": 1, "num_leaves": 3, "learning_rate": 0.03, "min_child_samples": 35, "reg_lambda": 10.0, "reg_alpha": 0.1}),
                ("lightgbm", {"n_estimators": 120, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.025, "min_child_samples": 45, "reg_lambda": 20.0, "reg_alpha": 0.2}),
            ]
        )
    if XGBClassifier is not None:
        out.append(("xgboost", {"n_estimators": 100, "max_depth": 1, "learning_rate": 0.03, "min_child_weight": 10, "reg_lambda": 15.0, "reg_alpha": 0.2}))
    return out


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else np.nan,
        "predicted_positive_rate": float(pred.mean()),
    }


def tune_and_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    if len(train) < 200 or len(val) < 80 or y_train.nunique() < 2 or y_val.nunique() < 2:
        return None
    best: dict[str, Any] | None = None
    for model_name, params in candidates():
        model = build_model(model_name, params, float(y_train.mean()))
        model.fit(train[feature_cols], y_train)
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        threshold, val_ba = best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba)
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "selected_threshold": threshold,
            "validation_roc_auc": float(val_auc),
            "validation_balanced_accuracy_at_threshold": float(val_ba),
        }
        if best is None or (row["validation_roc_auc"], row["validation_balanced_accuracy_at_threshold"]) > (
            best["validation_roc_auc"],
            best["validation_balanced_accuracy_at_threshold"],
        ):
            best = row
    if best is None:
        return None
    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[target].astype(int)
    model = build_model(best["selected_model"], best["selected_params"], float(y_train_full.mean()))
    model.fit(train_full[feature_cols], y_train_full)
    return model.predict_proba(test[feature_cols])[:, 1], best


def run_pipeline(project_root: str | None, dataset_name: str, scored_news_name: str, output_suffix: str) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_extended_targets(add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))))
    df, context_cols = add_known_context_features(df)
    feature_sets = build_feature_sets(df)
    for name in BASE_FEATURE_SETS:
        feature_sets[f"{name}_context"] = feature_sets[name] + context_cols

    rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for spec in target_specs():
        target = spec["target"]
        if target not in df.columns:
            continue
        usable = df.dropna(subset=[target]).copy()
        train, val, test = split_holdout(usable)
        if len(train) < 200 or len(val) < 80 or len(test) < 500:
            continue
        y_test = test[target].astype(int)
        if y_test.nunique() < 2:
            continue
        distribution = {
            **spec,
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "test_positive_rate": float(y_test.mean()),
            "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        }
        distribution_rows.append(distribution)
        for feature_set_name in [*BASE_FEATURE_SETS, *(f"{name}_context" for name in BASE_FEATURE_SETS)]:
            feature_cols = [col for col in feature_sets[feature_set_name] if col in usable.columns]
            output = tune_and_predict(train, val, test, feature_cols, target)
            if output is None:
                continue
            proba, selected = output
            row_base = {
                **distribution,
                "feature_set": feature_set_name,
                "feature_count": len(feature_cols),
                **selected,
            }
            row = {**row_base, **metrics(y_test, proba, selected["selected_threshold"])}
            row["accuracy_minus_baseline"] = row["accuracy"] - row["majority_baseline_accuracy"]
            row["defensible_70_candidate"] = (
                row["accuracy"] >= 0.70
                and row["accuracy_minus_baseline"] >= 0.03
                and row["balanced_accuracy"] >= 0.55
            )
            rows.append(row)
            confidence = np.abs(proba - 0.5)
            order = np.argsort(-confidence)
            for covered_rows in CONFIDENCE_ROWS:
                if len(order) < covered_rows:
                    continue
                idx = order[:covered_rows]
                covered_y = y_test.iloc[idx]
                if covered_y.nunique() < 2:
                    continue
                conf_row = {
                    **row_base,
                    "covered_rows": covered_rows,
                    "coverage": covered_rows / len(test),
                    "covered_positive_rate": float(covered_y.mean()),
                    "covered_majority_baseline_accuracy": float(max(covered_y.mean(), 1 - covered_y.mean())),
                    "confidence_cutoff": float(confidence[idx].min()),
                    **metrics(covered_y, proba[idx], 0.5),
                }
                conf_row["accuracy_minus_covered_baseline"] = conf_row["accuracy"] - conf_row["covered_majority_baseline_accuracy"]
                conf_row["defensible_70_candidate"] = (
                    covered_rows >= 500
                    and conf_row["accuracy"] >= 0.70
                    and conf_row["accuracy_minus_covered_baseline"] >= 0.03
                    and conf_row["balanced_accuracy"] >= 0.55
                )
                confidence_rows.append(conf_row)

    results = pd.DataFrame(rows)
    confidence = pd.DataFrame(confidence_rows)
    distributions = pd.DataFrame(distribution_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "train_period": f"2020-01-01 to {pd.Timestamp(TRAIN_END).date() - pd.Timedelta(days=1)}",
                "validation_period": f"{VALIDATION_START} to {pd.Timestamp(TEST_START).date() - pd.Timedelta(days=1)}",
                "test_period": f"{TEST_START} to {pd.Timestamp(TEST_END).date() - pd.Timedelta(days=1)}",
                "scope": "pooled_all_tickers",
                "horizons": ", ".join(str(h) for h in HORIZONS),
                "defensible_70_rule": "accuracy >= 0.70, rows >= 500, accuracy at least 0.03 above majority baseline, balanced accuracy >= 0.55",
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"target_formulation_screen_results{suffix}.csv", index=False)
    confidence.to_csv(paths.tables_dir / f"target_formulation_screen_confidence{suffix}.csv", index=False)
    distributions.to_csv(paths.tables_dir / f"target_formulation_screen_distributions{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"target_formulation_screen_metadata{suffix}.csv", index=False)
    return {"results": results, "confidence": confidence, "distributions": distributions, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast pooled target formulation screen.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.scored_news_name, args.output_suffix)
    print(outputs["metadata"].to_string(index=False))
    print("\nBest all-row results")
    print(
        outputs["results"]
        .sort_values(["accuracy", "accuracy_minus_baseline", "balanced_accuracy"], ascending=[False, False, False])
        .head(40)
        .to_string(index=False)
    )
    print("\nBest confidence results")
    print(
        outputs["confidence"]
        .sort_values(["accuracy", "accuracy_minus_covered_baseline", "balanced_accuracy"], ascending=[False, False, False])
        .head(40)
        .to_string(index=False)
    )
    print("\nDefensible 0.70 candidates")
    print(outputs["results"][outputs["results"]["defensible_70_candidate"]].to_string(index=False))
    print(outputs["confidence"][outputs["confidence"]["defensible_70_candidate"]].to_string(index=False))


if __name__ == "__main__":
    main()
