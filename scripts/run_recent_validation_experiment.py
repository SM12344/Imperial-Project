from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from run_2024_holdout_ablation import add_holdout_filter_columns
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_target_formulation_search import add_extended_targets
from run_target_formulation_ultra_screen import FEATURE_SET_NAMES, add_context, best_threshold, models, target_specs


TEST_START = "2024-01-01"
TEST_END = "2025-01-01"
VALIDATION_TRADING_DAYS = 63
MIN_TEST_ROWS = 500
MIN_VALIDATION_ROWS = 120
CONFIDENCE_FRACTIONS = [0.25, 0.40, 0.60, 0.80]


def metric_row(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
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


def split_recent_validation(
    df: pd.DataFrame,
    horizon: int,
    validation_trading_days: int = VALIDATION_TRADING_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df["trading_date"]).drop_duplicates()))
    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)
    test_start_idx = int(dates.searchsorted(test_start, side="left"))
    if test_start_idx <= horizon + validation_trading_days:
        raise ValueError("Not enough dates before the test period for recent validation split.")

    # Purge h trading days between validation features and the test start so
    # validation targets are fully observed before 2024 evaluation begins.
    validation_end_idx = max(0, test_start_idx - horizon)
    validation_start_idx = max(0, validation_end_idx - validation_trading_days)
    validation_start = dates[validation_start_idx]
    validation_end_exclusive = dates[validation_end_idx]

    train = df[df["trading_date"] < validation_start].copy()
    val = df[(df["trading_date"] >= validation_start) & (df["trading_date"] < validation_end_exclusive)].copy()
    test = df[(df["trading_date"] >= test_start) & (df["trading_date"] < test_end)].copy()
    split_info = {
        "train_start": train["trading_date"].min(),
        "train_end": train["trading_date"].max(),
        "validation_start": validation_start,
        "validation_end": val["trading_date"].max() if len(val) else pd.NaT,
        "test_start": test["trading_date"].min() if len(test) else pd.NaT,
        "test_end": test["trading_date"].max() if len(test) else pd.NaT,
        "embargo_trading_days": horizon,
        "validation_trading_days": validation_trading_days,
    }
    return train, val, test, split_info


def fit_selected(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[Any, dict[str, Any]] | None:
    y_train = train[target].astype(int)
    y_val = val[target].astype(int)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        return None

    best: dict[str, Any] | None = None
    for model_name, params, model in models(float(y_train.mean())):
        model.fit(train[feature_cols], y_train)
        val_proba = model.predict_proba(val[feature_cols])[:, 1]
        threshold, val_ba = best_threshold(y_val, val_proba)
        val_auc = roc_auc_score(y_val, val_proba) if y_val.nunique() == 2 else np.nan
        row = {
            "selected_model": model_name,
            "selected_params": params,
            "selected_threshold": threshold,
            "validation_roc_auc": float(val_auc),
            "validation_balanced_accuracy": float(val_ba),
        }
        if best is None or (row["validation_balanced_accuracy"], row["validation_roc_auc"]) > (
            best["validation_balanced_accuracy"],
            best["validation_roc_auc"],
        ):
            best = row

    if best is None:
        return None
    for model_name, params, model in models(float(y_train.mean())):
        if model_name == best["selected_model"] and params == best["selected_params"]:
            model.fit(train[feature_cols], y_train)
            return model, best
    return None


def final_model_from_selection(train_full: pd.DataFrame, target: str, selected: dict[str, Any]):
    positive_rate = float(train_full[target].astype(int).mean())
    for model_name, params, model in models(positive_rate):
        if model_name == selected["selected_model"] and params == selected["selected_params"]:
            model.fit(train_full, None)  # placeholder to satisfy type checkers
    raise RuntimeError("Selected model was not found.")


def refit_selected_model(train_full: pd.DataFrame, feature_cols: list[str], target: str, selected: dict[str, Any]):
    y_train_full = train_full[target].astype(int)
    for model_name, params, model in models(float(y_train_full.mean())):
        if model_name == selected["selected_model"] and params == selected["selected_params"]:
            model.fit(train_full[feature_cols], y_train_full)
            return model
    raise RuntimeError(f"Selected model not found: {selected['selected_model']} {selected['selected_params']}")


def prepare_dataset(project_root: str | None, dataset_name: str, scored_news_name: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    paths = build_paths(project_root)
    base = load_dataset(paths, dataset_name)
    scored_news = load_scored_news(paths, scored_news_name)
    df = add_extended_targets(
        add_expanded_features(
            add_holdout_filter_columns(
                add_features_and_targets(base, build_event_daily_features(scored_news))
            )
        )
    )
    df, context_cols = add_context(df)
    feature_sets = build_feature_sets(df)
    feature_sets["price_base_context"] = feature_sets["price_base"] + context_cols
    feature_sets["price_expanded_context"] = feature_sets["price_expanded"] + context_cols
    feature_sets["news_expanded_context"] = feature_sets["news_expanded"] + context_cols
    feature_sets["price_news_expanded_context"] = feature_sets["price_news_expanded"] + context_cols
    return df, feature_sets


def run_pipeline(
    project_root: str | None,
    dataset_name: str,
    scored_news_name: str,
    output_suffix: str,
    validation_trading_days: int = VALIDATION_TRADING_DAYS,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    df, feature_sets = prepare_dataset(project_root, dataset_name, scored_news_name)

    all_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for spec in target_specs():
        target = spec["target"]
        horizon = int(spec["horizon"])
        usable = df.dropna(subset=[target]).copy()
        train, val, test, split_info = split_recent_validation(usable, horizon, validation_trading_days)
        split_rows.append(
            {
                **spec,
                **{key: str(value.date()) if hasattr(value, "date") else value for key, value in split_info.items()},
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "train_positive_rate": float(train[target].mean()) if len(train) else np.nan,
                "validation_positive_rate": float(val[target].mean()) if len(val) else np.nan,
                "test_positive_rate": float(test[target].mean()) if len(test) else np.nan,
            }
        )
        if len(train) < 500 or len(val) < MIN_VALIDATION_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        y_val = val[target].astype(int)
        y_test = test[target].astype(int)
        if y_val.nunique() < 2 or y_test.nunique() < 2:
            continue

        for feature_set_name in FEATURE_SET_NAMES:
            feature_cols = [col for col in feature_sets[feature_set_name] if col in usable.columns]
            selected_output = fit_selected(train, val, feature_cols, target)
            if selected_output is None:
                continue
            train_model, selected = selected_output
            val_proba = train_model.predict_proba(val[feature_cols])[:, 1]

            train_full = pd.concat([train, val], ignore_index=True)
            final_model = refit_selected_model(train_full, feature_cols, target, selected)
            test_proba = final_model.predict_proba(test[feature_cols])[:, 1]
            baseline = float(max(y_test.mean(), 1 - y_test.mean()))
            row_base = {
                **spec,
                "feature_set": feature_set_name,
                "feature_count": len(feature_cols),
                "train_rows": len(train),
                "validation_rows": len(val),
                "train_plus_validation_rows": len(train_full),
                "test_rows": len(test),
                "train_start": str(split_info["train_start"].date()),
                "train_end": str(split_info["train_end"].date()),
                "validation_start": str(split_info["validation_start"].date()),
                "validation_end": str(split_info["validation_end"].date()),
                "test_start": str(split_info["test_start"].date()),
                "test_end": str(split_info["test_end"].date()),
                "embargo_trading_days": split_info["embargo_trading_days"],
                "test_positive_rate": float(y_test.mean()),
                "majority_baseline_accuracy": baseline,
                **selected,
            }
            all_metric = metric_row(y_test, test_proba, selected["selected_threshold"])
            all_row = {**row_base, **all_metric}
            all_row["accuracy_minus_baseline"] = all_row["accuracy"] - baseline
            all_row["defensible_candidate"] = (
                all_row["test_rows"] >= MIN_TEST_ROWS
                and all_row["accuracy"] >= 0.70
                and all_row["accuracy_minus_baseline"] >= 0.03
                and all_row["balanced_accuracy"] >= 0.55
            )
            all_rows.append(all_row)

            val_confidence = np.abs(val_proba - 0.5)
            test_confidence = np.abs(test_proba - 0.5)
            for fraction in CONFIDENCE_FRACTIONS:
                selected_val_rows = max(20, int(round(len(val) * fraction)))
                selected_val_rows = min(selected_val_rows, len(val))
                order = np.argsort(-val_confidence)
                idx = order[:selected_val_rows]
                confidence_cutoff = float(val_confidence[idx].min())
                mask = test_confidence >= confidence_cutoff
                if int(mask.sum()) < 150:
                    continue
                covered_y = y_test[mask]
                covered_baseline = float(max(covered_y.mean(), 1 - covered_y.mean()))
                conf_metric = metric_row(covered_y, test_proba[mask], 0.5)
                conf_row = {
                    **row_base,
                    "validation_confidence_fraction": fraction,
                    "validation_selected_rows": selected_val_rows,
                    "confidence_cutoff": confidence_cutoff,
                    "test_covered_rows": int(mask.sum()),
                    "test_coverage": float(mask.mean()),
                    "covered_positive_rate": float(covered_y.mean()),
                    "covered_majority_baseline_accuracy": covered_baseline,
                    **conf_metric,
                }
                conf_row["accuracy_minus_covered_baseline"] = conf_row["accuracy"] - covered_baseline
                conf_row["defensible_candidate"] = (
                    conf_row["test_covered_rows"] >= 300
                    and conf_row["accuracy"] >= 0.70
                    and conf_row["accuracy_minus_covered_baseline"] >= 0.03
                    and conf_row["balanced_accuracy"] >= 0.55
                )
                confidence_rows.append(conf_row)

    all_results = pd.DataFrame(all_rows)
    confidence = pd.DataFrame(confidence_rows)
    splits = pd.DataFrame(split_rows)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "scored_news_name": scored_news_name,
                "validation_trading_days": validation_trading_days,
                "test_period": f"{TEST_START} to 2024-12-31",
                "design": (
                    "Short recent validation window immediately before 2024. "
                    "For each target horizon, validation ends h trading days before the test period "
                    "so validation labels are observed before testing begins; the final model is refit "
                    "on train plus validation before 2024 evaluation."
                ),
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    all_results.to_csv(paths.tables_dir / f"recent_validation_all_rows{suffix}.csv", index=False)
    confidence.to_csv(paths.tables_dir / f"recent_validation_confidence{suffix}.csv", index=False)
    splits.to_csv(paths.tables_dir / f"recent_validation_splits{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"recent_validation_metadata{suffix}.csv", index=False)

    best_rows = []
    if not all_results.empty:
        best_rows.append(
            all_results.sort_values(
                ["accuracy_minus_baseline", "balanced_accuracy", "roc_auc"],
                ascending=[False, False, False],
            )
            .head(12)
            .assign(result_type="all_rows")
        )
    if not confidence.empty:
        best_rows.append(
            confidence.sort_values(
                ["accuracy_minus_covered_baseline", "balanced_accuracy", "roc_auc"],
                ascending=[False, False, False],
            )
            .head(12)
            .assign(result_type="selective")
        )
    best = pd.concat(best_rows, ignore_index=True, sort=False) if best_rows else pd.DataFrame()
    best.to_csv(paths.tables_dir / f"recent_validation_best_summary{suffix}.csv", index=False)

    return {
        "all_results": all_results,
        "confidence": confidence,
        "splits": splits,
        "metadata": metadata,
        "best": best,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recent-window validation experiment with horizon embargo.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--scored-news-name", required=True)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--validation-trading-days", type=int, default=VALIDATION_TRADING_DAYS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        project_root=args.project_root,
        dataset_name=args.dataset_name,
        scored_news_name=args.scored_news_name,
        output_suffix=args.output_suffix,
        validation_trading_days=args.validation_trading_days,
    )
    print(outputs["metadata"].to_string(index=False))
    print("\nBest recent-validation all-row results")
    if outputs["all_results"].empty:
        print("No all-row results.")
    else:
        print(
            outputs["all_results"]
            .sort_values(["accuracy_minus_baseline", "balanced_accuracy", "roc_auc"], ascending=[False, False, False])
            .head(25)
            .to_string(index=False)
        )
    print("\nBest recent-validation selective results")
    if outputs["confidence"].empty:
        print("No selective results.")
    else:
        print(
            outputs["confidence"]
            .sort_values(["accuracy_minus_covered_baseline", "balanced_accuracy", "roc_auc"], ascending=[False, False, False])
            .head(25)
            .to_string(index=False)
        )
    print("\nDefensible candidates")
    if not outputs["all_results"].empty:
        print(outputs["all_results"][outputs["all_results"]["defensible_candidate"]].to_string(index=False))
    if not outputs["confidence"].empty:
        print(outputs["confidence"][outputs["confidence"]["defensible_candidate"]].to_string(index=False))


if __name__ == "__main__":
    main()
