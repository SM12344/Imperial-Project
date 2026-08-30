from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_modelling_baselines import build_paths


def load_required_outputs(paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_path = paths.tables_dir / "tsla_focus_results.csv"
    importance_path = paths.tables_dir / "tsla_feature_importance.csv"
    predictions_path = paths.processed_dir / "tsla_focus_predictions.csv"

    missing = [str(p) for p in [results_path, importance_path, predictions_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TSLA focus outputs. Run notebook 06 first.\n" + "\n".join(missing)
        )

    results_df = pd.read_csv(results_path)
    importance_df = pd.read_csv(importance_path)
    predictions_df = pd.read_csv(predictions_path)
    predictions_df["trading_date"] = pd.to_datetime(predictions_df["trading_date"])
    return results_df, importance_df, predictions_df


def build_benchmark_comparison(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_label, split_df in results_df.groupby("split_label", sort=True):
        by_feature = {
            row["feature_set"]: row for _, row in split_df.iterrows()
        }
        price_only = by_feature["price_only"]
        price_plus = by_feature["price_plus_finbert"]
        rows.append(
            {
                "split_label": split_label,
                "always_up_accuracy": price_plus["always_up_accuracy"],
                "price_only_accuracy": price_only["accuracy"],
                "price_plus_finbert_accuracy": price_plus["accuracy"],
                "price_only_roc_auc": price_only["roc_auc"],
                "price_plus_finbert_roc_auc": price_plus["roc_auc"],
                "accuracy_gain_vs_price_only": price_plus["accuracy"] - price_only["accuracy"],
                "roc_auc_gain_vs_price_only": price_plus["roc_auc"] - price_only["roc_auc"],
                "accuracy_gain_vs_always_up": price_plus["accuracy"] - price_plus["always_up_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("split_label")


def build_top_features(importance_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    top_rows = []
    for (split_label, feature_set), group in importance_df.groupby(["split_label", "feature_set"], sort=True):
        top_rows.append(group.sort_values("importance", ascending=False).head(top_n))
    return pd.concat(top_rows, ignore_index=True)


def build_prediction_examples(predictions_df: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for split_label, split_df in predictions_df.groupby("split_label", sort=True):
        plus_df = split_df[split_df["feature_set"] == "price_plus_finbert"].copy()
        correct = plus_df[plus_df["correct_prediction"] == 1].sort_values(
            "predicted_probability_up", ascending=False
        ).head(3)
        wrong = plus_df[plus_df["correct_prediction"] == 0].sort_values(
            "predicted_probability_up", ascending=False
        ).head(3)
        examples.extend([correct, wrong])
    if not examples:
        return pd.DataFrame()
    return pd.concat(examples, ignore_index=True).sort_values(["split_label", "trading_date"])


def run_pipeline(project_root: str | None = None) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    results_df, importance_df, predictions_df = load_required_outputs(paths)

    benchmark_df = build_benchmark_comparison(results_df)
    top_features_df = build_top_features(importance_df)
    examples_df = build_prediction_examples(predictions_df)

    benchmark_df.to_csv(paths.tables_dir / "tsla_report_benchmark_comparison.csv", index=False)
    top_features_df.to_csv(paths.tables_dir / "tsla_report_top_features.csv", index=False)
    examples_df.to_csv(paths.tables_dir / "tsla_report_prediction_examples.csv", index=False)

    return {
        "results_df": results_df,
        "benchmark_df": benchmark_df,
        "top_features_df": top_features_df,
        "examples_df": examples_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TSLA interpretation tables from the TSLA focus analysis outputs."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project folder containing data/processed and outputs/tables. Defaults to the current directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(project_root=args.project_root)
    print(outputs["benchmark_df"].to_string(index=False))
    print()
    print(outputs["top_features_df"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
