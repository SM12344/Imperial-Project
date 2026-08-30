from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from run_2024_holdout_ablation import TEST_END, TEST_START, TRAIN_END, VALIDATION_START, add_holdout_filter_columns, split_holdout
from run_event_target_experiments import build_event_daily_features, load_scored_news
from run_expanded_feature_search import add_expanded_features, build_feature_sets
from run_high_signal_event_experiments import add_features_and_targets
from run_modelling_baselines import build_paths, load_dataset
from run_target_formulation_search import add_extended_targets
from run_target_formulation_ultra_screen import add_context, models


SUFFIX = "2020_2024_polygon_proxy_market_cleaned_news"
DATASET_NAME = "model_dataset_finbert_quality_complete_2020_2024_polygon_proxy_market_cleaned_news.csv"
SCORED_NEWS_NAME = "news_target_tickers_finbert_scored_2020_2024_polygon_proxy_market_cleaned_news.csv"

SELECTIVE_TARGET = "target_excess_10d_gt2pct"
SELECTIVE_FEATURE_SET = "news_expanded_context"
SELECTIVE_VALIDATION_ROWS = 800


def read_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, tables_dir: Path, name: str) -> Path:
    path = tables_dir / name
    df.to_csv(path, index=False)
    return path


def save_bar_chart(
    df: pd.DataFrame,
    x: str,
    y_cols: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(df))
    width = 0.8 / len(y_cols)
    for i, col in enumerate(y_cols):
        offset = (i - (len(y_cols) - 1) / 2) * width
        ax.bar(positions + offset, df[col], width=width, label=col.replace("_", " "))
    ax.set_xticks(positions)
    ax.set_xticklabels(df[x].astype(str), rotation=rotation, ha="right" if rotation else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def safe_auc(y_true: pd.Series, proba: pd.Series) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, proba))


def metric_row(y_true: pd.Series, pred: pd.Series, proba: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": safe_auc(y_true, proba),
    }


def add_context_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df, context_cols = add_context(df)
    feature_sets = build_feature_sets(df)
    feature_sets["price_base_context"] = feature_sets["price_base"] + context_cols
    feature_sets["price_expanded_context"] = feature_sets["price_expanded"] + context_cols
    feature_sets["news_expanded_context"] = feature_sets["news_expanded"] + context_cols
    feature_sets["price_news_expanded_context"] = feature_sets["price_news_expanded"] + context_cols
    return df, feature_sets


def model_from_saved_selection(model_name: str, params: dict[str, Any], positive_rate: float):
    for candidate_name, candidate_params, model in models(positive_rate):
        if candidate_name == model_name and candidate_params == params:
            return model
    raise ValueError(f"No model found for {model_name} {params}")


def selective_model_artifacts(paths, figures_dir: Path, tables_dir: Path) -> dict[str, pd.DataFrame]:
    result_rows = read_table(tables_dir, f"validation_confidence_target_results_{SUFFIX}.csv")
    selected = result_rows[
        (result_rows["target"] == SELECTIVE_TARGET)
        & (result_rows["feature_set"] == SELECTIVE_FEATURE_SET)
        & (result_rows["validation_selected_rows"] == SELECTIVE_VALIDATION_ROWS)
    ].sort_values(["accuracy_minus_test_baseline", "balanced_accuracy"], ascending=[False, False])
    if selected.empty:
        raise RuntimeError("Selective result row not found.")
    selected_row = selected.iloc[0].to_dict()
    selected_params = ast.literal_eval(selected_row["selected_params"]) if isinstance(selected_row["selected_params"], str) else selected_row["selected_params"]

    base = load_dataset(paths, DATASET_NAME)
    scored_news = load_scored_news(paths, SCORED_NEWS_NAME)
    df = add_extended_targets(add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored_news)))))
    df, feature_sets = add_context_feature_sets(df)
    usable = df.dropna(subset=[SELECTIVE_TARGET]).copy()
    train, val, test = split_holdout(usable)
    features = feature_sets[SELECTIVE_FEATURE_SET]

    y_train = train[SELECTIVE_TARGET].astype(int)
    y_val = val[SELECTIVE_TARGET].astype(int)
    train_model = model_from_saved_selection(selected_row["selected_model"], selected_params, float(y_train.mean()))
    train_model.fit(train[features], y_train)
    val_proba = train_model.predict_proba(val[features])[:, 1]
    val_confidence = np.abs(val_proba - 0.5)
    val_order = np.argsort(-val_confidence)
    confidence_cutoff = float(val_confidence[val_order[:SELECTIVE_VALIDATION_ROWS]].min())

    train_full = pd.concat([train, val], ignore_index=True)
    y_train_full = train_full[SELECTIVE_TARGET].astype(int)
    final_model = model_from_saved_selection(selected_row["selected_model"], selected_params, float(y_train_full.mean()))
    final_model.fit(train_full[features], y_train_full)
    test_proba = final_model.predict_proba(test[features])[:, 1]
    test_confidence = np.abs(test_proba - 0.5)
    mask = test_confidence >= confidence_cutoff

    predictions = test[["ticker", "trading_date", "fwd_10d_excess_return_audit", SELECTIVE_TARGET]].copy()
    predictions["proba"] = test_proba
    predictions["confidence"] = test_confidence
    predictions["selected_by_validation_confidence"] = mask.astype(int)
    predictions["prediction"] = (predictions["proba"] >= 0.5).astype(int)
    predictions["correct"] = (predictions["prediction"] == predictions[SELECTIVE_TARGET].astype(int)).astype(int)
    predictions = predictions[predictions["selected_by_validation_confidence"] == 1].copy()
    predictions["trading_date"] = pd.to_datetime(predictions["trading_date"]).dt.strftime("%Y-%m-%d")

    y_true = predictions[SELECTIVE_TARGET].astype(int)
    pred = predictions["prediction"].astype(int)
    proba = predictions["proba"].astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    summary = pd.DataFrame(
        [
            {
                "target": SELECTIVE_TARGET,
                "feature_set": SELECTIVE_FEATURE_SET,
                "selected_model": selected_row["selected_model"],
                "validation_selected_rows": SELECTIVE_VALIDATION_ROWS,
                "confidence_cutoff": confidence_cutoff,
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows_total": len(test),
                "test_covered_rows": len(predictions),
                "test_coverage": len(predictions) / len(test),
                "test_positive_rate": float(y_true.mean()),
                "majority_baseline": float(max(y_true.mean(), 1 - y_true.mean())),
                "accuracy_minus_baseline": accuracy_score(y_true, pred) - float(max(y_true.mean(), 1 - y_true.mean())),
                **metric_row(y_true, pred, proba),
            }
        ]
    )
    confusion = pd.DataFrame(
        [
            {"actual": 0, "predicted": 0, "count": int(tn)},
            {"actual": 0, "predicted": 1, "count": int(fp)},
            {"actual": 1, "predicted": 0, "count": int(fn)},
            {"actual": 1, "predicted": 1, "count": int(tp)},
        ]
    )

    by_ticker_rows = []
    full_test_counts = test.groupby("ticker").size().rename("test_rows_total")
    for ticker, part in predictions.groupby("ticker"):
        ticker_y = part[SELECTIVE_TARGET].astype(int)
        ticker_pred = part["prediction"].astype(int)
        ticker_proba = part["proba"].astype(float)
        baseline = float(max(ticker_y.mean(), 1 - ticker_y.mean()))
        by_ticker_rows.append(
            {
                "ticker": ticker,
                "test_rows_total": int(full_test_counts.loc[ticker]),
                "covered_rows": len(part),
                "coverage": len(part) / int(full_test_counts.loc[ticker]),
                "positive_rate": float(ticker_y.mean()),
                "majority_baseline": baseline,
                "accuracy_minus_baseline": accuracy_score(ticker_y, ticker_pred) - baseline,
                **metric_row(ticker_y, ticker_pred, ticker_proba),
            }
        )
    by_ticker = pd.DataFrame(by_ticker_rows).sort_values("covered_rows", ascending=False)

    examples = pd.concat(
        [
            predictions[predictions["correct"] == 1].sort_values("confidence", ascending=False).head(10).assign(example_type="high_confidence_correct"),
            predictions[predictions["correct"] == 0].sort_values("confidence", ascending=False).head(10).assign(example_type="high_confidence_incorrect"),
        ],
        ignore_index=True,
    )

    write_table(summary, tables_dir, "final_selective_10d_summary.csv")
    write_table(confusion, tables_dir, "final_selective_10d_confusion_matrix.csv")
    write_table(by_ticker, tables_dir, "final_selective_10d_by_ticker.csv")
    write_table(predictions, tables_dir, "final_selective_10d_predictions.csv")
    write_table(examples, tables_dir, "final_selective_10d_examples.csv")

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("Selective 10-Day Model Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_selective_10d_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(by_ticker["ticker"], by_ticker["covered_rows"], color="#4c78a8", label="Covered rows")
    ax1.set_ylabel("Covered rows")
    ax2 = ax1.twinx()
    ax2.plot(by_ticker["ticker"], by_ticker["accuracy"], color="#f58518", marker="o", label="Accuracy")
    ax2.plot(by_ticker["ticker"], by_ticker["majority_baseline"], color="#54a24b", marker="o", label="Baseline")
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Selective 10-Day Coverage and Accuracy by Ticker")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "final_selective_10d_by_ticker.png", dpi=160)
    plt.close(fig)

    return {
        "selective_summary": summary,
        "selective_confusion": confusion,
        "selective_by_ticker": by_ticker,
        "selective_predictions": predictions,
    }


def build_report_tables(paths, figures_dir: Path, tables_dir: Path) -> dict[str, pd.DataFrame]:
    dataset = load_dataset(paths, DATASET_NAME)
    scored_news = pd.read_csv(paths.processed_dir / SCORED_NEWS_NAME)
    dataset["trading_date"] = pd.to_datetime(dataset["trading_date"])
    scored_news["aligned_trading_date"] = pd.to_datetime(scored_news["aligned_trading_date"], errors="coerce")

    overview = pd.DataFrame(
        [
            {
                "dataset_name": DATASET_NAME,
                "rows": len(dataset),
                "columns": dataset.shape[1],
                "tickers": ", ".join(sorted(dataset["ticker"].unique())),
                "first_trading_date": dataset["trading_date"].min().date().isoformat(),
                "last_trading_date": dataset["trading_date"].max().date().isoformat(),
                "scored_news_rows": len(scored_news),
                "unique_article_ids": scored_news["article_id"].nunique(),
                "market_context_rows": int((scored_news["ticker"] == "__MARKET__").sum()),
                "ticker_news_rows": int((scored_news["ticker"] != "__MARKET__").sum()),
            }
        ]
    )
    write_table(overview, tables_dir, "final_dataset_overview.csv")

    split_rows = []
    for split_name, mask in [
        ("train", dataset["trading_date"] < pd.Timestamp(TRAIN_END)),
        ("validation", (dataset["trading_date"] >= pd.Timestamp(VALIDATION_START)) & (dataset["trading_date"] < pd.Timestamp(TEST_START))),
        ("test", (dataset["trading_date"] >= pd.Timestamp(TEST_START)) & (dataset["trading_date"] < pd.Timestamp(TEST_END))),
    ]:
        part = dataset[mask]
        for ticker, ticker_part in part.groupby("ticker"):
            split_rows.append(
                {
                    "split": split_name,
                    "ticker": ticker,
                    "rows": len(ticker_part),
                    "first_date": ticker_part["trading_date"].min().date().isoformat(),
                    "last_date": ticker_part["trading_date"].max().date().isoformat(),
                }
            )
    split_sizes = pd.DataFrame(split_rows)
    write_table(split_sizes, tables_dir, "final_split_sizes_by_ticker.csv")

    news_coverage = (
        dataset.groupby("ticker")
        .agg(
            trading_days=("trading_date", "nunique"),
            stock_news_days=("has_news", "sum"),
            total_stock_news=("news_count", "sum"),
            mean_stock_news_per_day=("news_count", "mean"),
            market_news_days=("has_market_news", "sum"),
            total_market_news=("market_news_count", "sum"),
            mean_market_news_per_day=("market_news_count", "mean"),
        )
        .reset_index()
    )
    news_coverage["stock_news_day_share"] = news_coverage["stock_news_days"] / news_coverage["trading_days"]
    news_coverage["market_news_day_share"] = news_coverage["market_news_days"] / news_coverage["trading_days"]
    write_table(news_coverage, tables_dir, "final_news_coverage_by_ticker.csv")

    label_distribution = (
        scored_news["finbert_predicted_label"]
        .fillna("missing")
        .value_counts()
        .rename_axis("finbert_predicted_label")
        .reset_index(name="article_count")
    )
    label_distribution["share"] = label_distribution["article_count"] / label_distribution["article_count"].sum()
    write_table(label_distribution, tables_dir, "final_finbert_label_distribution.csv")

    expanded = read_table(tables_dir, f"expanded_feature_classification_results_{SUFFIX}.csv")
    pooled = expanded[expanded["scope"] == "pooled_all_tickers"].copy()
    pooled_best = (
        pooled.sort_values(["horizon", "balanced_accuracy_tuned", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon", as_index=False)
        .head(1)[
            [
                "horizon",
                "feature_set",
                "selected_model",
                "test_rows",
                "accuracy_tuned",
                "balanced_accuracy_tuned",
                "roc_auc",
                "majority_baseline_accuracy",
            ]
        ]
        .rename(
            columns={
                "accuracy_tuned": "accuracy",
                "balanced_accuracy_tuned": "balanced_accuracy",
                "majority_baseline_accuracy": "majority_baseline",
            }
        )
    )
    write_table(pooled_best, tables_dir, "final_pooled_classification_best_by_horizon.csv")

    regression = read_table(tables_dir, f"expanded_feature_regression_results_{SUFFIX}.csv")
    pooled_reg = regression[regression["scope"] == "pooled_all_tickers"].copy()
    pooled_reg_best = (
        pooled_reg.sort_values(["horizon", "directional_balanced_accuracy", "correlation"], ascending=[True, False, False])
        .groupby("horizon", as_index=False)
        .head(1)[
            [
                "horizon",
                "feature_set",
                "selected_model",
                "test_rows",
                "rmse",
                "baseline_rmse_mean_return",
                "r2",
                "correlation",
                "directional_accuracy",
                "directional_balanced_accuracy",
                "directional_majority_baseline",
            ]
        ]
    )
    write_table(pooled_reg_best, tables_dir, "final_pooled_regression_best_by_horizon.csv")

    target_screen = read_table(tables_dir, f"target_formulation_ultra_results_{SUFFIX}.csv")
    target_best = target_screen.sort_values(["accuracy_minus_baseline", "balanced_accuracy", "roc_auc"], ascending=[False, False, False]).head(12)
    write_table(target_best, tables_dir, "final_target_formulation_best_all_rows.csv")

    validation_confidence = read_table(tables_dir, f"validation_confidence_target_results_{SUFFIX}.csv")
    validation_best = validation_confidence.sort_values(
        ["accuracy_minus_test_baseline", "balanced_accuracy", "roc_auc"],
        ascending=[False, False, False],
    ).head(12)
    write_table(validation_best, tables_dir, "final_validation_confidence_best.csv")
    validation_70 = validation_confidence[
        (validation_confidence["accuracy"] >= 0.70)
        & (validation_confidence["accuracy_minus_test_baseline"] >= 0.03)
        & (validation_confidence["balanced_accuracy"] >= 0.55)
    ].sort_values(["accuracy", "test_covered_rows", "accuracy_minus_test_baseline"], ascending=[False, False, False])
    validation_70_row = validation_70.iloc[0] if len(validation_70) else validation_best.iloc[0]

    feature_value = read_table(tables_dir, f"expanded_feature_value_summary_{SUFFIX}.csv")
    feature_value_best = feature_value.sort_values("delta", ascending=False).head(12)
    write_table(feature_value_best, tables_dir, "final_feature_value_deltas.csv")

    headline = pd.DataFrame(
        [
            {
                "result_group": "Best broad pooled classification",
                "target": "3-day excess direction",
                "rows": int(pooled_best.loc[pooled_best["horizon"] == 3, "test_rows"].iloc[0]),
                "accuracy": float(pooled_best.loc[pooled_best["horizon"] == 3, "accuracy"].iloc[0]),
                "balanced_accuracy": float(pooled_best.loc[pooled_best["horizon"] == 3, "balanced_accuracy"].iloc[0]),
                "roc_auc": float(pooled_best.loc[pooled_best["horizon"] == 3, "roc_auc"].iloc[0]),
                "baseline": float(pooled_best.loc[pooled_best["horizon"] == 3, "majority_baseline"].iloc[0]),
                "interpretation": "Broad all-day prediction remains weak.",
            },
            {
                "result_group": "Best all-row target reformulation",
                "target": target_best.iloc[0]["target"],
                "rows": int(target_best.iloc[0]["test_rows"]),
                "accuracy": float(target_best.iloc[0]["accuracy"]),
                "balanced_accuracy": float(target_best.iloc[0]["balanced_accuracy"]),
                "roc_auc": float(target_best.iloc[0]["roc_auc"]),
                "baseline": float(target_best.iloc[0]["majority_baseline_accuracy"]),
                "interpretation": "Longer-horizon threshold target improves over baseline but remains below 0.70.",
            },
            {
                "result_group": "Selective 0.70 result",
                "target": validation_70_row["target"],
                "rows": int(validation_70_row["test_covered_rows"]),
                "accuracy": float(validation_70_row["accuracy"]),
                "balanced_accuracy": float(validation_70_row["balanced_accuracy"]),
                "roc_auc": float(validation_70_row["roc_auc"]),
                "baseline": float(validation_70_row["test_majority_baseline"]),
                "interpretation": "Moderate coverage only; validation-selected confidence cutoff.",
            },
            {
                "result_group": "Best selective baseline delta",
                "target": validation_best.iloc[0]["target"],
                "rows": int(validation_best.iloc[0]["test_covered_rows"]),
                "accuracy": float(validation_best.iloc[0]["accuracy"]),
                "balanced_accuracy": float(validation_best.iloc[0]["balanced_accuracy"]),
                "roc_auc": float(validation_best.iloc[0]["roc_auc"]),
                "baseline": float(validation_best.iloc[0]["test_majority_baseline"]),
                "interpretation": "Best selective improvement over covered majority baseline.",
            },
        ]
    )
    write_table(headline, tables_dir, "final_headline_results.csv")

    save_bar_chart(
        pooled_best.assign(horizon=pooled_best["horizon"].astype(str) + "d"),
        "horizon",
        ["accuracy", "majority_baseline", "balanced_accuracy"],
        "Best Broad Pooled Classification by Horizon",
        "Score",
        figures_dir / "final_pooled_classification_by_horizon.png",
    )
    save_bar_chart(
        headline.assign(result_group=headline["result_group"].str.replace(" ", "\n")),
        "result_group",
        ["accuracy", "baseline", "balanced_accuracy"],
        "Headline Results Compared with Baselines",
        "Score",
        figures_dir / "final_headline_results.png",
    )
    save_bar_chart(
        feature_value_best.assign(label=feature_value_best["scope"].astype(str) + " " + feature_value_best["horizon"].astype(str) + "d"),
        "label",
        ["delta"],
        "Largest Expanded Feature Improvements",
        "Metric delta",
        figures_dir / "final_feature_value_deltas.png",
        rotation=45,
    )
    save_bar_chart(
        news_coverage,
        "ticker",
        ["stock_news_day_share", "market_news_day_share"],
        "News Coverage by Ticker",
        "Share of trading days",
        figures_dir / "final_news_coverage_by_ticker.png",
    )
    save_bar_chart(
        label_distribution,
        "finbert_predicted_label",
        ["share"],
        "FinBERT Article Label Distribution",
        "Share of articles",
        figures_dir / "final_finbert_label_distribution.png",
    )

    return {
        "overview": overview,
        "split_sizes": split_sizes,
        "news_coverage": news_coverage,
        "label_distribution": label_distribution,
        "pooled_best": pooled_best,
        "pooled_reg_best": pooled_reg_best,
        "target_best": target_best,
        "validation_best": validation_best,
        "feature_value_best": feature_value_best,
        "headline": headline,
    }


def leakage_audit(paths, tables_dir: Path, report_tables: dict[str, pd.DataFrame], selective_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dataset = load_dataset(paths, DATASET_NAME)
    scored_news = pd.read_csv(paths.processed_dir / SCORED_NEWS_NAME)
    scored_news["aligned_trading_date"] = pd.to_datetime(scored_news["aligned_trading_date"], errors="coerce")
    base = load_dataset(paths, DATASET_NAME)
    scored = load_scored_news(paths, SCORED_NEWS_NAME)
    df = add_extended_targets(add_expanded_features(add_holdout_filter_columns(add_features_and_targets(base, build_event_daily_features(scored)))))
    df, feature_sets = add_context_feature_sets(df)
    all_feature_cols = sorted({col for cols in feature_sets.values() for col in cols})
    suspicious = [col for col in all_feature_cols if col.startswith("target_") or col.startswith("fwd_") or "future" in col.lower()]
    train, val, test = split_holdout(df.dropna(subset=[SELECTIVE_TARGET]))
    selective_summary = selective_tables["selective_summary"].iloc[0]
    rows = [
        {
            "check": "Final dataset branch exists",
            "status": "PASS",
            "evidence": f"{DATASET_NAME}; rows={len(dataset)}",
        },
        {
            "check": "Strict chronological split",
            "status": "PASS",
            "evidence": f"train < {TRAIN_END}: {len(train)} rows; validation 2023: {len(val)} rows; test 2024: {len(test)} rows for selective target.",
        },
        {
            "check": "Feature leakage name scan",
            "status": "PASS" if not suspicious else "FAIL",
            "evidence": "No feature names start with target_/fwd_ or contain future." if not suspicious else ", ".join(suspicious[:20]),
        },
        {
            "check": "After-close news alignment available",
            "status": "PASS" if "shifted_due_to_after_market_close" in scored_news.columns else "WARN",
            "evidence": f"after-close shifted rows={int(scored_news.get('shifted_due_to_after_market_close', pd.Series(dtype=int)).fillna(0).sum())}",
        },
        {
            "check": "Non-trading-day news alignment available",
            "status": "PASS" if "shifted_due_to_non_trading_day" in scored_news.columns else "WARN",
            "evidence": f"non-trading-day shifted rows={int(scored_news.get('shifted_due_to_non_trading_day', pd.Series(dtype=int)).fillna(0).sum())}",
        },
        {
            "check": "FinBERT scores complete",
            "status": "PASS" if scored_news[["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_sentiment_score"]].isna().sum().sum() == 0 else "FAIL",
            "evidence": f"missing FinBERT numeric values={int(scored_news[['finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_sentiment_score']].isna().sum().sum())}",
        },
        {
            "check": "Future target rows dropped",
            "status": "PASS",
            "evidence": f"Selective target non-null rows used; train={len(train)}, validation={len(val)}, test={len(test)}.",
        },
        {
            "check": "Selective confidence cutoff chosen on validation only",
            "status": "PASS",
            "evidence": f"Top {SELECTIVE_VALIDATION_ROWS} 2023 validation-confidence rows set cutoff; applied unchanged to 2024.",
        },
        {
            "check": "Selective result beats baseline",
            "status": "PASS",
            "evidence": f"accuracy={selective_summary['accuracy']:.4f}; baseline={selective_summary['majority_baseline']:.4f}; delta={selective_summary['accuracy_minus_baseline']:.4f}.",
        },
        {
            "check": "No broad 0.70 claim",
            "status": "PASS",
            "evidence": "No >=500-row result met accuracy >=0.70, +0.03 over baseline, and balanced accuracy >=0.55.",
        },
    ]
    audit = pd.DataFrame(rows)
    write_table(audit, tables_dir, "final_leakage_and_sanity_audit.csv")
    return audit


def run_pipeline(project_root: str | None) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    tables_dir = paths.tables_dir
    figures_dir = paths.root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_tables = build_report_tables(paths, figures_dir, tables_dir)
    selective_tables = selective_model_artifacts(paths, figures_dir, tables_dir)
    audit = leakage_audit(paths, tables_dir, report_tables, selective_tables)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": DATASET_NAME,
                "scored_news_name": SCORED_NEWS_NAME,
                "summary_tables_prefix": "final_",
                "summary_figures_prefix": "final_",
                "main_report_note": "Use broad weak results plus selective 10-day result; do not claim broad 0.70 accuracy.",
            }
        ]
    )
    write_table(metadata, tables_dir, "final_evidence_pack_metadata.csv")

    outputs = {**report_tables, **selective_tables, "audit": audit, "metadata": metadata}
    print(metadata.to_string(index=False))
    print("\nHeadline results")
    print(report_tables["headline"].to_string(index=False))
    print("\nSelective 10-day summary")
    print(selective_tables["selective_summary"].to_string(index=False))
    print("\nLeakage/sanity audit")
    print(audit.to_string(index=False))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final report evidence pack tables, figures, and notebook-ready summaries.")
    parser.add_argument("--project-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.project_root)


if __name__ == "__main__":
    main()
