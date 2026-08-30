from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_modelling_baselines import ALL_FINBERT_FEATURES, PRICE_ONLY_FEATURES, build_paths, load_dataset


TARGETS = {
    "next_day_excess_gt_0": ("next_day_excess_return", 0.0),
    "next_day_excess_gt_0_5pct": ("next_day_excess_return", 0.005),
    "next_day_up": ("target_next_day_up", None),
}

FEATURE_SETS = {
    "price_only": PRICE_ONLY_FEATURES,
    "price_plus_finbert_market": PRICE_ONLY_FEATURES + ALL_FINBERT_FEATURES,
}


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "trading_date"]).copy()
    out["next_day_stock_return"] = out.groupby("ticker")["return_1d"].shift(-1)
    out["next_day_spy_return"] = out.groupby("ticker")["spy_return_1d"].shift(-1)
    out["next_day_excess_return"] = out["next_day_stock_return"] - out["next_day_spy_return"]
    return out.dropna(subset=["next_day_stock_return", "next_day_spy_return"]).reset_index(drop=True)


def assign_target(df: pd.DataFrame, target_name: str) -> pd.Series:
    source_col, threshold = TARGETS[target_name]
    if threshold is None:
        return df[source_col].astype(int)
    return (df[source_col] > threshold).astype(int)


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(df["trading_date"].drop_duplicates())
    split_idx = max(1, int(len(dates) * (1 - test_fraction)))
    split_idx = min(split_idx, len(dates) - 1)
    cutoff = dates[split_idx]
    train = df[df["trading_date"] < cutoff].copy()
    test = df[df["trading_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train, test, cutoff


def build_model(model_name: str, feature_cols: list[str]) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42)
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1)
        numeric_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, l2_regularization=0.1, random_state=42)
        numeric_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer([("numeric", Pipeline(numeric_steps), feature_cols)], remainder="drop")),
            ("model", estimator),
        ]
    )


def safe_metrics(y_true: pd.Series, pred: pd.Series, proba: pd.Series) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else float("nan"),
    }
    return metrics


def run_one(
    ticker_df: pd.DataFrame,
    ticker: str,
    target_name: str,
    feature_set_name: str,
    model_name: str,
    test_fraction: float,
) -> dict[str, Any] | None:
    train, test, cutoff = time_split(ticker_df, test_fraction)
    y_train = assign_target(train, target_name)
    y_test = assign_target(test, target_name)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    feature_cols = FEATURE_SETS[feature_set_name]
    model = build_model(model_name, feature_cols)
    model.fit(train[feature_cols], y_train)
    pred = model.predict(test[feature_cols])
    proba = model.predict_proba(test[feature_cols])[:, 1]
    metrics = safe_metrics(y_test, pred, proba)

    return {
        "ticker": ticker,
        "target_name": target_name,
        "feature_set": feature_set_name,
        "model_name": model_name,
        "test_fraction": test_fraction,
        "cutoff_date": cutoff.date().isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positive_rate": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        **metrics,
    }


def build_best_delta_table(results: pd.DataFrame) -> pd.DataFrame:
    best = (
        results.sort_values(["ticker", "target_name", "feature_set", "roc_auc"], ascending=[True, True, True, False])
        .groupby(["ticker", "target_name", "feature_set"], as_index=False)
        .head(1)
    )
    price = best[best["feature_set"] == "price_only"][
        ["ticker", "target_name", "roc_auc", "accuracy", "model_name"]
    ].rename(columns={"roc_auc": "best_price_roc_auc", "accuracy": "best_price_accuracy", "model_name": "best_price_model"})
    news = best[best["feature_set"] == "price_plus_finbert_market"][
        ["ticker", "target_name", "roc_auc", "accuracy", "model_name"]
    ].rename(columns={"roc_auc": "best_news_roc_auc", "accuracy": "best_news_accuracy", "model_name": "best_news_model"})
    merged = price.merge(news, on=["ticker", "target_name"], how="inner")
    merged["roc_auc_delta_news_minus_price"] = merged["best_news_roc_auc"] - merged["best_price_roc_auc"]
    merged["accuracy_delta_news_minus_price"] = merged["best_news_accuracy"] - merged["best_price_accuracy"]
    return merged.sort_values(["target_name", "roc_auc_delta_news_minus_price"], ascending=[True, False])


def run_pipeline(
    project_root: str | None = None,
    dataset_name: str = "model_dataset_finbert_complete.csv",
    output_suffix: str = "",
    test_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    paths = build_paths(project_root)
    df = prepare_dataset(load_dataset(paths, dataset_name))

    rows: list[dict[str, Any]] = []
    for ticker, ticker_df in df.groupby("ticker"):
        ticker_df = ticker_df.sort_values("trading_date").reset_index(drop=True)
        for target_name in TARGETS:
            for feature_set_name in FEATURE_SETS:
                for model_name in ["logistic_regression", "random_forest", "hist_gradient_boosting"]:
                    row = run_one(ticker_df, ticker, target_name, feature_set_name, model_name, test_fraction)
                    if row is not None:
                        rows.append(row)

    results = pd.DataFrame(rows).sort_values(["target_name", "ticker", "roc_auc"], ascending=[True, True, False])
    deltas = build_best_delta_table(results)
    metadata = pd.DataFrame(
        [
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_name": dataset_name,
                "rows": len(df),
                "tickers": ", ".join(sorted(df["ticker"].unique())),
                "test_fraction": test_fraction,
                "output_suffix": output_suffix,
            }
        ]
    )

    suffix = f"_{output_suffix}" if output_suffix else ""
    results.to_csv(paths.tables_dir / f"single_ticker_focus_results{suffix}.csv", index=False)
    deltas.to_csv(paths.tables_dir / f"single_ticker_focus_best_deltas{suffix}.csv", index=False)
    metadata.to_csv(paths.tables_dir / f"single_ticker_focus_metadata{suffix}.csv", index=False)
    return {"results": results, "deltas": deltas, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-ticker focus experiments for each ticker.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-name", default="model_dataset_finbert_complete.csv")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.project_root, args.dataset_name, args.output_suffix, args.test_fraction)
    print(outputs["metadata"].to_string(index=False))
    print()
    print(outputs["deltas"].to_string(index=False))


if __name__ == "__main__":
    main()
