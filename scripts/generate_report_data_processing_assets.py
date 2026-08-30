from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "model_dataset_finbert_quality_complete_2020_2024_polygon_proxy_market_cleaned_news.csv"
OUT_DIR = ROOT / "report" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[object]], align: str, resize: bool = False) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append(r"}")
    lines.extend([r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_num(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}bn"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}m"
    return f"{x:.3f}"


def max_drawdown(adj_close: pd.Series) -> float:
    wealth = adj_close / adj_close.iloc[0]
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def classify_columns(columns: list[str]) -> list[list[str]]:
    categories: list[tuple[str, callable]] = [
        ("Identifiers/raw price", lambda c: c in {"ticker", "trading_date", "open", "high", "low", "close", "adj_close", "volume"}),
        ("Targets", lambda c: c.startswith("target_")),
        ("Price returns and trend", lambda c: c.startswith(("return_", "moving_avg", "ma_", "volatility_", "volume_change", "spy_"))),
        ("Sentiment interactions", lambda c: c.startswith(("sentiment_x_", "market_sentiment_x_"))),
        ("Market-context news", lambda c: c.startswith("market_") or c == "has_market_news"),
        ("Lagged and rolling sentiment", lambda c: c.endswith(("_lag1", "_rolling3", "_rolling5")) or c.endswith("_surprise")),
        ("Quality-weighted news", lambda c: c.startswith("quality_")),
        ("Topic-specific news", lambda c: c.startswith(("earnings_", "analyst_", "product_ai_chips_", "legal_regulation_", "macro_", "deals_", "layoffs_costs_"))),
        ("Base ticker-news sentiment", lambda c: c.startswith("finbert_") or c in {"news_count", "log_news_count", "news_count_above_1", "news_count_above_2", "has_news"}),
    ]
    assigned: set[str] = set()
    rows: list[list[str]] = []
    for name, predicate in categories:
        matched = [c for c in columns if c not in assigned and predicate(c)]
        if matched:
            assigned.update(matched)
            rows.append([name, len(matched), ", ".join(matched)])
    remaining = [c for c in columns if c not in assigned]
    if remaining:
        rows.append(["Other", len(remaining), ", ".join(remaining)])
    return rows


def make_data_snapshot(df: pd.DataFrame) -> None:
    cols = [
        "ticker",
        "trading_date",
        "adj_close",
        "return_1d",
        "volatility_5d",
        "news_count",
        "finbert_sentiment_score_mean",
        "market_news_count",
        "spy_return_1d",
        "quality_source_weighted_sentiment_mean",
    ]
    sample = df.loc[df["ticker"].isin(["AAPL", "TSLA"]), cols].head(8).copy()
    for col in sample.columns:
        if pd.api.types.is_float_dtype(sample[col]):
            sample[col] = sample[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(13.5, 3.2))
    ax.axis("off")
    table = ax.table(cellText=sample.values, colLabels=sample.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_snapshot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_correlation_heatmaps(df: pd.DataFrame) -> None:
    piv = df.pivot(index="trading_date", columns="ticker", values="return_1d").dropna(how="all")
    corr = piv.corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Daily Return Correlation, 2020-2024")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_return_correlation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    target = "target_next_day_up"
    candidates = [
        "return_1d",
        "return_3d",
        "return_5d",
        "volatility_5d",
        "volume_change_1d",
        "news_count",
        "finbert_sentiment_score_mean",
        "finbert_positive_mean",
        "finbert_negative_mean",
        "market_news_count",
        "market_finbert_sentiment_score_mean",
        "quality_source_weighted_sentiment_mean",
        "quality_high_conf_net_count",
        "quality_sentiment_disagreement",
        "spy_return_1d",
        "spy_volatility_5d",
    ]
    corr_rows = []
    for col in candidates:
        valid = df[[col, target]].dropna()
        if valid[col].nunique() > 1:
            corr_rows.append((col, valid[col].corr(valid[target])))
    corr_df = pd.DataFrame(corr_rows, columns=["feature", "correlation"]).sort_values("correlation")

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    colors = ["#b91c1c" if v < 0 else "#1d4ed8" for v in corr_df["correlation"]]
    ax.barh(corr_df["feature"], corr_df["correlation"], color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with next-day-up target")
    ax.set_title("Selected Feature/Target Correlations")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_target_correlations.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    corr_rows = [[r.feature, f"{r.correlation:.3f}"] for r in corr_df.itertuples()]
    write_latex_table(
        OUT_DIR / "feature_target_correlations.tex",
        "Selected feature correlations with the next-day-up target. Values are Pearson correlations over the full modelling table and are used only as descriptive diagnostics, not as model-selection evidence.",
        "tab:feature_target_correlations",
        ["Feature", "Correlation"],
        corr_rows,
        "lr",
    )


def make_price_summary(df: pd.DataFrame) -> None:
    rows = []
    for ticker, group in df.sort_values("trading_date").groupby("ticker"):
        g = group.dropna(subset=["adj_close", "return_1d"])
        total_return = g["adj_close"].iloc[-1] / g["adj_close"].iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / len(g)) - 1
        ann_vol = g["return_1d"].std() * np.sqrt(252)
        rows.append(
            [
                ticker,
                len(g),
                f"{g['trading_date'].min()} to {g['trading_date'].max()}",
                fmt_pct(total_return),
                fmt_pct(ann_return),
                fmt_pct(ann_vol),
                fmt_pct(max_drawdown(g["adj_close"])),
                fmt_num(g["volume"].mean()),
            ]
        )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Price-data diagnostics computed from adjusted close and daily volume. Annualised return and volatility use 252 trading days per year.}",
        r"\label{tab:price_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Ticker & Rows & Date range & Total return & Ann. return & Ann. vol. & Max drawdown & Mean volume \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    (OUT_DIR / "price_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def make_feature_inventory(df: pd.DataFrame) -> None:
    rows = classify_columns(list(df.columns))
    counts = {category: int(count) for category, count, _features in rows}
    summary_rows = [
        [
            "Price and market state",
            counts.get("Price returns and trend", 0),
            "Lagged returns, rolling volatility, volume change, trend measures, and SPY market controls",
            "Baseline signal and control for common market movement",
        ],
        [
            "Ticker-news sentiment",
            counts.get("Base ticker-news sentiment", 0),
            "Daily article volume, FinBERT probability summaries, polarity, confidence, and news-availability indicators",
            "Tests firm-specific public-news information",
        ],
        [
            "Temporal sentiment dynamics",
            counts.get("Lagged and rolling sentiment", 0),
            "Lagged, rolling, and surprise summaries of ticker-level and quality-weighted sentiment",
            "Tests delayed information incorporation",
        ],
        [
            "Market-context news",
            counts.get("Market-context news", 0),
            "Broad market article volume and aggregate market-news sentiment joined to each ticker-day",
            "Captures news that can affect all five equities",
        ],
        [
            "News quality and event structure",
            counts.get("Quality-weighted news", 0) + counts.get("Topic-specific news", 0) + counts.get("Sentiment interactions", 0),
            "Duplicate-aware story counts, source-weighted sentiment, high-confidence sentiment, sentiment interactions, and event categories",
            "Separates information quality and event type from raw article volume",
        ],
        [
            "Audit and target fields",
            counts.get("Identifiers/raw price", 0) + counts.get("Targets", 0),
            "Ticker/date keys, raw OHLCV fields, and target variables retained for traceability",
            "Used for joins, audits, and labels; excluded from predictors where required to avoid leakage",
        ],
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Conceptual feature blocks used in the modelling experiments. The table reports feature families and their modelling role rather than listing implementation column names.}",
        r"\label{tab:feature_groups}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.20\textwidth}p{0.10\textwidth}p{0.34\textwidth}p{0.26\textwidth}}",
        r"\toprule",
        r"Block & Count & Variables represented & Modelling role \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (OUT_DIR / "feature_inventory.tex").write_text("\n".join(lines), encoding="utf-8")


def make_processing_counts(df: pd.DataFrame) -> None:
    rows = [
        ["Ticker-day rows", len(df)],
        ["Tickers", df["ticker"].nunique()],
        ["Trading dates", df["trading_date"].nunique()],
        ["Columns", df.shape[1]],
        ["Rows with ticker-specific news", int(df["has_news"].sum())],
        ["Rows with market-context news", int(df["has_market_news"].sum())],
        ["Mean ticker-news count per row", f"{df['news_count'].mean():.2f}"],
        ["Mean market-news count per row", f"{df['market_news_count'].mean():.2f}"],
        ["Mean one-day return", fmt_pct(df["return_1d"].mean())],
        ["One-day return standard deviation", fmt_pct(df["return_1d"].std())],
    ]
    write_latex_table(
        OUT_DIR / "processing_counts.tex",
        "Final processed dataset checks after joining price, ticker-news, market-news, and feature-engineering outputs.",
        "tab:processing_counts",
        ["Check", "Value"],
        rows,
        "lr",
    )


def make_split_and_target_tables(df: pd.DataFrame) -> None:
    dates = pd.to_datetime(df["trading_date"])
    split = np.select(
        [dates.dt.year <= 2022, dates.dt.year == 2023, dates.dt.year == 2024],
        ["Training", "Validation", "Test"],
        default="Other",
    )
    tmp = df.assign(split=split)
    rows = []
    for name in ["Training", "Validation", "Test"]:
        part = tmp[tmp["split"] == name]
        rows.append(
            [
                name,
                len(part),
                part["trading_date"].min(),
                part["trading_date"].max(),
                fmt_pct(part["target_next_day_up"].mean()),
                fmt_pct(1 - part["target_next_day_up"].mean()),
                f"{part['news_count'].mean():.2f}",
            ]
        )
    write_latex_table(
        OUT_DIR / "split_target_summary.tex",
        "Chronological split diagnostics for the next-day direction target. The positive rate is the fraction of ticker-days where the next adjusted-close move is upward.",
        "tab:split_target_summary",
        ["Split", "Rows", "Start", "End", "Next-day up", "Next-day down", "Mean news"],
        rows,
        "lrrrrrr",
        resize=True,
    )

    rows = []
    for ticker, group in df.groupby("ticker"):
        rows.append(
            [
                ticker,
                len(group),
                fmt_pct(group["target_next_day_up"].mean()),
                int(group["has_news"].sum()),
                fmt_pct(group["has_news"].mean()),
                f"{group['news_count'].mean():.2f}",
                f"{group['market_news_count'].mean():.2f}",
            ]
        )
    write_latex_table(
        OUT_DIR / "ticker_target_news_summary.tex",
        "Ticker-level target balance and news coverage in the final modelling table.",
        "tab:ticker_target_news_summary",
        ["Ticker", "Rows", "Next-day up", "News rows", "News coverage", "Mean ticker news", "Mean market news"],
        rows,
        "lrrrrrr",
        resize=True,
    )


def make_missingness_summary(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    rows = []
    if len(missing) == 0:
        rows.append(["All columns", 0, "0.0%"])
    else:
        for col, count in missing.head(12).items():
            rows.append([col, int(count), fmt_pct(count / len(df))])
    write_latex_table(
        OUT_DIR / "missingness_summary.tex",
        "Missing-value audit after feature joins. Missingness is reported on the full modelling table before horizon-specific target filtering.",
        "tab:missingness_summary",
        ["Column", "Missing rows", "Missing share"],
        rows,
        "lrr",
    )


def make_topic_summary(df: pd.DataFrame) -> None:
    topics = [
        ("Earnings", "earnings_count", "earnings_sentiment_mean"),
        ("Analyst", "analyst_count", "analyst_sentiment_mean"),
        ("Product/AI/chips", "product_ai_chips_count", "product_ai_chips_sentiment_mean"),
        ("Legal/regulation", "legal_regulation_count", "legal_regulation_sentiment_mean"),
        ("Macro", "macro_count", "macro_sentiment_mean"),
        ("Deals", "deals_count", "deals_sentiment_mean"),
        ("Layoffs/costs", "layoffs_costs_count", "layoffs_costs_sentiment_mean"),
    ]
    rows = []
    for label, count_col, sent_col in topics:
        mask = df[count_col] > 0
        weighted_sentiment = np.average(df.loc[mask, sent_col], weights=df.loc[mask, count_col]) if mask.any() else np.nan
        rows.append(
            [
                label,
                int(df[count_col].sum()),
                int(mask.sum()),
                fmt_pct(mask.mean()),
                f"{weighted_sentiment:.3f}" if np.isfinite(weighted_sentiment) else "n/a",
            ]
        )
    write_latex_table(
        OUT_DIR / "topic_summary.tex",
        "Topic-tagged news diagnostics. Article counts are summed across ticker-day rows; sentiment is count-weighted across non-zero topic days.",
        "tab:topic_summary",
        ["Topic", "Article count", "Ticker-days", "Coverage", "Mean sentiment"],
        rows,
        "lrrrr",
    )


def make_return_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for ticker, group in df.groupby("ticker"):
        clipped = group["return_1d"].clip(-0.12, 0.12)
        axes[0].hist(clipped, bins=55, alpha=0.45, label=ticker)
    axes[0].axvline(0, color="#111827", linewidth=0.8)
    axes[0].set_title("Clipped Daily Return Distribution")
    axes[0].set_xlabel("One-day adjusted-close return")
    axes[0].set_ylabel("Ticker-days")
    axes[0].legend(fontsize=8, ncol=3)

    ordered = ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]
    data = [df.loc[df["ticker"] == t, "return_1d"].dropna() * 100 for t in ordered]
    axes[1].boxplot(data, tick_labels=ordered, showfliers=False)
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Daily Return Spread by Ticker")
    axes[1].set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "return_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_news_volume_over_time(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["trading_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month")[["news_count", "market_news_count"]].sum()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(monthly.index, monthly["news_count"], label="Ticker-specific news", color="#1d4ed8", linewidth=1.8)
    ax.plot(monthly.index, monthly["market_news_count"], label="Market-context news", color="#b45309", linewidth=1.8)
    ax.set_title("Monthly News Volume After Trading-Date Alignment")
    ax.set_ylabel("Aggregated article count")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_news_volume.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_sentiment_bucket_table(df: pd.DataFrame) -> None:
    tmp = df[df["has_news"] == 1].copy()
    tmp["sentiment_bucket"] = pd.qcut(
        tmp["finbert_sentiment_score_mean"],
        q=5,
        labels=["Most negative", "Negative", "Middle", "Positive", "Most positive"],
        duplicates="drop",
    )
    rows = []
    for bucket, group in tmp.groupby("sentiment_bucket", observed=True):
        rows.append(
            [
                bucket,
                len(group),
                f"{group['finbert_sentiment_score_mean'].mean():.3f}",
                f"{group['news_count'].mean():.2f}",
                fmt_pct(group["target_next_day_up"].mean()),
                fmt_pct(group["return_1d"].mean()),
            ]
        )
    write_latex_table(
        OUT_DIR / "sentiment_bucket_summary.tex",
        "Ticker-days with news grouped into quintiles by same-day mean FinBERT sentiment score. The next-day-up rate is descriptive and was not used to choose model settings.",
        "tab:sentiment_bucket_summary",
        ["Sentiment bucket", "Rows", "Mean sentiment", "Mean news", "Next-day up", "Same-day return"],
        rows,
        "lrrrrr",
        resize=True,
    )


def main() -> None:
    df = pd.read_csv(DATASET)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.strftime("%Y-%m-%d")
    make_data_snapshot(df)
    make_correlation_heatmaps(df)
    make_price_summary(df)
    make_feature_inventory(df)
    make_processing_counts(df)
    make_split_and_target_tables(df)
    make_missingness_summary(df)
    make_topic_summary(df)
    make_return_distribution(df)
    make_news_volume_over_time(df)
    make_sentiment_bucket_table(df)
    print(f"Wrote data-processing report assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
