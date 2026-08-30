from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_finbert_sentiment_pipeline import (
    MODEL_NAME,
    build_paths,
    load_existing_finbert_scores,
    score_unique_articles,
)


def run_pipeline(
    project_root: str | None,
    news_input_name: str,
    existing_scored_name: str,
    output_scored_name: str,
    output_missing_scores_name: str,
    missing_articles_output_name: str,
    missing_scores_input_name: str | None,
    prepare_only: bool,
    model_path: str | None,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
) -> dict[str, int | str]:
    paths = build_paths(project_root)
    news_path = paths.processed_dir / news_input_name
    if not news_path.exists():
        raise FileNotFoundError(news_path)

    news = pd.read_csv(news_path)
    news["article_id"] = news["article_id"].astype(str)
    news["text"] = news["text"].fillna("").astype(str)
    unique_articles = news[["article_id", "published_utc", "text"]].drop_duplicates(subset=["article_id"]).copy()

    existing_scores = load_existing_finbert_scores(paths, existing_scored_name)
    scored_ids = set(existing_scores["article_id"].astype(str))
    missing_articles = unique_articles[~unique_articles["article_id"].isin(scored_ids)].copy()
    missing_articles_path = paths.processed_dir / missing_articles_output_name
    missing_articles.to_csv(missing_articles_path, index=False)

    if prepare_only:
        missing_scores = pd.DataFrame(columns=existing_scores.columns)
    elif missing_scores_input_name:
        missing_scores_path = paths.processed_dir / missing_scores_input_name
        if not missing_scores_path.exists():
            raise FileNotFoundError(missing_scores_path)
        missing_scores = pd.read_csv(missing_scores_path)
    elif len(missing_articles):
        model_name = str(Path(model_path).resolve()) if model_path else MODEL_NAME
        missing_scores = score_unique_articles(
            news_target=missing_articles,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=paths.cache_dir,
            local_files_only=local_files_only,
        )
    else:
        missing_scores = pd.DataFrame(columns=existing_scores.columns)

    score_cols = [
        "article_id",
        "finbert_positive",
        "finbert_negative",
        "finbert_neutral",
        "finbert_sentiment_score",
        "finbert_predicted_label",
    ]
    combined_scores = (
        pd.concat([existing_scores[score_cols], missing_scores[score_cols]], ignore_index=True)
        .drop_duplicates(subset=["article_id"], keep="last")
        .copy()
    )
    news_scored = news.merge(combined_scores, on="article_id", how="left", validate="many_to_one")

    missing_output_path = paths.processed_dir / output_missing_scores_name
    scored_output_path = paths.processed_dir / output_scored_name
    missing_scores.to_csv(missing_output_path, index=False)
    news_scored.to_csv(scored_output_path, index=False)

    return {
        "unique_articles": len(unique_articles),
        "existing_score_articles": len(existing_scores),
        "missing_articles_scored": len(missing_scores),
        "combined_score_articles": len(combined_scores),
        "news_rows": len(news),
        "news_rows_without_score": int(news_scored["finbert_sentiment_score"].isna().sum()),
        "missing_articles_path": str(missing_articles_path),
        "missing_output_path": str(missing_output_path),
        "scored_output_path": str(scored_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score only FinBERT articles missing from an existing scored file.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--news-input-name", required=True)
    parser.add_argument("--existing-scored-name", required=True)
    parser.add_argument("--output-scored-name", required=True)
    parser.add_argument("--output-missing-scores-name", required=True)
    parser.add_argument("--missing-articles-output-name", required=True)
    parser.add_argument(
        "--missing-scores-input-name",
        default=None,
        help="Optional CSV of scores produced elsewhere, such as Colab. If supplied, local FinBERT inference is skipped.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only write the missing-articles CSV; do not run local FinBERT.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        project_root=args.project_root,
        news_input_name=args.news_input_name,
        existing_scored_name=args.existing_scored_name,
        output_scored_name=args.output_scored_name,
        output_missing_scores_name=args.output_missing_scores_name,
        missing_articles_output_name=args.missing_articles_output_name,
        missing_scores_input_name=args.missing_scores_input_name,
        prepare_only=args.prepare_only,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
