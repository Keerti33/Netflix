"""Success/failure case analysis for recommendations.

Analyses recommendation quality by examining individual user cases,
diagnosing failure patterns, and performing era/genre overlap analysis.

Usage::

    python src/recommend/analysis.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import _build_relevant_sets


# ---------------------------------------------------------------------------
# Per-user AP@K
# ---------------------------------------------------------------------------

def compute_per_user_ap(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> Dict[int, float]:
    """Compute AP@K for each user individually.

    Args:
        recommendations: {user_id: [ranked movie_ids]}.
        test_df: Test DataFrame.
        k: Cutoff.
        relevance_threshold: Relevance threshold.

    Returns:
        Dict mapping user_id -> AP@K score.
    """
    relevant_sets = _build_relevant_sets(test_df, relevance_threshold)
    user_ap = {}

    for uid, rec_list in recommendations.items():
        rel_set = relevant_sets.get(uid)
        if not rel_set:
            user_ap[uid] = 0.0
            continue
        R = len(rel_set)
        hits = 0
        ap_sum = 0.0
        for i, mid in enumerate(rec_list[:k], 1):
            if mid in rel_set:
                hits += 1
                ap_sum += hits / i
        user_ap[uid] = ap_sum / R

    return user_ap


def _get_decade(year_str) -> str:
    """Convert a year string to decade label."""
    try:
        y = int(str(year_str).strip())
        return f"{(y // 10) * 10}s"
    except (ValueError, TypeError):
        return "Unknown"


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def success_case_analysis(
    user_ap: Dict[int, float],
    train_df: pd.DataFrame,
    recs_dict: Dict[int, List[int]],
    movies_df: pd.DataFrame,
    n_cases: int = 5,
) -> str:
    """Analyse users where recommendations worked well (AP@K > 0.5).

    Args:
        user_ap: Per-user AP@K scores.
        train_df: Training DataFrame.
        recs_dict: Recommendations dict.
        movies_df: Movie metadata.
        n_cases: Number of cases to show.

    Returns:
        Markdown string with analysis.
    """
    title_map = movies_df.set_index("movie_id")["title"].to_dict()
    year_map = movies_df.set_index("movie_id")["year"].to_dict()

    success_users = sorted(
        [(uid, ap) for uid, ap in user_ap.items() if ap > 0.5],
        key=lambda x: -x[1],
    )[:n_cases]

    if not success_users:
        success_users = sorted(
            [(uid, ap) for uid, ap in user_ap.items() if ap > 0],
            key=lambda x: -x[1],
        )[:n_cases]

    lines = ["## Success Cases", ""]
    if not success_users:
        lines.append("No users with AP@10 > 0 found.")
        return "\n".join(lines)

    lines.append(f"Users with highest AP@10 (top {len(success_users)}):")
    lines.append("")

    for uid, ap in success_users:
        lines.append(f"### User {uid} (AP@10 = {ap:.4f})")
        lines.append("")

        # Top-5 rated in train
        user_train = train_df[train_df["user_id"] == uid].sort_values(
            "rating", ascending=False
        ).head(5)
        lines.append("**Top-5 rated movies (training):**")
        lines.append("")
        lines.append("| Movie ID | Title | Year | Rating |")
        lines.append("|---|---|---|---|")
        for _, row in user_train.iterrows():
            mid = int(row["movie_id"])
            t = title_map.get(mid, "Unknown")
            y = year_map.get(mid, "N/A")
            lines.append(f"| {mid} | {t} | {y} | {row['rating']:.0f} |")
        lines.append("")

        # Top-10 recommendations
        recs = recs_dict.get(uid, [])[:10]
        lines.append("**Top-10 recommendations:**")
        lines.append("")
        lines.append("| Rank | Movie ID | Title | Year |")
        lines.append("|---|---|---|---|")
        for rank, mid in enumerate(recs, 1):
            t = title_map.get(mid, "Unknown")
            y = year_map.get(mid, "N/A")
            lines.append(f"| {rank} | {mid} | {t} | {y} |")
        lines.append("")

        # Era overlap
        train_decades = {_get_decade(year_map.get(int(r["movie_id"]), ""))
                         for _, r in user_train.iterrows()}
        rec_decades = {_get_decade(year_map.get(m, "")) for m in recs}
        overlap = train_decades & rec_decades - {"Unknown"}
        if overlap:
            lines.append(f"**Era overlap:** Shared decades: {', '.join(sorted(overlap))}. "
                         "The model captures this user's era preferences well.")
        lines.append("")

    return "\n".join(lines)


def failure_case_analysis(
    user_ap: Dict[int, float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    recs_dict: Dict[int, List[int]],
    movies_df: pd.DataFrame,
    n_cases: int = 5,
) -> str:
    """Analyse users where recommendations failed (AP@K = 0).

    Diagnoses whether failures are due to cold-start, power-user effects,
    or niche tastes.

    Args:
        user_ap: Per-user AP@K scores.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        recs_dict: Recommendations dict.
        movies_df: Movie metadata.
        n_cases: Number of cases.

    Returns:
        Markdown string with diagnosis.
    """
    title_map = movies_df.set_index("movie_id")["title"].to_dict()
    year_map = movies_df.set_index("movie_id")["year"].to_dict()

    failure_users = [uid for uid, ap in user_ap.items() if ap == 0.0]
    rng = np.random.default_rng(42)
    if len(failure_users) > n_cases:
        failure_users = rng.choice(failure_users, size=n_cases, replace=False).tolist()

    train_counts = train_df.groupby("user_id").size()
    global_median = train_counts.median()

    lines = ["## Failure Cases", ""]
    if not failure_users:
        lines.append("No failure cases (AP@10 = 0) found.")
        return "\n".join(lines)

    lines.append(f"Users with AP@10 = 0.0 ({len(failure_users)} sampled):")
    lines.append("")

    for uid in failure_users:
        n_train = int(train_counts.get(uid, 0))
        n_test = len(test_df[test_df["user_id"] == uid])

        # Classify user type
        if n_train == 0:
            user_type = "Cold-start user (0 training ratings)"
        elif n_train < 5:
            user_type = f"Near cold-start ({n_train} training ratings)"
        elif n_train > global_median * 3:
            user_type = f"Power user ({n_train} training ratings, median={global_median:.0f})"
        else:
            user_type = f"Regular user ({n_train} training ratings)"

        # Check taste diversity
        user_train = train_df[train_df["user_id"] == uid]
        if len(user_train) > 0:
            rating_std = user_train["rating"].std()
            mean_rating = user_train["rating"].mean()
            decades = {_get_decade(year_map.get(int(r["movie_id"]), ""))
                       for _, r in user_train.iterrows()} - {"Unknown"}
            if rating_std < 0.5 and mean_rating > 4.0:
                taste_note = "Uniformly high rater (low variance) -- hard to distinguish preferences."
            elif len(decades) > 4:
                taste_note = f"Eclectic taste spanning {len(decades)} decades -- niche/diverse."
            else:
                taste_note = "Standard taste profile."
        else:
            taste_note = "No training data."

        lines.append(f"### User {uid}")
        lines.append(f"- **Type:** {user_type}")
        lines.append(f"- **Test items:** {n_test}")
        lines.append(f"- **Diagnosis:** {taste_note}")
        lines.append("")

    return "\n".join(lines)


def era_analysis(
    recs_dict: Dict[int, List[int]],
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_sample: int = 200,
) -> str:
    """Analyse decade overlap between rated and recommended movies.

    Args:
        recs_dict: Recommendations dict.
        train_df: Training DataFrame.
        movies_df: Movie metadata.
        n_sample: Number of users to sample.

    Returns:
        Markdown string with era analysis.
    """
    year_map = movies_df.set_index("movie_id")["year"].to_dict()

    users = list(recs_dict.keys())
    rng = np.random.default_rng(42)
    if len(users) > n_sample:
        users = rng.choice(users, size=n_sample, replace=False).tolist()

    overlap_fractions = []

    for uid in users:
        user_train = train_df[train_df["user_id"] == uid]
        if len(user_train) == 0:
            continue

        top_rated = user_train.sort_values("rating", ascending=False).head(10)
        rated_decades = {_get_decade(year_map.get(int(r["movie_id"]), ""))
                         for _, r in top_rated.iterrows()} - {"Unknown"}

        rec_movies = recs_dict.get(uid, [])[:10]
        rec_decades = {_get_decade(year_map.get(m, ""))
                       for m in rec_movies} - {"Unknown"}

        if rated_decades:
            overlap = len(rated_decades & rec_decades) / len(rated_decades)
            overlap_fractions.append(overlap)

    lines = ["## Era / Decade Analysis", ""]

    if overlap_fractions:
        mean_overlap = np.mean(overlap_fractions)
        lines.append(
            f"Across {len(overlap_fractions)} users, the average decade overlap "
            f"between a user's top-10 rated movies and their top-10 recommendations "
            f"is **{mean_overlap:.1%}**."
        )
        lines.append("")
        lines.append(
            "This means the model tends to recommend movies from similar eras "
            "as the user's favourites, suggesting it captures temporal taste patterns."
            if mean_overlap > 0.3 else
            "The model recommends across a wider range of eras than the user's "
            "favourites, which may indicate good diversity but also potential "
            "misalignment with temporal preferences."
        )
    else:
        lines.append("Insufficient data for era analysis.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full recommendation analysis and save markdown report."""
    parser = argparse.ArgumentParser(
        description="Recommendation success/failure analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_dir", default="outputs/predictions")
    parser.add_argument("--train_path", default="data/processed/train.parquet")
    parser.add_argument("--test_path", default="data/processed/test.parquet")
    parser.add_argument("--movies_path", default="data/processed/movies.parquet")
    parser.add_argument("--output_dir", default="outputs/analysis")
    parser.add_argument("--rec_file", default="svd_recommendations.parquet",
                        help="Recommendations file to analyse.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data ...")
    train_df = pd.read_parquet(args.train_path)
    test_df = pd.read_parquet(args.test_path)
    movies_df = pd.read_parquet(args.movies_path)

    # Load recommendations
    rec_path = Path(args.predictions_dir) / args.rec_file
    if not rec_path.exists():
        print(f"Recommendations file not found: {rec_path}")
        print("Run run_mf.py first to generate recommendations.")
        return

    rec_df = pd.read_parquet(rec_path)
    rec_df = rec_df.sort_values(["user_id", "rank"])
    recs_dict: Dict[int, List[int]] = (
        rec_df.groupby("user_id")["movie_id"].apply(list).to_dict()
    )
    print(f"  Loaded recs for {len(recs_dict)} users")

    # Compute per-user AP@K
    print("Computing per-user AP@10 ...")
    user_ap = compute_per_user_ap(recs_dict, test_df, k=10, relevance_threshold=3.5)

    n_zero = sum(1 for v in user_ap.values() if v == 0.0)
    n_pos = sum(1 for v in user_ap.values() if v > 0.0)
    print(f"  AP@10 > 0: {n_pos} users | AP@10 = 0: {n_zero} users")

    # Build report
    report_parts = [
        "# Recommendation Analysis Report",
        "",
        f"Model: SVD | Users analysed: {len(recs_dict)} | "
        f"AP@10 > 0: {n_pos} | AP@10 = 0: {n_zero}",
        "",
        "---",
        "",
    ]

    print("Analysing success cases ...")
    report_parts.append(success_case_analysis(
        user_ap, train_df, recs_dict, movies_df, n_cases=5,
    ))

    print("Analysing failure cases ...")
    report_parts.append(failure_case_analysis(
        user_ap, train_df, test_df, recs_dict, movies_df, n_cases=5,
    ))

    print("Running era analysis ...")
    report_parts.append(era_analysis(recs_dict, train_df, movies_df))

    # Save
    report = "\n".join(report_parts)
    md_path = output_dir / "rec_analysis.md"
    md_path.write_text(report, encoding="utf-8")
    print(f"\nAnalysis saved -> {md_path}")


if __name__ == "__main__":
    main()
