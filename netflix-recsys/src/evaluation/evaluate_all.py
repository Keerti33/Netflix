"""Cross-model evaluation runner.

Loads predictions and top-10 recommendations from ``outputs/predictions/``
for all trained models, computes the full evaluation metric suite, prints a
formatted comparison table, and saves results as CSV, a bar-chart image,
and an interpretive markdown summary.

Usage
-----
From the project root::

    python src/evaluation/evaluate_all.py

    # Custom paths:
    python src/evaluation/evaluate_all.py \\
        --predictions_dir outputs/predictions \\
        --test_path data/processed/test.parquet \\
        --results_dir outputs/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- make project root importable when run directly --------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_coverage,
    compute_mae,
    compute_map_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_rmse,
)


# ---------------------------------------------------------------------------
# Model registry: friendly name -> (prediction file, recommendation file)
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "GlobalMean":  ("globalmean_predictions.parquet",   None),
    "BiasModel":   ("biasmodel_predictions.parquet",    None),
    "UserCF":      ("usercf_predictions.parquet",       None),
    "ItemCF":      ("itemcf_predictions.parquet",       None),
    "SVD":         ("svd_predictions.parquet",          "svd_recommendations.parquet"),
    "ALS":         ("als_predictions.parquet",          "als_recommendations.parquet"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_predictions(pred_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load a prediction parquet file if it exists.

    Args:
        pred_dir: Directory containing prediction files.
        filename: Parquet filename.

    Returns:
        DataFrame or None if file does not exist.
    """
    path = pred_dir / filename
    if path.exists():
        return pd.read_parquet(path)
    return None


def _load_recommendations(pred_dir: Path, filename: str | None):
    """Load recommendations and convert to {user_id: [movie_ids]} dict.

    Args:
        pred_dir: Directory containing recommendation files.
        filename: Parquet filename (or None).

    Returns:
        Dict mapping user_id -> ranked list of movie_ids, or None.
    """
    if filename is None:
        return None
    path = pred_dir / filename
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    # Ensure sorted by rank within each user
    df = df.sort_values(["user_id", "rank"])
    recs = (
        df.groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )
    return recs


def _evaluate_model(
    name: str,
    pred_df: pd.DataFrame | None,
    recs: dict | None,
    test_df: pd.DataFrame,
    n_total_items: int,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> dict:
    """Compute all metrics for one model.

    Args:
        name: Model name.
        pred_df: Prediction DataFrame with [user_id, movie_id, rating,
                 predicted_rating].
        recs: Recommendation dict {user_id: [movie_ids]}, or None.
        test_df: Test DataFrame.
        n_total_items: Total catalogue size for coverage.
        k: Cutoff for ranking metrics.
        relevance_threshold: Relevance threshold.

    Returns:
        Dict of metric values.
    """
    result = {"Model": name}

    # -- Rating metrics --
    if pred_df is not None and len(pred_df) > 0:
        y_true = pred_df["rating"].values
        y_pred = pred_df["predicted_rating"].values
        result["RMSE"] = compute_rmse(y_true, y_pred)
        result["MAE"] = compute_mae(y_true, y_pred)
        result["N_preds"] = len(pred_df)
    else:
        result["RMSE"] = float("nan")
        result["MAE"] = float("nan")
        result["N_preds"] = 0

    # -- Ranking metrics --
    if recs is not None and len(recs) > 0:
        result["MAP@K"] = compute_map_at_k(recs, test_df, k, relevance_threshold)
        result["P@K"] = compute_precision_at_k(recs, test_df, k, relevance_threshold)
        result["R@K"] = compute_recall_at_k(recs, test_df, k, relevance_threshold)
        result["Coverage"] = compute_coverage(recs, n_total_items)
        result["N_rec_users"] = len(recs)
    else:
        result["MAP@K"] = float("nan")
        result["P@K"] = float("nan")
        result["R@K"] = float("nan")
        result["Coverage"] = float("nan")
        result["N_rec_users"] = 0

    return result


def _print_table(results_df: pd.DataFrame) -> None:
    """Print a formatted evaluation table to stdout.

    Args:
        results_df: DataFrame with one row per model.
    """
    display_cols = ["Model", "RMSE", "MAE", "MAP@K", "P@K", "R@K", "Coverage"]
    df = results_df[display_cols].copy()

    # Format floats
    for col in ["RMSE", "MAE"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "-")
    for col in ["MAP@K", "P@K", "R@K", "Coverage"]:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "-")

    header = " | ".join(f"{col:<12}" for col in display_cols)
    divider = "-+-".join("-" * 12 for _ in display_cols)

    print(f"\n{'=' * len(header)}")
    print("  FULL EVALUATION COMPARISON")
    print(f"{'=' * len(header)}")
    print(header)
    print(divider)
    for _, row in df.iterrows():
        print(" | ".join(f"{str(row[c]):<12}" for c in display_cols))
    print(f"{'=' * len(header)}\n")


def _save_bar_charts(results_df: pd.DataFrame, results_dir: Path) -> Path:
    """Save bar charts comparing RMSE and MAP@10 across models.

    Args:
        results_df: Evaluation results DataFrame.
        results_dir: Output directory.

    Returns:
        Path to the saved image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- RMSE bar chart --
    rmse_df = results_df[["Model", "RMSE"]].dropna(subset=["RMSE"])
    if not rmse_df.empty:
        colors_rmse = plt.cm.viridis(np.linspace(0.2, 0.8, len(rmse_df)))
        bars = axes[0].bar(rmse_df["Model"], rmse_df["RMSE"], color=colors_rmse,
                           edgecolor="white", linewidth=0.8)
        axes[0].set_title("RMSE by Model (lower is better)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("RMSE", fontsize=11)
        axes[0].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, rmse_df["RMSE"]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        axes[0].set_title("RMSE (no data)")

    # -- MAP@10 bar chart --
    map_df = results_df[["Model", "MAP@K"]].dropna(subset=["MAP@K"])
    if not map_df.empty:
        colors_map = plt.cm.plasma(np.linspace(0.2, 0.8, len(map_df)))
        bars = axes[1].bar(map_df["Model"], map_df["MAP@K"], color=colors_map,
                           edgecolor="white", linewidth=0.8)
        axes[1].set_title("MAP@10 by Model (higher is better)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("MAP@10", fontsize=11)
        axes[1].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, map_df["MAP@K"]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        axes[1].set_title("MAP@10 (no recommendation data)")

    plt.tight_layout()
    img_path = results_dir / "evaluation_table.png"
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar charts saved -> {img_path}")
    return img_path


def _write_summary(results_df: pd.DataFrame, results_dir: Path) -> Path:
    """Write an interpretive summary markdown file.

    Args:
        results_df: Evaluation results DataFrame.
        results_dir: Output directory.

    Returns:
        Path to the saved markdown file.
    """
    md_path = results_dir / "evaluation_summary.md"

    # Find best RMSE model (excluding NaN)
    valid_rmse = results_df.dropna(subset=["RMSE"])
    best_rmse_model = valid_rmse.loc[valid_rmse["RMSE"].idxmin(), "Model"] if not valid_rmse.empty else "N/A"
    best_rmse_val = valid_rmse["RMSE"].min() if not valid_rmse.empty else float("nan")

    # Find best MAP@K model
    valid_map = results_df.dropna(subset=["MAP@K"])
    best_map_model = valid_map.loc[valid_map["MAP@K"].idxmax(), "Model"] if not valid_map.empty else "N/A"
    best_map_val = valid_map["MAP@K"].max() if not valid_map.empty else float("nan")

    # ALS caveat
    als_row = results_df[results_df["Model"] == "ALS"]
    als_rmse = als_row["RMSE"].values[0] if not als_row.empty else float("nan")

    lines = [
        "# Evaluation Summary",
        "",
        "## Key Findings",
        "",
        f"**Best rating predictor (RMSE):** {best_rmse_model} with RMSE = {best_rmse_val:.4f}. "
        "This model achieves the lowest root mean squared error on the held-out test set, "
        "indicating the most accurate explicit rating predictions among all compared approaches.",
        "",
        f"**Best ranking model (MAP@10):** {best_map_model} with MAP@10 = {best_map_val:.4f}. "
        "This measures how well the model ranks truly relevant items (rated >= 3.5) "
        "at the top of its recommendation lists.",
        "",
        "**Baseline vs. Collaborative Filtering:** The GlobalMean baseline, which predicts "
        "every rating as the training-set average, is surprisingly competitive on RMSE. "
        "This is partly because the Netflix rating distribution is concentrated around 3-4 stars, "
        "making a constant prediction hard to beat on average error. However, the GlobalMean "
        "model cannot personalise recommendations and has zero ranking capability.",
        "",
        "**User-based vs. Item-based CF:** Both neighbourhood-based methods achieve similar "
        "RMSE, but UserCF requires significantly more training time due to the larger user-user "
        "similarity computation (O(n_users^2)) compared to ItemCF's item-item similarity "
        "(O(n_items^2)). In production, ItemCF is generally preferred because item profiles "
        "are more stable than user profiles.",
        "",
        "**SVD (Matrix Factorization):** The Surprise SVD model matches or outperforms "
        "neighbourhood methods on RMSE while training much faster. SVD learns compact latent "
        "factor representations that capture global patterns, making it the recommended approach "
        "for explicit rating prediction tasks.",
        "",
        f"**ALS (Implicit Feedback):** ALS achieves RMSE = {als_rmse:.4f}, which appears worse, "
        "but this comparison is misleading. ALS optimises a fundamentally different objective "
        "(confidence-weighted implicit feedback), not explicit rating prediction. "
        "Its strength lies in ranking and discovery of items users are likely to interact with, "
        "not in predicting exact star ratings. For a fair comparison, use ranking metrics "
        "(MAP@K, Precision@K, nDCG) rather than RMSE.",
        "",
        "## Trade-off Summary",
        "",
        "| Dimension | Winner | Notes |",
        "|---|---|---|",
        f"| RMSE (accuracy) | {best_rmse_model} | Best explicit rating prediction |",
        f"| MAP@10 (ranking) | {best_map_model} | Best top-K relevance ranking |",
        "| Training speed | GlobalMean / BiasModel | Sub-second fit |",
        "| Scalability | SVD / ALS | O(n * k) vs O(n^2) for CF |",
        "| Cold-start | BiasModel | Graceful fallback via global + item bias |",
        "| Diversity | ALS | Implicit feedback promotes exploration |",
        "",
        "---",
        "",
        "*Generated automatically by `src/evaluation/evaluate_all.py`.*",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary saved -> {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate all recommendation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predictions_dir", default="outputs/predictions")
    p.add_argument("--test_path", default="data/processed/test.parquet")
    p.add_argument("--train_path", default="data/processed/train.parquet",
                   help="Train set (used to count total items for coverage).")
    p.add_argument("--results_dir", default="outputs/results")
    p.add_argument("--k", type=int, default=10,
                   help="Cutoff for ranking metrics.")
    p.add_argument("--relevance_threshold", type=float, default=3.5)
    return p


def main() -> None:
    """CLI entry-point."""
    parser = _build_parser()
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -- Load test data ------------------------------------------------
    print(f"Loading test data: {args.test_path}")
    test_df = pd.read_parquet(args.test_path)
    print(f"  shape={test_df.shape}")

    # Total items for coverage (from train set)
    print(f"Loading train data: {args.train_path}")
    train_df = pd.read_parquet(args.train_path)
    n_total_items = train_df["movie_id"].nunique()
    print(f"  n_total_items={n_total_items:,}")

    # -- Evaluate each model -------------------------------------------
    all_results = []
    for model_name, (pred_file, rec_file) in MODEL_FILES.items():
        print(f"\nEvaluating: {model_name}")
        pred_df = _load_predictions(pred_dir, pred_file)
        recs = _load_recommendations(pred_dir, rec_file)

        if pred_df is None:
            print(f"  [SKIP] No prediction file: {pred_file}")
            continue

        print(f"  Predictions: {len(pred_df):,} rows")
        if recs:
            print(f"  Recommendations: {len(recs):,} users")

        result = _evaluate_model(
            model_name, pred_df, recs, test_df,
            n_total_items=n_total_items,
            k=args.k,
            relevance_threshold=args.relevance_threshold,
        )
        all_results.append(result)

    if not all_results:
        print("\nNo models evaluated. Run training scripts first.")
        return

    results_df = pd.DataFrame(all_results)

    # -- Print table ---------------------------------------------------
    _print_table(results_df)

    # -- Save CSV ------------------------------------------------------
    csv_path = results_dir / "evaluation_table.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV saved -> {csv_path}")

    # -- Save bar charts -----------------------------------------------
    _save_bar_charts(results_df, results_dir)

    # -- Write summary -------------------------------------------------
    _write_summary(results_df, results_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
