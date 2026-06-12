"""Matrix factorization model training & evaluation runner.

Trains SVD (Surprise) and ALS (implicit) models on the processed training
data, evaluates on the test set, generates top-10 recommendations for a
sample of test users, and saves all outputs.

Usage
-----
From the project root::

    python src/models/run_mf.py

    # Disable SVD tuning for a faster run:
    python src/models/run_mf.py --no_tune

    # Custom paths:
    python src/models/run_mf.py \\
        --train_path data/processed/train.parquet \\
        --test_path  data/processed/test.parquet  \\
        --implicit_path data/processed/train_implicit.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# -- make project root importable when run directly --------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseRecommender
from src.models.svd_model import SVDModel
from src.models.als_model import ALSModel
from src.utils.profiling import FitProfiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(
    train_path: str,
    test_path: str,
    implicit_path: str | None = None,
    max_test_rows: int = 0,
):
    """Load train/test DataFrames and optional implicit matrix.

    Args:
        train_path: Path to training Parquet.
        test_path:  Path to test Parquet.
        implicit_path: Path to .npz implicit confidence matrix (item x user).
        max_test_rows: Sub-sample test set (0 = all).

    Returns:
        (train_df, test_df, implicit_matrix_or_None)
    """
    import scipy.sparse as sp

    for p in (train_path, test_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Data file not found: {p}\n"
                "Run src/data/preprocess.py first."
            )

    print(f"Loading train : {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"  shape={train_df.shape}, mean_rating={train_df['rating'].mean():.3f}")

    print(f"Loading test  : {test_path}")
    test_df = pd.read_parquet(test_path)
    if max_test_rows > 0 and len(test_df) > max_test_rows:
        test_df = test_df.sample(max_test_rows, random_state=42)
        print(f"  sub-sampled to {len(test_df):,} rows")
    else:
        print(f"  shape={test_df.shape}")

    impl = None
    if implicit_path and Path(implicit_path).exists():
        impl = sp.load_npz(implicit_path)
        print(f"Loading implicit matrix: shape={impl.shape}, nnz={impl.nnz:,}")

    return train_df, test_df, impl


def _train_and_eval(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path,
    pred_dir: Path,
    fit_kwargs: dict,
) -> dict:
    """Fit, predict, evaluate, and save a single model.

    Args:
        model: Model instance.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        save_dir: Directory for serialised models.
        pred_dir: Directory for prediction parquets.
        fit_kwargs: Extra kwargs passed to model.fit().

    Returns:
        Result dict.
    """
    name = model.model_name
    print(f"\n{'-' * 55}")
    print(f"  Model: {name}")
    print(f"{'-' * 55}")

    # -- fit with profiling --
    with FitProfiler(name, verbose=True) as prof:
        try:
            model.fit(train_df, **fit_kwargs)
        except Exception as exc:
            print(f"  [ERROR] fit() failed: {exc}")
            traceback.print_exc()
            return {"name": name, "rmse": float("nan"), "train_time": 0.0,
                    "peak_ram_mb": 0.0, "size_mb": 0.0, "n_preds": 0}

    # -- save model --
    model_path = save_dir / f"{name.lower().replace(' ', '_')}.joblib"
    model.save(model_path)
    prof.set_model_path(model_path)
    size_mb = model_path.stat().st_size / (1024 ** 2)

    # -- predict on test set --
    print(f"  Predicting on {len(test_df):,} test rows ...")
    t1 = time.time()
    try:
        preds = model.predict_batch(test_df)
    except Exception as exc:
        print(f"  [ERROR] predict_batch() failed: {exc}")
        traceback.print_exc()
        return {"name": name, "rmse": float("nan"),
                "train_time": prof.result.wall_time_s,
                "peak_ram_mb": prof.result.peak_ram_mb,
                "size_mb": size_mb, "n_preds": 0}
    pred_time = time.time() - t1
    print(f"  Prediction took {pred_time:.2f}s")

    # -- RMSE --
    y_true = test_df["rating"].values.astype(np.float64)
    rmse = BaseRecommender.rmse(y_true, preds)
    print(f"  RMSE = {rmse:.6f}")

    # -- save predictions --
    pred_df = test_df[["user_id", "movie_id", "rating"]].copy()
    pred_df["predicted_rating"] = preds.astype(np.float32)
    pred_path = pred_dir / f"{name.lower()}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Predictions saved -> {pred_path}")

    return {
        "name":        name,
        "rmse":        rmse,
        "train_time":  prof.result.wall_time_s,
        "peak_ram_mb": prof.result.peak_ram_mb,
        "size_mb":     size_mb,
        "n_preds":     len(preds),
    }


def _generate_recommendations(
    model: BaseRecommender,
    test_df: pd.DataFrame,
    pred_dir: Path,
    n_sample_users: int = 1000,
    top_k: int = 10,
) -> None:
    """Generate top-K recommendations for a sample of test users.

    Args:
        model: Fitted model with a recommend() method.
        test_df: Test DataFrame (used to pick sample users).
        pred_dir: Output directory.
        n_sample_users: Number of users to sample.
        top_k: Recommendations per user.
    """
    name = model.model_name
    test_users = test_df["user_id"].unique()
    n_sample = min(n_sample_users, len(test_users))
    rng = np.random.default_rng(42)
    sample_users = rng.choice(test_users, size=n_sample, replace=False)

    print(f"  [{name}] Generating top-{top_k} recs for {n_sample} users ...")
    recs_list = []
    for uid in sample_users:
        try:
            recs = model.recommend(int(uid), top_k=top_k)
            for rank, (mid, score) in enumerate(recs, 1):
                recs_list.append({
                    "user_id": int(uid),
                    "rank": rank,
                    "movie_id": mid,
                    "score": score,
                })
        except Exception:
            pass  # skip users that cause errors

    if recs_list:
        recs_df = pd.DataFrame(recs_list)
        recs_path = pred_dir / f"{name.lower()}_recommendations.parquet"
        recs_df.to_parquet(recs_path, index=False)
        print(f"  [{name}] {len(recs_df):,} recs saved -> {recs_path}")
    else:
        print(f"  [{name}] No recommendations generated.")


def _print_table(results: list[dict]) -> None:
    """Print a formatted comparison table.

    Args:
        results: List of result dicts.
    """
    col_widths = [12, 10, 14, 14, 12, 10]
    headers = ["Model", "RMSE", "Train Time", "Peak RAM MB", "Size (MB)", "# Preds"]
    sep = "-"

    def row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    divider = "-+-".join(sep * w for w in col_widths)

    print(f"\n{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}")
    print("  MATRIX FACTORIZATION MODEL COMPARISON")
    print(f"{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}")
    print(row(headers))
    print(divider)
    for r in results:
        rmse_str = f"{r['rmse']:.4f}" if not np.isnan(r["rmse"]) else "N/A*"
        time_str = f"{r['train_time']:.2f}s"
        ram_str = f"{r['peak_ram_mb']:.1f}"
        size_str = f"{r['size_mb']:.2f}"
        npred_str = f"{r['n_preds']:,}"
        print(row([r["name"], rmse_str, time_str, ram_str, size_str, npred_str]))
    print(f"{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}")
    print()
    print("  * ALS RMSE is NOT directly comparable (implicit objective).")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate matrix factorization models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_path", default="data/processed/train.parquet")
    p.add_argument("--test_path", default="data/processed/test.parquet")
    p.add_argument("--implicit_path", default="data/processed/train_implicit.npz")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--no_tune", action="store_true",
                   help="Skip SVD GridSearchCV (use defaults).")
    p.add_argument("--cv_folds", type=int, default=3,
                   help="Number of CV folds for SVD tuning.")
    p.add_argument("--max_test_rows", type=int, default=0,
                   help="Sub-sample test set (0 = all).")
    p.add_argument("--n_sample_users", type=int, default=1000,
                   help="Number of users for recommendation generation.")
    return p


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir = output_dir / "models"
    pred_dir = output_dir / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # -- load data --------------------------------------------------------
    train_df, test_df, impl_matrix = _load_data(
        args.train_path, args.test_path,
        implicit_path=args.implicit_path,
        max_test_rows=args.max_test_rows,
    )

    results = []

    # -- SVD --------------------------------------------------------------
    svd = SVDModel()
    svd_kwargs = {
        "tune": not args.no_tune,
        "cv_folds": args.cv_folds,
        "best_params_path": save_dir / "svd_best_params.json",
    }
    result = _train_and_eval(svd, train_df, test_df, save_dir, pred_dir, svd_kwargs)
    results.append(result)

    if result["n_preds"] > 0:
        _generate_recommendations(svd, test_df, pred_dir,
                                  n_sample_users=args.n_sample_users)

    # -- ALS --------------------------------------------------------------
    als = ALSModel(factors=100, regularization=0.01, iterations=20)
    als_kwargs = {}
    if impl_matrix is not None:
        als_kwargs["implicit_matrix"] = impl_matrix
    else:
        als_kwargs["implicit_matrix_path"] = args.implicit_path

    result = _train_and_eval(als, train_df, test_df, save_dir, pred_dir, als_kwargs)
    results.append(result)

    if result["n_preds"] > 0:
        _generate_recommendations(als, test_df, pred_dir,
                                  n_sample_users=args.n_sample_users)

    # -- comparison table -------------------------------------------------
    _print_table(results)


if __name__ == "__main__":
    main()
