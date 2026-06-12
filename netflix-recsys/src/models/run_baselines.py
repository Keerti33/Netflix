"""Baseline model training & evaluation runner.

Trains all four baseline models on the processed training data, evaluates
them on the test set, saves predictions, and prints a formatted comparison
table showing RMSE, training time, and serialised model size.

Usage
-----
From the project root::

    python src/models/run_baselines.py

    # Custom paths / hyperparameters:
    python src/models/run_baselines.py \\
        --train_path data/processed/train.parquet \\
        --test_path  data/processed/test.parquet  \\
        --output_dir outputs/                     \\
        --k 50 --max_test_rows 200000
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ── make project root importable when run directly ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseRecommender
from src.models.global_mean import BiasModel, GlobalMeanModel
from src.models.item_cf import ItemCFModel
from src.models.user_cf import UserCFModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(train_path: str, test_path: str, max_test_rows: int = 0):
    """Load train and test Parquet files.

    Args:
        train_path: Path to training Parquet.
        test_path:  Path to test Parquet.
        max_test_rows: If > 0, subsample the test set for faster evaluation.

    Returns:
        (train_df, test_df) DataFrames.

    Raises:
        FileNotFoundError: if either file is missing.
    """
    for p in (train_path, test_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Data file not found: {p}\n"
                "Run src/data/preprocess.py first to generate train/test splits."
            )

    print(f"Loading train  : {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"  shape = {train_df.shape}, ratings: {train_df['rating'].mean():.3f} avg")

    print(f"Loading test   : {test_path}")
    test_df = pd.read_parquet(test_path)

    if max_test_rows > 0 and len(test_df) > max_test_rows:
        test_df = test_df.sample(max_test_rows, random_state=42)
        print(f"  sub-sampled test to {len(test_df):,} rows")
    else:
        print(f"  shape = {test_df.shape}")

    return train_df, test_df


def _train_and_eval(
    model: BaseRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path,
    pred_dir: Path,
    k_neighbours: int = 50,
) -> dict:
    """Fit *model*, predict on *test_df*, compute RMSE, save artefacts.

    Args:
        model:        Uninitialised BaseRecommender instance.
        train_df:     Training DataFrame.
        test_df:      Test DataFrame with 'rating' ground truth.
        save_dir:     Directory to save the serialised model.
        pred_dir:     Directory to save test-set predictions.
        k_neighbours: Passed as hint (informational only here since k is set
                      in model __init__).

    Returns:
        dict with keys: name, rmse, train_time, size_mb, n_preds.
    """
    name = model.model_name
    print(f"\n{'-' * 55}")
    print(f"  Model: {name}")
    print(f"{'-' * 55}")

    # ── fit ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        model.fit(train_df)
    except Exception as exc:
        print(f"  [ERROR] fit() failed: {exc}")
        traceback.print_exc()
        return {"name": name, "rmse": float("nan"), "train_time": 0.0,
                "size_mb": 0.0, "n_preds": 0}
    train_time = time.time() - t0

    # ── save model ──────────────────────────────────────────────────────────
    model_path = save_dir / f"{name.lower().replace(' ', '_')}.joblib"
    model.save(model_path)
    size_mb = model_path.stat().st_size / (1024 ** 2)

    # ── predict on test set ─────────────────────────────────────────────────
    print(f"  Predicting on {len(test_df):,} test rows ...")
    t1 = time.time()
    try:
        preds = model.predict_batch(test_df)
    except Exception as exc:
        print(f"  [ERROR] predict_batch() failed: {exc}")
        traceback.print_exc()
        return {"name": name, "rmse": float("nan"), "train_time": train_time,
                "size_mb": size_mb, "n_preds": 0}
    pred_time = time.time() - t1
    print(f"  Prediction took {pred_time:.2f}s")

    # ── RMSE ────────────────────────────────────────────────────────────────
    y_true = test_df["rating"].values.astype(np.float64)
    rmse = BaseRecommender.rmse(y_true, preds)
    print(f"  RMSE = {rmse:.6f}")

    # ── save predictions ────────────────────────────────────────────────────
    pred_df = test_df[["user_id", "movie_id", "rating"]].copy()
    pred_df["predicted_rating"] = preds.astype(np.float32)
    pred_path = pred_dir / f"{name.lower().replace(' ', '_')}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"  Predictions saved -> {pred_path}")

    return {
        "name":       name,
        "rmse":       rmse,
        "train_time": train_time,
        "size_mb":    size_mb,
        "n_preds":    len(preds),
    }


def _print_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: List of result dicts from ``_train_and_eval()``.
    """
    col_widths = [22, 10, 14, 12, 10]
    headers    = ["Model", "RMSE", "Train Time", "Size (MB)", "# Preds"]
    sep        = "-"

    def row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    divider = "-+-".join(sep * w for w in col_widths)

    print(f"\n{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}")
    print("  BASELINE MODEL COMPARISON")
    print(f"{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}")
    print(row(headers))
    print(divider)
    for r in results:
        rmse_str  = f"{r['rmse']:.4f}"  if not np.isnan(r["rmse"]) else "N/A"
        time_str  = f"{r['train_time']:.2f}s"
        size_str  = f"{r['size_mb']:.2f}"
        npred_str = f"{r['n_preds']:,}"
        print(row([r["name"], rmse_str, time_str, size_str, npred_str]))
    print(f"{'=' * sum(col_widths + [3 * (len(col_widths) - 1)])}\n")

    # Best model
    valid = [r for r in results if not np.isnan(r["rmse"])]
    if valid:
        best = min(valid, key=lambda r: r["rmse"])
        print(f"  [BEST] Best RMSE: {best['name']} ({best['rmse']:.4f})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate baseline recommendation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train_path",   default="data/processed/train.parquet")
    p.add_argument("--test_path",    default="data/processed/test.parquet")
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--k",            type=int, default=50,
                   help="Neighbours for UserCF and ItemCF.")
    p.add_argument("--batch_size",   type=int, default=500,
                   help="Batch size for neighbour computation.")
    p.add_argument("--max_test_rows", type=int, default=0,
                   help="Sub-sample test set for faster evaluation (0 = use all).")
    return p


def main() -> None:
    """CLI entry-point."""
    parser = _build_parser()
    args   = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir   = output_dir / "models"
    pred_dir   = output_dir / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    train_df, test_df = _load_data(
        args.train_path, args.test_path, max_test_rows=args.max_test_rows
    )

    # ── define models ────────────────────────────────────────────────────────
    models = [
        GlobalMeanModel(),
        BiasModel(),
        UserCFModel(k=args.k, batch_size=args.batch_size),
        ItemCFModel(k=args.k, batch_size=args.batch_size),
    ]

    # ── train & evaluate each model ──────────────────────────────────────────
    results = []
    for model in models:
        result = _train_and_eval(
            model, train_df, test_df, save_dir, pred_dir, k_neighbours=args.k
        )
        results.append(result)

    # ── comparison table ─────────────────────────────────────────────────────
    _print_table(results)


if __name__ == "__main__":
    main()
