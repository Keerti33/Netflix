"""Data preprocessing module for Netflix Prize Recommendation System.

Prepares all data structures needed for modelling:
  - Temporal or random train/test splits (saved as Parquet)
  - Contiguous 0-indexed user/movie ID remapping (saved as JSON)
  - scipy.sparse CSR matrix of the train set (saved as .npz)
  - surprise.Dataset builder (rating_scale=(1,5))
  - Implicit ALS confidence matrix builder (C = 1 + alpha * rating)
  - Cold-start user / cold-start movie CSV exports

Run directly::

    python src/data/preprocess.py --help

or import individual functions from other modules::

    from src.data.preprocess import build_csr_matrix, build_surprise_dataset
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# 1. Train / Test split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    cutoff: str = "2005-10-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test by a calendar cutoff date.

    All ratings whose ``date`` is strictly before *cutoff* go to train;
    ratings on or after *cutoff* go to test.  This prevents any future
    information leaking into the model during training.

    Args:
        df: Full ratings DataFrame with at least columns
            [user_id, movie_id, rating, date].
        cutoff: ISO-format date string (YYYY-MM-DD).  Default ``"2005-10-01"``.

    Returns:
        (train_df, test_df) tuple of DataFrames.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df["date"] < cutoff_ts].copy()
    test = df[df["date"] >= cutoff_ts].copy()
    print(
        f"[temporal_split] cutoff={cutoff} | "
        f"train={len(train):,} rows, test={len(test):,} rows"
    )
    return train, test


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings randomly into train/test.

    Args:
        df: Full ratings DataFrame.
        test_size: Fraction of data to put in the test set.  Default ``0.2``.
        random_state: Numpy random seed for reproducibility.  Default ``42``.

    Returns:
        (train_df, test_df) tuple of DataFrames.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(df))
    n_test = int(np.ceil(len(df) * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    print(
        f"[random_split] test_size={test_size} | "
        f"train={len(train):,} rows, test={len(test):,} rows"
    )
    return train, test


def split_dataset(
    df: pd.DataFrame,
    split: str = "temporal",
    cutoff: str = "2005-10-01",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch to the requested split strategy.

    Args:
        df: Full ratings DataFrame.
        split: One of ``"temporal"`` (default) or ``"random"``.
        cutoff: ISO date cutoff for the temporal split.
        test_size: Test fraction for the random split.
        random_state: RNG seed for the random split.

    Returns:
        (train_df, test_df) tuple.

    Raises:
        ValueError: If *split* is not one of the supported strategies.
    """
    if split == "temporal":
        return temporal_split(df, cutoff=cutoff)
    elif split == "random":
        return random_split(df, test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown split strategy: {split!r}. Use 'temporal' or 'random'.")


# ---------------------------------------------------------------------------
# 2. User / item ID remapping
# ---------------------------------------------------------------------------

def build_id_mappings(
    train: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Create contiguous 0-indexed integer mappings for users and movies.

    Mappings are derived exclusively from the *train* set so that the model
    never encodes information about unseen test users or items.

    Args:
        train: Training ratings DataFrame.

    Returns:
        A 4-tuple ``(user2idx, idx2user, movie2idx, idx2movie)`` where each
        element is a plain Python ``dict`` with ``int`` keys and values.
    """
    unique_users = sorted(train["user_id"].unique().tolist())
    unique_movies = sorted(train["movie_id"].unique().tolist())

    user2idx: Dict[int, int] = {u: i for i, u in enumerate(unique_users)}
    idx2user: Dict[int, int] = {i: u for u, i in user2idx.items()}
    movie2idx: Dict[int, int] = {m: i for i, m in enumerate(unique_movies)}
    idx2movie: Dict[int, int] = {i: m for m, i in movie2idx.items()}

    print(
        f"[build_id_mappings] n_users={len(user2idx):,}, "
        f"n_movies={len(movie2idx):,}"
    )
    return user2idx, idx2user, movie2idx, idx2movie


def apply_id_mappings(
    df: pd.DataFrame,
    user2idx: Dict[int, int],
    movie2idx: Dict[int, int],
) -> pd.DataFrame:
    """Map raw user/movie IDs to 0-indexed integers in *df*.

    Rows whose user_id or movie_id is not present in the mapping dicts
    (i.e. cold-start items) receive ``NaN`` and are **dropped** from the
    returned DataFrame.  This keeps matrices well-formed.

    Args:
        df: Ratings DataFrame with original user_id / movie_id columns.
        user2idx: Mapping from raw user_id → integer index.
        movie2idx: Mapping from raw movie_id → integer index.

    Returns:
        A copy of *df* with ``user_idx`` and ``movie_idx`` columns added and
        rows with unmapped IDs removed.
    """
    out = df.copy()
    out["user_idx"] = out["user_id"].map(user2idx)
    out["movie_idx"] = out["movie_id"].map(movie2idx)
    before = len(out)
    out = out.dropna(subset=["user_idx", "movie_idx"])
    out["user_idx"] = out["user_idx"].astype(int)
    out["movie_idx"] = out["movie_idx"].astype(int)
    dropped = before - len(out)
    if dropped:
        print(f"[apply_id_mappings] dropped {dropped:,} rows (unmapped IDs)")
    return out


def save_mappings(
    user2idx: Dict[int, int],
    idx2user: Dict[int, int],
    movie2idx: Dict[int, int],
    idx2movie: Dict[int, int],
    output_dir: Path,
) -> None:
    """Persist all four mapping dicts to JSON files.

    Args:
        user2idx: user_id → index.
        idx2user: index → user_id.
        movie2idx: movie_id → index.
        idx2movie: index → movie_id.
        output_dir: Directory where JSON files are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(data: dict, filename: str) -> None:
        path = output_dir / filename
        # JSON requires string keys
        with open(path, "w") as fh:
            json.dump({str(k): v for k, v in data.items()}, fh)
        print(f"[save_mappings] saved {filename} ({len(data):,} entries)")

    _save(user2idx, "user2idx.json")
    _save(idx2user, "idx2user.json")
    _save(movie2idx, "movie2idx.json")
    _save(idx2movie, "idx2movie.json")


# ---------------------------------------------------------------------------
# 3. Sparse matrix (scipy CSR)
# ---------------------------------------------------------------------------

def build_csr_matrix(
    train: pd.DataFrame,
    n_users: int,
    n_movies: int,
) -> sp.csr_matrix:
    """Build a scipy CSR rating matrix from the training data.

    Args:
        train: DataFrame with ``user_idx``, ``movie_idx``, ``rating`` columns
               (already remapped to 0-indexed integers).
        n_users: Total number of unique users (matrix row count).
        n_movies: Total number of unique movies (matrix column count).

    Returns:
        ``scipy.sparse.csr_matrix`` of shape ``(n_users, n_movies)`` with
        float32 rating values.
    """
    rows = train["user_idx"].values
    cols = train["movie_idx"].values
    data = train["rating"].values.astype(np.float32)

    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    print(
        f"[build_csr_matrix] shape={matrix.shape}, "
        f"nnz={matrix.nnz:,}, "
        f"density={matrix.nnz / (n_users * n_movies):.6f}"
    )
    return matrix


# ---------------------------------------------------------------------------
# 4. Surprise dataset builder
# ---------------------------------------------------------------------------

def build_surprise_dataset(train: pd.DataFrame):
    """Build a Surprise Dataset from the training DataFrame.

    Uses ``surprise.Reader`` with ``rating_scale=(1, 5)`` to tell Surprise
    the valid rating range.

    Args:
        train: Training DataFrame with columns ``user_id``, ``movie_id``,
               ``rating``.  Raw (non-remapped) IDs are fine here – Surprise
               handles its own internal encoding.

    Returns:
        A ``surprise.Dataset`` object ready to be passed to Surprise
        cross-validation helpers or ``build_full_trainset()``.

    Raises:
        ImportError: If the ``scikit-surprise`` package is not installed.
    """
    try:
        from surprise import Dataset, Reader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scikit-surprise is required for build_surprise_dataset(). "
            "Install it with: pip install scikit-surprise"
        ) from exc

    reader = Reader(rating_scale=(1, 5))
    surprise_df = train[["user_id", "movie_id", "rating"]].copy()
    surprise_df.columns = ["user_id", "item_id", "rating"]

    dataset = Dataset.load_from_df(surprise_df, reader)
    print(
        f"[build_surprise_dataset] loaded {len(surprise_df):,} ratings "
        f"into Surprise Dataset"
    )
    return dataset


# ---------------------------------------------------------------------------
# 5. Implicit ALS confidence matrix
# ---------------------------------------------------------------------------

def build_implicit_matrix(
    train: pd.DataFrame,
    n_users: int,
    n_movies: int,
    alpha: float = 40.0,
) -> sp.csr_matrix:
    """Build a confidence-weighted interaction matrix for implicit ALS.

    Following the formulation from Hu, Koren & Volinsky (2008):
    ``C_{ui} = 1 + alpha * r_{ui}``

    The resulting matrix is item × user (transposed) as expected by the
    ``implicit`` library's ALS model.

    Args:
        train: DataFrame with ``user_idx``, ``movie_idx``, ``rating`` cols.
        n_users: Number of unique users.
        n_movies: Number of unique movies.
        alpha: Scaling factor for confidence weights.  Default ``40``.

    Returns:
        ``scipy.sparse.csr_matrix`` of shape ``(n_movies, n_users)``
        (item-user orientation expected by ``implicit``).
    """
    rows = train["user_idx"].values
    cols = train["movie_idx"].values
    ratings = train["rating"].values.astype(np.float32)

    confidence = 1.0 + alpha * ratings

    # Build user×item first, then transpose to item×user
    user_item = sp.csr_matrix(
        (confidence, (rows, cols)),
        shape=(n_users, n_movies),
        dtype=np.float32,
    )
    item_user = user_item.T.tocsr()

    print(
        f"[build_implicit_matrix] alpha={alpha} | "
        f"item×user shape={item_user.shape}, nnz={item_user.nnz:,}"
    )
    return item_user


# ---------------------------------------------------------------------------
# 6. Cold-start sets
# ---------------------------------------------------------------------------

def identify_cold_start(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_train_ratings: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify cold-start users and movies in the test set.

    Cold users:  users appearing in *test* who have fewer than
                 ``min_train_ratings`` ratings in *train*.
    Cold movies: movies appearing in *test* that do not appear in *train*
                 at all (completely unseen items).

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        min_train_ratings: Threshold below which a test user is considered
                           "cold".  Default ``5``.

    Returns:
        ``(cold_users_df, cold_movies_df)`` – DataFrames with columns
        ``[user_id, train_count]`` and ``[movie_id]`` respectively.
    """
    train_user_counts = (
        train.groupby("user_id").size().rename("train_count").reset_index()
    )
    test_users = test[["user_id"]].drop_duplicates()
    merged_users = test_users.merge(train_user_counts, on="user_id", how="left")
    merged_users["train_count"] = merged_users["train_count"].fillna(0).astype(int)
    cold_users = merged_users[merged_users["train_count"] < min_train_ratings].copy()

    train_movies = set(train["movie_id"].unique())
    test_movies = set(test["movie_id"].unique())
    unseen_movies = test_movies - train_movies
    cold_movies = pd.DataFrame({"movie_id": sorted(unseen_movies)})

    print(
        f"[identify_cold_start] cold_users={len(cold_users):,} "
        f"(< {min_train_ratings} train ratings) | "
        f"cold_movies={len(cold_movies):,} (unseen in train)"
    )
    return cold_users, cold_movies


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocess(cfg: dict) -> dict:
    """Execute the complete preprocessing pipeline.

    Args:
        cfg: Configuration dictionary with keys:
            - ratings_path (str)
            - processed_dir (str): output root for parquet / npz / json
            - split (str): "temporal" or "random"
            - cutoff (str): ISO date for temporal split
            - test_size (float): fraction for random split
            - random_state (int)
            - alpha (float): implicit confidence scaling
            - min_train_ratings (int): cold-start threshold

    Returns:
        A dict with artefact paths and key stats (useful for downstream
        scripts and tests).
    """
    t0 = time.time()

    # ---- Load ---------------------------------------------------------------
    ratings_path = cfg["ratings_path"]
    print(f"\n{'='*60}")
    print(f" PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    print(f"Loading ratings from {ratings_path} ...")
    df = pd.read_parquet(ratings_path)
    print(f"Loaded {len(df):,} rows, shape={df.shape}")

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    out_root = Path(cfg["processed_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. Split -----------------------------------------------------------
    print(f"\n-- Step 1: {cfg['split']} split --")
    train, test = split_dataset(
        df,
        split=cfg["split"],
        cutoff=cfg.get("cutoff", "2005-10-01"),
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
    )

    train_path = out_root / "train.parquet"
    test_path = out_root / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(f"Saved train -> {train_path}")
    print(f"Saved test  -> {test_path}")

    # ---- 2. ID mappings -----------------------------------------------------
    print("\n-- Step 2: ID remapping --")
    user2idx, idx2user, movie2idx, idx2movie = build_id_mappings(train)

    mappings_dir = out_root / "mappings"
    save_mappings(user2idx, idx2user, movie2idx, idx2movie, mappings_dir)

    train_mapped = apply_id_mappings(train, user2idx, movie2idx)
    test_mapped = apply_id_mappings(test, user2idx, movie2idx)
    print(
        f"train_mapped={len(train_mapped):,} rows | "
        f"test_mapped={len(test_mapped):,} rows (cold-start rows dropped)"
    )

    n_users = len(user2idx)
    n_movies = len(movie2idx)

    # ---- 3. CSR matrix ------------------------------------------------------
    print("\n-- Step 3: CSR matrix --")
    csr = build_csr_matrix(train_mapped, n_users, n_movies)
    npz_path = out_root / "train_csr.npz"
    sp.save_npz(str(npz_path), csr)
    print(f"Saved CSR matrix -> {npz_path}")

    # ---- 4. Surprise dataset (no file – returned as object) -----------------
    print("\n-- Step 4: Surprise Dataset --")
    try:
        surprise_ds = build_surprise_dataset(train_mapped)
    except ImportError as e:
        print(f"  Warning: {e}")
        surprise_ds = None

    # ---- 5. Implicit matrix -------------------------------------------------
    print("\n-- Step 5: Implicit confidence matrix --")
    alpha = cfg.get("alpha", 40.0)
    impl_matrix = build_implicit_matrix(train_mapped, n_users, n_movies, alpha=alpha)
    impl_path = out_root / "train_implicit.npz"
    sp.save_npz(str(impl_path), impl_matrix)
    print(f"Saved implicit matrix -> {impl_path}")

    # ---- 6. Cold-start sets -------------------------------------------------
    print("\n-- Step 6: Cold-start sets --")
    min_train = cfg.get("min_train_ratings", 5)
    cold_users, cold_movies = identify_cold_start(train, test, min_train_ratings=min_train)

    cold_users_path = out_root / "cold_users.csv"
    cold_movies_path = out_root / "cold_movies.csv"
    cold_users.to_csv(cold_users_path, index=False)
    cold_movies.to_csv(cold_movies_path, index=False)
    print(f"Saved cold_users -> {cold_users_path}")
    print(f"Saved cold_movies -> {cold_movies_path}")

    elapsed = time.time() - t0
    print(f"\nPreprocessing complete in {elapsed:.2f}s")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "mappings_dir": str(mappings_dir),
        "npz_path": str(npz_path),
        "impl_path": str(impl_path),
        "cold_users_path": str(cold_users_path),
        "cold_movies_path": str(cold_movies_path),
        "n_users": n_users,
        "n_movies": n_movies,
        "n_train": len(train),
        "n_test": len(test),
        "csr_matrix": csr,
        "impl_matrix": impl_matrix,
        "surprise_dataset": surprise_ds,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "movie2idx": movie2idx,
        "idx2movie": idx2movie,
        "cold_users": cold_users,
        "cold_movies": cold_movies,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess Netflix Prize data for modelling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ratings_path",
        default="data/processed/ratings_sample.parquet",
        help="Input ratings parquet file.",
    )
    p.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Root output directory for artefacts.",
    )
    p.add_argument(
        "--split",
        choices=["temporal", "random"],
        default="temporal",
        help="Train/test split strategy.",
    )
    p.add_argument(
        "--cutoff",
        default="2005-10-01",
        help="ISO date cutoff for temporal split.",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test fraction for random split.",
    )
    p.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="RNG seed for random split.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=40.0,
        help="Confidence scaling factor for implicit ALS matrix.",
    )
    p.add_argument(
        "--min_train_ratings",
        type=int,
        default=5,
        help="Min train ratings below which a test user is cold-start.",
    )
    return p


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    cfg = vars(args)
    run_preprocess(cfg)


if __name__ == "__main__":
    main()
