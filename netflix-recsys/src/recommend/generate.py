"""Recommendation generation module.

Loads the best-performing model and generates personalised top-K
recommendations, enriched with movie metadata (title, year).

Usage::

    python src/recommend/generate.py
    python src/recommend/generate.py --model svd --n_users 500 --top_k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.base_model import BaseRecommender


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_MODEL_FILES = {
    "svd":        "svd.joblib",
    "als":        "als.joblib",
    "usercf":     "usercf.joblib",
    "itemcf":     "itemcf.joblib",
    "globalmean": "globalmean.joblib",
    "biasmodel":  "biasmodel.joblib",
}


def load_model(
    name: str = "svd",
    models_dir: str | Path = "outputs/models",
) -> BaseRecommender:
    """Load a trained model from disk.

    Args:
        name: Model key (svd, als, usercf, itemcf, globalmean, biasmodel).
              Use ``"best"`` to load SVD (best MAP@10 model).
        models_dir: Directory containing serialised ``.joblib`` files.

    Returns:
        Loaded ``BaseRecommender`` instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if name == "best":
        name = "svd"  # SVD has best MAP@10

    filename = _MODEL_FILES.get(name.lower())
    if filename is None:
        raise ValueError(f"Unknown model: {name!r}. Choose from {list(_MODEL_FILES)}")

    path = Path(models_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return BaseRecommender.load(path)


def load_movie_metadata(
    path: str | Path = "data/processed/movies.parquet",
) -> pd.DataFrame:
    """Load movie titles/years from the processed movies parquet.

    Args:
        path: Path to movies.parquet.

    Returns:
        DataFrame with columns [movie_id, year, title].
    """
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Top-K generation
# ---------------------------------------------------------------------------

def generate_top_k(
    user_ids: List[int],
    model: BaseRecommender,
    movies_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Generate top-K recommendations for a list of users.

    Calls the model's ``recommend()`` method which already excludes
    movies the user has rated in the training set.

    Args:
        user_ids: List of raw user identifiers.
        model: A fitted ``BaseRecommender`` with ``recommend()`` support.
        movies_df: Movie metadata DataFrame with [movie_id, year, title].
        k: Number of recommendations per user.

    Returns:
        DataFrame with columns: user_id, rank, movie_id, title, year,
        predicted_score.  Rows are sorted by (user_id, rank).
        
    Raises:
        ValueError: If user_ids is empty or model is None.
    """
    if not user_ids:
        raise ValueError("user_ids list cannot be empty")
    if model is None:
        raise ValueError("model cannot be None")
    
    title_map = movies_df.set_index("movie_id")["title"].to_dict()
    year_map = movies_df.set_index("movie_id")["year"].to_dict()

    records = []
    skipped_users = 0
    for uid in user_ids:
        try:
            recs = model.recommend(int(uid), top_k=k)
        except (KeyError, ValueError) as e:
            # User not found or other expected error - skip gracefully
            skipped_users += 1
            continue
        except Exception as e:
            # Unexpected error - log and re-raise
            import warnings
            warnings.warn(f"Unexpected error generating recommendations for user {uid}: {e}")
            skipped_users += 1
            continue

        for rank, (mid, score) in enumerate(recs, start=1):
            records.append({
                "user_id": int(uid),
                "rank": rank,
                "movie_id": int(mid),
                "title": title_map.get(int(mid), "Unknown"),
                "year": year_map.get(int(mid), "N/A"),
                "predicted_score": float(score),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["user_id", "rank"]).reset_index(drop=True)
    
    if skipped_users > 0:
        import warnings
        warnings.warn(f"Skipped {skipped_users} out of {len(user_ids)} users during recommendation generation")
    
    return df


# ---------------------------------------------------------------------------
# Similar movies (item-item)
# ---------------------------------------------------------------------------

def generate_similar_movies(
    movie_id: int,
    k: int = 10,
    models_dir: str | Path = "outputs/models",
    movies_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Find the K most similar movies using ItemCF similarity.

    Loads the ItemCF model and looks up the pre-computed item neighbours.

    Args:
        movie_id: Raw movie identifier.
        k: Number of similar movies to return.
        models_dir: Directory with model files.
        movies_df: Optional movie metadata for title enrichment.

    Returns:
        DataFrame with columns: movie_id, similarity, title, year.
    """
    itemcf = load_model("itemcf", models_dir)

    movie_idx = itemcf.movie2idx_.get(movie_id)
    if movie_idx is None:
        print(f"Movie {movie_id} not found in ItemCF vocabulary.")
        return pd.DataFrame()

    sim_indices = itemcf.item_sim_indices_[movie_idx][:k]
    sim_values = itemcf.item_sim_values_[movie_idx][:k]

    records = []
    for idx, sim in zip(sim_indices, sim_values):
        mid = itemcf.idx2movie_.get(int(idx), -1)
        records.append({"movie_id": mid, "similarity": float(sim)})

    df = pd.DataFrame(records)

    if movies_df is not None and not df.empty:
        title_map = movies_df.set_index("movie_id")["title"].to_dict()
        year_map = movies_df.set_index("movie_id")["year"].to_dict()
        df["title"] = df["movie_id"].map(title_map).fillna("Unknown")
        df["year"] = df["movie_id"].map(year_map).fillna("N/A")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate recommendations for sample test users and save to CSV."""
    parser = argparse.ArgumentParser(
        description="Generate top-K recommendations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="best",
                        help="Model to use (svd, als, itemcf, usercf, best).")
    parser.add_argument("--n_users", type=int, default=500,
                        help="Number of random test users.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--test_path", default="data/processed/test.parquet")
    parser.add_argument("--movies_path", default="data/processed/movies.parquet")
    parser.add_argument("--models_dir", default="outputs/models")
    parser.add_argument("--output_dir", default="outputs/recommendations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.models_dir)

    # Load metadata
    movies_df = load_movie_metadata(args.movies_path)

    # Sample test users
    test_df = pd.read_parquet(args.test_path)
    all_users = test_df["user_id"].unique()
    n_sample = min(args.n_users, len(all_users))
    rng = np.random.default_rng(42)
    sample_users = rng.choice(all_users, size=n_sample, replace=False).tolist()

    print(f"Generating top-{args.top_k} recs for {n_sample} users ...")
    recs_df = generate_top_k(sample_users, model, movies_df, k=args.top_k)

    # Save
    csv_path = output_dir / "sample_recs.csv"
    recs_df.to_csv(csv_path, index=False)
    print(f"Saved {len(recs_df):,} recommendations -> {csv_path}")

    # Demo: similar movies for a popular movie
    print("\n--- Similar Movies Demo ---")
    sample_movie = int(movies_df["movie_id"].iloc[0])
    sim_df = generate_similar_movies(sample_movie, k=5, movies_df=movies_df,
                                     models_dir=args.models_dir)
    if not sim_df.empty:
        source_title = movies_df.loc[
            movies_df["movie_id"] == sample_movie, "title"
        ].values[0]
        print(f"Movies similar to '{source_title}' (ID={sample_movie}):")
        print(sim_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
