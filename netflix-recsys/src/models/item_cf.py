"""Item-based collaborative filtering model.

Approach
--------
1. Build a sparse (n_movies, n_users) item–user CSR matrix from training data.
2. Pre-compute, for every movie, the top-*k* most similar movies using
   batched cosine similarity.  Only the top-k indices and values are stored,
   keeping memory at O(n_movies × k).
3. ``predict(user_id, movie_id)`` performs a similarity-weighted average of
   the user's ratings for the most similar items they have seen.
4. ``recommend(user_id, top_k)`` is **fully vectorised** via numpy array ops
   over the stored (n_movies, k) index / value arrays.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base_model import BaseRecommender


class ItemCFModel(BaseRecommender):
    """Item-based k-NN collaborative filtering.

    Parameters
    ----------
    k : int
        Number of most-similar items used for prediction.  Default 50.
    batch_size : int
        Items processed per similarity-computation batch.  Default 200.
    """

    model_name = "ItemCF"

    def __init__(self, k: int = 50, batch_size: int = 200) -> None:
        self.k = k
        self.batch_size = batch_size

    # ------------------------------------------------------------------ fit

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "ItemCFModel":
        """Build item–item neighbourhood structure.

        Steps:
          1. Map raw IDs to contiguous 0-based indices.
          2. Build a sparse (n_movies, n_users) item–user CSR matrix.
          3. Batch-compute cosine similarities; store top-k per item.

        Args:
            train_df: Training ratings DataFrame.

        Returns:
            self
        """
        t0 = time.time()

        # --- ID mappings ---
        unique_users  = np.sort(train_df["user_id"].unique())
        unique_movies = np.sort(train_df["movie_id"].unique())

        self.user2idx_:  Dict[int, int] = {u: i for i, u in enumerate(unique_users)}
        self.idx2user_:  Dict[int, int] = {i: u for u, i in self.user2idx_.items()}
        self.movie2idx_: Dict[int, int] = {m: i for i, m in enumerate(unique_movies)}
        self.idx2movie_: Dict[int, int] = {i: m for m, i in self.movie2idx_.items()}

        n_users  = len(unique_users)
        n_movies = len(unique_movies)

        self.global_mean_: float = float(train_df["rating"].mean())

        # --- Sparse user×movie CSR (for predict/recommend lookups) ---
        u_rows = train_df["user_id"].map(self.user2idx_).values
        m_cols = train_df["movie_id"].map(self.movie2idx_).values
        data   = train_df["rating"].values.astype(np.float32)

        user_item_csr: sp.csr_matrix = sp.csr_matrix(
            (data, (u_rows, m_cols)), shape=(n_users, n_movies), dtype=np.float32
        )
        # Transpose to item×user for similarity computation
        self.item_user_csr_: sp.csr_matrix = user_item_csr.T.tocsr()
        # Keep user×item for fast row (user) retrieval
        self.user_item_csr_: sp.csr_matrix = user_item_csr

        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        print(
            f"[{self.model_name}] item×user CSR shape={self.item_user_csr_.shape}, "
            f"nnz={self.item_user_csr_.nnz:,}, k={self.k}"
        )

        # --- Pre-compute top-k item neighbours ---
        self._compute_item_neighbours(n_movies)

        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    def _compute_item_neighbours(self, n_movies: int) -> None:
        """Batched cosine-similarity item–item neighbourhood computation.

        Stores ``self.item_sim_indices_`` and ``self.item_sim_values_``,
        both shape (n_movies, k), dtype int32 / float32.

        Args:
            n_movies: Number of unique movies in the training set.
        """
        try:
            from tqdm import tqdm
            batches = tqdm(
                range(0, n_movies, self.batch_size),
                desc=f"[{self.model_name}] computing item neighbours",
                unit="batch",
            )
        except ImportError:
            batches = range(0, n_movies, self.batch_size)

        k = min(self.k, n_movies - 1)
        self.item_sim_indices_ = np.zeros((n_movies, k), dtype=np.int32)
        self.item_sim_values_  = np.zeros((n_movies, k), dtype=np.float32)

        for start in batches:
            end   = min(start + self.batch_size, n_movies)
            batch = self.item_user_csr_[start:end]          # (B, n_users) sparse

            # (B, n_movies) dense cosine similarities
            sims: np.ndarray = cosine_similarity(batch, self.item_user_csr_)

            # Zero out self-similarity
            for i in range(end - start):
                sims[i, start + i] = -1.0

            # top-k per row
            if n_movies - 1 >= k:
                top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            else:
                top_idx = np.tile(np.arange(n_movies), (end - start, 1))

            top_sims = np.take_along_axis(sims, top_idx, axis=1)

            order    = np.argsort(top_sims, axis=1)[:, ::-1]
            top_idx  = np.take_along_axis(top_idx,  order, axis=1)
            top_sims = np.take_along_axis(top_sims, order, axis=1)

            self.item_sim_indices_[start:end] = top_idx.astype(np.int32)
            self.item_sim_values_[start:end]  = top_sims.astype(np.float32)

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating as similarity-weighted average of user's rated neighbours.

        Args:
            user_id: Raw user identifier.
            movie_id: Raw movie identifier.

        Returns:
            Predicted rating clipped to [1, 5].
        """
        user_idx  = self.user2idx_.get(user_id)
        movie_idx = self.movie2idx_.get(movie_id)

        if user_idx is None or movie_idx is None:
            return self.global_mean_

        sim_item_idxs = self.item_sim_indices_[movie_idx]  # (k,)
        sim_values    = self.item_sim_values_[movie_idx]   # (k,)

        # Ratings user gave to each of those similar items.
        # user_item_csr_[user, cols] returns a sparse matrix;
        # use .toarray() to densify before flattening.
        user_ratings = (
            self.user_item_csr_[user_idx, sim_item_idxs]
            .toarray().ravel().astype(np.float32)
        )                                                  # (k,)

        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return self.global_mean_

        r = user_ratings[rated_mask]
        s = sim_values[rated_mask]
        denom = s.sum()
        if denom == 0.0:
            return self.global_mean_

        return float(np.clip(np.dot(s, r) / denom, 1.0, 5.0))

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return top-K unseen movie recommendations.

        Fully vectorised via numpy array indexing over the stored
        (n_movies, k) item-similarity arrays.

        Args:
            user_id: Raw user identifier.
            top_k: Number of items to return.

        Returns:
            List of (movie_id, predicted_score) tuples sorted descending.
        """
        user_idx = self.user2idx_.get(user_id)
        if user_idx is None:
            return []

        n_movies = self.item_user_csr_.shape[0]

        # User's rating vector (dense), shape (n_movies,)
        user_row = self.user_item_csr_[user_idx]
        rated_indices  = user_row.indices
        rated_ratings  = user_row.data.astype(np.float32)

        # Build a lookup array: rating_lookup[movie_idx] = rating (0 if unrated)
        rating_lookup = np.zeros(n_movies, dtype=np.float32)
        rating_lookup[rated_indices] = rated_ratings

        # For every movie m, its similar-item ratings given by this user:
        # item_sim_indices_[m] → shape (n_movies, k)
        # rating_lookup[item_sim_indices_] → shape (n_movies, k)
        sim_ratings  = rating_lookup[self.item_sim_indices_]      # (n_movies, k)
        rated_mask   = sim_ratings > 0                             # (n_movies, k)

        weighted_sum = (self.item_sim_values_ * sim_ratings).sum(axis=1)  # (n_movies,)
        sim_sum      = (self.item_sim_values_ * rated_mask).sum(axis=1)   # (n_movies,)

        pred = np.where(sim_sum > 0, weighted_sum / sim_sum, -np.inf)
        pred = np.where(pred > -np.inf, np.clip(pred, 1.0, 5.0), -np.inf)

        # Mask already-rated items
        pred[rated_indices] = -np.inf

        actual_k = min(top_k, n_movies - len(rated_indices))
        if actual_k <= 0:
            return []

        top_idx = np.argpartition(pred, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(pred[top_idx])[::-1]]

        return [
            (int(self.idx2movie_[i]), float(pred[i]))
            for i in top_idx
            if pred[i] > -np.inf
        ]
