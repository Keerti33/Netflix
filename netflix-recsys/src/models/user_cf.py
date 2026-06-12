"""User-based collaborative filtering model.

Approach
--------
1. Build a sparse user × movie CSR rating matrix from the training data.
2. Pre-compute, for every user, the top-*k* most similar users (neighbours)
   using batched cosine similarity – no dense all-pairs matrix is stored.
3. ``predict(user_id, movie_id)`` performs a similarity-weighted average of
   the neighbours' ratings for that movie.
4. ``recommend(user_id, top_k)`` vectorises over all movies using the
   pre-fetched neighbour rating sub-matrix.

Memory footprint scales as O(n_users × k), not O(n_users²).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base_model import BaseRecommender


class UserCFModel(BaseRecommender):
    """User-based k-NN collaborative filtering.

    Parameters
    ----------
    k : int
        Number of nearest neighbours used for prediction.  Default 50.
    batch_size : int
        Number of users processed per similarity-computation batch.
        Larger values are faster but use more memory.  Default 500.
    """

    model_name = "UserCF"

    def __init__(self, k: int = 50, batch_size: int = 500) -> None:
        self.k = k
        self.batch_size = batch_size

    # ------------------------------------------------------------------ fit

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "UserCFModel":
        """Build the user–user neighbourhood structure.

        Steps:
          1. Map raw IDs to contiguous 0-based indices.
          2. Build a sparse (n_users, n_movies) CSR rating matrix.
          3. Batch-compute cosine similarities; keep top-k per user.

        Args:
            train_df: Training ratings DataFrame with columns
                      [user_id, movie_id, rating].

        Returns:
            self
        """
        t0 = time.time()

        # --- ID mappings (built internally from train) ---
        unique_users = np.sort(train_df["user_id"].unique())
        unique_movies = np.sort(train_df["movie_id"].unique())

        self.user2idx_: Dict[int, int] = {u: i for i, u in enumerate(unique_users)}
        self.idx2user_: Dict[int, int] = {i: u for u, i in self.user2idx_.items()}
        self.movie2idx_: Dict[int, int] = {m: i for i, m in enumerate(unique_movies)}
        self.idx2movie_: Dict[int, int] = {i: m for m, i in self.movie2idx_.items()}

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        self.global_mean_: float = float(train_df["rating"].mean())

        # --- Sparse rating matrix ---
        rows = train_df["user_id"].map(self.user2idx_).values
        cols = train_df["movie_id"].map(self.movie2idx_).values
        data = train_df["rating"].values.astype(np.float32)
        self.train_csr_: sp.csr_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_movies), dtype=np.float32
        )

        # --- Which movies each user has already rated ---
        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        print(
            f"[{self.model_name}] CSR shape={self.train_csr_.shape}, "
            f"nnz={self.train_csr_.nnz:,}, k={self.k}"
        )

        # --- Pre-compute top-k neighbours ---
        self._compute_neighbours(n_users)

        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    def _compute_neighbours(self, n_users: int) -> None:
        """Batched cosine-similarity neighbour computation.

        Stores ``self.nbr_indices_`` and ``self.nbr_sims_``,
        both of shape (n_users, k), dtype int32 / float32.

        Args:
            n_users: Total number of users in the training set.
        """
        try:
            from tqdm import tqdm
            batches = tqdm(
                range(0, n_users, self.batch_size),
                desc=f"[{self.model_name}] computing neighbours",
                unit="batch",
            )
        except ImportError:
            batches = range(0, n_users, self.batch_size)

        k = min(self.k, n_users - 1)
        self.nbr_indices_ = np.zeros((n_users, k), dtype=np.int32)
        self.nbr_sims_    = np.zeros((n_users, k), dtype=np.float32)

        for start in batches:
            end = min(start + self.batch_size, n_users)
            batch = self.train_csr_[start:end]          # (B, n_movies) sparse

            # (B, n_users) dense cosine similarities
            sims: np.ndarray = cosine_similarity(batch, self.train_csr_)

            # Zero out self-similarity
            for i in range(end - start):
                sims[i, start + i] = -1.0

            # argpartition: top-k indices per row (unsorted within top-k)
            if n_users - 1 >= k:
                top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            else:
                top_idx = np.tile(np.arange(n_users), (end - start, 1))
                for i in range(end - start):
                    top_idx[i, start + i] = top_idx[i, 0]  # placeholder

            top_sims = np.take_along_axis(sims, top_idx, axis=1)

            # Sort each row descending by similarity
            order = np.argsort(top_sims, axis=1)[:, ::-1]
            top_idx  = np.take_along_axis(top_idx,  order, axis=1)
            top_sims = np.take_along_axis(top_sims, order, axis=1)

            self.nbr_indices_[start:end] = top_idx.astype(np.int32)
            self.nbr_sims_[start:end]    = top_sims.astype(np.float32)

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating using similarity-weighted neighbour average.

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

        nbr_idxs = self.nbr_indices_[user_idx]   # (k,)
        nbr_sims = self.nbr_sims_[user_idx]       # (k,)

        # Ratings those neighbours gave this movie.
        # train_csr_[rows, col] returns a sparse matrix;
        # use .toarray() to densify before flattening.
        nbr_ratings = (
            self.train_csr_[nbr_idxs, movie_idx]
            .toarray().ravel().astype(np.float32)
        )                                                      # (k,)

        rated_mask = nbr_ratings > 0
        if not rated_mask.any():
            return self.global_mean_

        r = nbr_ratings[rated_mask]
        s = nbr_sims[rated_mask]
        denom = s.sum()
        if denom == 0.0:
            return self.global_mean_

        return float(np.clip(np.dot(s, r) / denom, 1.0, 5.0))

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return top-K unseen movie recommendations.

        Uses a fully vectorised computation over neighbour rating matrices.

        Args:
            user_id: Raw user identifier.
            top_k: Number of items to return.

        Returns:
            List of (movie_id, predicted_score) tuples sorted descending.
        """
        user_idx = self.user2idx_.get(user_id)
        if user_idx is None:
            return []

        nbr_idxs = self.nbr_indices_[user_idx]   # (k,)
        nbr_sims = self.nbr_sims_[user_idx]       # (k,)

        # Neighbour rating sub-matrix: (k, n_movies)
        nbr_ratings = self.train_csr_[nbr_idxs].toarray()

        # Weighted sum and normalisation mask, both (n_movies,)
        rated_mask   = nbr_ratings > 0                                         # (k, n_movies)
        weighted_sum = (nbr_sims[:, np.newaxis] * nbr_ratings).sum(axis=0)
        sim_sum      = (nbr_sims[:, np.newaxis] * rated_mask).sum(axis=0)

        pred = np.where(sim_sum > 0, weighted_sum / sim_sum, self.global_mean_)
        pred = np.clip(pred, 1.0, 5.0)

        # Mask already-rated items
        user_rated_idxs = [
            self.movie2idx_[m]
            for m in self.user_rated_.get(user_id, set())
            if m in self.movie2idx_
        ]
        pred[user_rated_idxs] = -np.inf

        n_movies = self.train_csr_.shape[1]
        actual_k = min(top_k, n_movies - len(user_rated_idxs))
        if actual_k <= 0:
            return []

        top_idx = np.argpartition(pred, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(pred[top_idx])[::-1]]

        return [
            (int(self.idx2movie_[i]), float(pred[i]))
            for i in top_idx
            if pred[i] > -np.inf
        ]
