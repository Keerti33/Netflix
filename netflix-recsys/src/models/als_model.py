"""ALS implicit-feedback model using the *implicit* library.

Wraps ``implicit.als.AlternatingLeastSquares`` behind the
``BaseRecommender`` interface.

.. important::

   ALS is trained on **implicit feedback** (confidence-weighted ratings)
   using the formulation ``C_ui = 1 + alpha * r_ui`` (Hu, Koren & Volinsky
   2008).  The latent factors optimise a different objective than explicit
   rating prediction, so **RMSE on the explicit test set is not directly
   comparable** to the baselines or SVD.  Use ranking metrics (precision@K,
   nDCG, MAP) for a fair comparison.

Typical usage::

    from src.models.als_model import ALSModel
    model = ALSModel(factors=100)
    model.fit(train_df, implicit_matrix=impl_csr)
    model.recommend(user_id=123, top_k=10)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models.base_model import BaseRecommender


class ALSModel(BaseRecommender):
    """ALS implicit-feedback collaborative filtering.

    Parameters
    ----------
    factors : int
        Number of latent factors.  Default 100.
    regularization : float
        L2 regularisation weight.  Default 0.01.
    iterations : int
        Number of ALS iterations.  Default 20.
    alpha : float
        Confidence scaling: ``C = 1 + alpha * rating``.  Default 40.
    """

    model_name = "ALS"

    def __init__(
        self,
        factors: int = 100,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        train_df: pd.DataFrame,
        implicit_matrix: Optional[sp.csr_matrix] = None,
        implicit_matrix_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "ALSModel":
        """Train ALS on the confidence-weighted implicit matrix.

        The implicit matrix must be in **item x user** orientation
        (as expected by the ``implicit`` library).  It can be provided
        directly, loaded from an ``.npz`` file, or built on the fly from
        *train_df* using the stored alpha.

        Args:
            train_df: Training DataFrame with [user_id, movie_id, rating].
            implicit_matrix: Pre-built item-user CSR confidence matrix.
            implicit_matrix_path: Path to a ``.npz`` file containing the
                                  item-user matrix.

        Returns:
            self
        """
        t0 = time.time()

        # Suppress OpenBLAS threading warning from implicit
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        from implicit.als import AlternatingLeastSquares

        # -- ID mappings -----------------------------------------------
        unique_users = np.sort(train_df["user_id"].unique())
        unique_movies = np.sort(train_df["movie_id"].unique())

        self.user2idx_: Dict[int, int] = {u: i for i, u in enumerate(unique_users)}
        self.idx2user_: Dict[int, int] = {i: u for u, i in self.user2idx_.items()}
        self.movie2idx_: Dict[int, int] = {m: i for i, m in enumerate(unique_movies)}
        self.idx2movie_: Dict[int, int] = {i: m for m, i in self.movie2idx_.items()}

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        self.global_mean_: float = float(train_df["rating"].mean())

        # -- Build or load implicit matrix -----------------------------
        # The preprocess module stores item-user orientation (n_items, n_users).
        # implicit 0.7.x expects user-item (n_users, n_items) for fit().
        if implicit_matrix is not None:
            item_user_csr = implicit_matrix
        elif implicit_matrix_path is not None:
            item_user_csr = sp.load_npz(str(implicit_matrix_path))
        else:
            # Build on the fly (already in item-user orientation)
            u_rows = train_df["user_id"].map(self.user2idx_).values
            m_cols = train_df["movie_id"].map(self.movie2idx_).values
            ratings = train_df["rating"].values.astype(np.float32)
            confidence = 1.0 + self.alpha * ratings
            user_item = sp.csr_matrix(
                (confidence, (u_rows, m_cols)),
                shape=(n_users, n_movies),
                dtype=np.float32,
            )
            item_user_csr = user_item.T.tocsr()

        # Transpose to user-item for implicit 0.7.x API
        self.user_item_csr_: sp.csr_matrix = item_user_csr.T.tocsr()

        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        # -- Fit ALS ---------------------------------------------------
        self.algo_ = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )

        print(
            f"[{self.model_name}] Training ALS: factors={self.factors}, "
            f"reg={self.regularization}, iters={self.iterations}, "
            f"alpha={self.alpha}"
        )
        print(
            f"[{self.model_name}] user-item matrix shape={self.user_item_csr_.shape}, "
            f"nnz={self.user_item_csr_.nnz:,}"
        )

        # implicit 0.7.x: fit(user_items) expects (n_users, n_items) CSR
        self.algo_.fit(self.user_item_csr_, show_progress=True)

        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a score for (user_id, movie_id).

        .. note::
           ALS scores are **not calibrated ratings** (they are dot products
           of latent factor vectors).  Returning them clipped to [1, 5]
           allows the ``run_mf.py`` runner to compute RMSE, but the values
           should not be interpreted as true rating predictions.

        Args:
            user_id: Raw user identifier.
            movie_id: Raw movie identifier.

        Returns:
            ALS score clipped to [1, 5].
        """
        user_idx = self.user2idx_.get(user_id)
        movie_idx = self.movie2idx_.get(movie_id)

        if user_idx is None or movie_idx is None:
            return self.global_mean_

        # Dot product of user and item factors
        score = float(
            self.algo_.user_factors[user_idx]
            @ self.algo_.item_factors[movie_idx]
        )
        return float(np.clip(score, 1.0, 5.0))

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return top-K unseen movie recommendations.

        Uses the ``implicit`` library's optimised recommend method which
        computes dot products efficiently against all item factors.

        Args:
            user_id: Raw user identifier.
            top_k: Number of items to return.

        Returns:
            List of (movie_id, score) sorted descending by ALS score.
        """
        user_idx = self.user2idx_.get(user_id)
        if user_idx is None:
            return []

        # implicit's recommend returns (item_indices, scores)
        item_indices, scores = self.algo_.recommend(
            user_idx,
            self.user_item_csr_[user_idx],
            N=top_k,
            filter_already_liked_items=True,
        )

        return [
            (int(self.idx2movie_.get(int(idx), -1)), float(s))
            for idx, s in zip(item_indices, scores)
            if int(idx) in self.idx2movie_
        ]
