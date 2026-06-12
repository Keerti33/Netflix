"""Global mean and bias-based baseline models.

Two models are provided:

GlobalMeanModel
    Predicts every (user, movie) pair as the training-set global mean.
    ``recommend()`` returns the most-rated unseen movies.

BiasModel
    Predicts: ``pred = global_mean + user_bias + item_bias``
    where biases are per-entity mean deviations from the global mean.
    ``recommend()`` scores unseen movies using their bias term.
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.models.base_model import BaseRecommender


class GlobalMeanModel(BaseRecommender):
    """Predicts every rating as the global training-set mean.

    Also records per-movie popularity so that ``recommend()`` can fall back
    to the most-watched unseen titles.
    """

    model_name = "GlobalMean"

    # ------------------------------------------------------------------ fit

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "GlobalMeanModel":
        """Compute global mean and per-movie popularity counts.

        Args:
            train_df: Training ratings DataFrame.

        Returns:
            self
        """
        t0 = time.time()
        self.global_mean_: float = float(train_df["rating"].mean())
        print(f"[{self.model_name}] global_mean = {self.global_mean_:.4f}")

        # popularity (count) per movie – for recommend()
        pop = train_df.groupby("movie_id").size().sort_values(ascending=False)
        self.movie_popularity_: pd.Series = pop

        # keep which items each user has already rated
        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the global mean (ignores both arguments).

        Args:
            user_id: Raw user identifier (unused).
            movie_id: Raw movie identifier (unused).

        Returns:
            Global mean rating.
        """
        return self.global_mean_

    # ------------------------------------------------------ predict_batch (vectorised)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return a constant array of global-mean values.

        Args:
            test_df: Test ratings DataFrame.

        Returns:
            np.ndarray of float32, all equal to the global mean.
        """
        return np.full(len(test_df), self.global_mean_, dtype=np.float32)

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Recommend the most-rated movies the user has not yet seen.

        Args:
            user_id: Raw user identifier.
            top_k: Number of recommendations to return.

        Returns:
            List of (movie_id, predicted_score) tuples.
        """
        rated = self.user_rated_.get(user_id, set())
        recs: List[Tuple[int, float]] = []
        for movie_id in self.movie_popularity_.index:
            if movie_id not in rated:
                recs.append((int(movie_id), self.global_mean_))
            if len(recs) >= top_k:
                break
        return recs


# ---------------------------------------------------------------------------


class BiasModel(BaseRecommender):
    """Additive bias model: pred = global_mean + user_bias + item_bias.

    User/item biases are the mean deviation of each entity's ratings from
    the global mean, computed on the training set.  Cold-start entities
    (unknown at inference) receive a bias of 0.
    """

    model_name = "BiasModel"

    # ------------------------------------------------------------------ fit

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "BiasModel":
        """Fit global mean and user/item bias terms.

        Args:
            train_df: Training ratings DataFrame.

        Returns:
            self
        """
        t0 = time.time()
        self.global_mean_: float = float(train_df["rating"].mean())

        user_means = train_df.groupby("user_id")["rating"].mean()
        movie_means = train_df.groupby("movie_id")["rating"].mean()

        self.user_bias_: Dict[int, float] = (user_means - self.global_mean_).to_dict()
        self.movie_bias_: Dict[int, float] = (movie_means - self.global_mean_).to_dict()

        # For recommend() – sorted by movie bias descending
        self.movie_bias_series_: pd.Series = (
            movie_means - self.global_mean_
        ).sort_values(ascending=False)

        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        n_users = len(self.user_bias_)
        n_movies = len(self.movie_bias_)
        print(
            f"[{self.model_name}] global_mean={self.global_mean_:.4f} | "
            f"n_users={n_users:,} | n_movies={n_movies:,}"
        )
        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating as global_mean + user_bias + movie_bias.

        Args:
            user_id: Raw user identifier.
            movie_id: Raw movie identifier.

        Returns:
            Clipped predicted rating in [1, 5].
        """
        ub = self.user_bias_.get(user_id, 0.0)
        mb = self.movie_bias_.get(movie_id, 0.0)
        return float(np.clip(self.global_mean_ + ub + mb, 1.0, 5.0))

    # ------------------------------------------------------ predict_batch (vectorised)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Vectorised batch prediction using pandas map.

        Args:
            test_df: Test ratings DataFrame.

        Returns:
            np.ndarray of float32 predictions.
        """
        ub = test_df["user_id"].map(self.user_bias_).fillna(0.0).values
        mb = test_df["movie_id"].map(self.movie_bias_).fillna(0.0).values
        preds = np.clip(self.global_mean_ + ub + mb, 1.0, 5.0)
        return preds.astype(np.float32)

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Recommend unseen movies with the highest item bias.

        A high positive bias means the movie is rated above average by most
        users; it is a reasonable proxy for quality.

        Args:
            user_id: Raw user identifier.
            top_k: Number of recommendations.

        Returns:
            List of (movie_id, predicted_score) tuples.
        """
        rated = self.user_rated_.get(user_id, set())
        ub = self.user_bias_.get(user_id, 0.0)
        recs: List[Tuple[int, float]] = []
        for movie_id, mb in self.movie_bias_series_.items():
            if movie_id not in rated:
                score = float(np.clip(self.global_mean_ + ub + mb, 1.0, 5.0))
                recs.append((int(movie_id), score))
            if len(recs) >= top_k:
                break
        return recs
