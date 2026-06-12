"""SVD matrix factorization model using scikit-surprise.

Wraps Surprise's ``SVD`` algorithm behind the ``BaseRecommender`` interface.
Provides optional hyper-parameter tuning via ``GridSearchCV`` and saves the
best parameters to a JSON file for reproducibility.

Typical usage::

    from src.models.svd_model import SVDModel
    model = SVDModel(n_factors=100)
    model.fit(train_df, tune=True)          # GridSearchCV + refit
    model.predict(user_id=123, movie_id=456)
    model.recommend(user_id=123, top_k=10)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from surprise import SVD, Dataset, Reader, Trainset
from surprise.model_selection import GridSearchCV

from src.models.base_model import BaseRecommender


# Default hyper-parameter grid for tuning
DEFAULT_PARAM_GRID = {
    "n_factors": [50, 100, 150],
    "n_epochs": [20, 30],
    "lr_all": [0.005, 0.01],
    "reg_all": [0.02, 0.1],
}


class SVDModel(BaseRecommender):
    """SVD matrix factorization via scikit-surprise.

    Parameters
    ----------
    n_factors : int
        Number of latent factors.  Default 100.
    n_epochs : int
        Number of SGD epochs.  Default 20.
    lr_all : float
        Learning rate for all parameters.  Default 0.005.
    reg_all : float
        Regularisation term for all parameters.  Default 0.02.
    """

    model_name = "SVD"

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        train_df: pd.DataFrame,
        tune: bool = False,
        param_grid: Optional[dict] = None,
        cv_folds: int = 3,
        best_params_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "SVDModel":
        """Train the SVD model on *train_df*.

        Args:
            train_df: Training DataFrame with [user_id, movie_id, rating].
            tune: If True, run GridSearchCV before fitting.
            param_grid: Custom parameter grid; defaults to
                        ``DEFAULT_PARAM_GRID``.
            cv_folds: Number of cross-validation folds for tuning.
            best_params_path: If provided, save best params JSON here.

        Returns:
            self
        """
        t0 = time.time()

        # -- Build Surprise Dataset ------------------------------------
        reader = Reader(rating_scale=(1, 5))
        surprise_df = train_df[["user_id", "movie_id", "rating"]].copy()
        surprise_df.columns = ["user_id", "item_id", "rating"]
        self._surprise_ds = Dataset.load_from_df(surprise_df, reader)

        self.global_mean_: float = float(train_df["rating"].mean())

        # -- Optional GridSearchCV -------------------------------------
        if tune:
            grid = param_grid or DEFAULT_PARAM_GRID
            print(f"[{self.model_name}] Running GridSearchCV "
                  f"({cv_folds} folds, {len(grid)} param axes) ...")
            gs = GridSearchCV(
                SVD,
                grid,
                measures=["rmse"],
                cv=cv_folds,
                n_jobs=-1,
                joblib_verbose=0,
            )
            gs.fit(self._surprise_ds)
            best = gs.best_params["rmse"]
            print(f"[{self.model_name}] Best params (RMSE={gs.best_score['rmse']:.4f}): {best}")

            self.n_factors = best.get("n_factors", self.n_factors)
            self.n_epochs = best.get("n_epochs", self.n_epochs)
            self.lr_all = best.get("lr_all", self.lr_all)
            self.reg_all = best.get("reg_all", self.reg_all)
            self.best_params_ = best
            self.best_cv_rmse_ = gs.best_score["rmse"]

            if best_params_path:
                p = Path(best_params_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    json.dump(best, f, indent=2)
                print(f"[{self.model_name}] Best params saved -> {p}")

        # -- Fit on full trainset --------------------------------------
        self.algo_ = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
        )
        trainset: Trainset = self._surprise_ds.build_full_trainset()
        self.algo_.fit(trainset)
        self._trainset = trainset

        # Pre-compute user rated sets for recommend()
        self.user_rated_: Dict[int, Set[int]] = (
            train_df.groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )
        self.all_movie_ids_: np.ndarray = train_df["movie_id"].unique()

        n_users = trainset.n_users
        n_items = trainset.n_items
        print(
            f"[{self.model_name}] fitted: n_factors={self.n_factors}, "
            f"n_epochs={self.n_epochs}, n_users={n_users:,}, n_items={n_items:,}"
        )
        self._log_fit_time(self.model_name, time.time() - t0)
        return self

    # --------------------------------------------------------------- predict

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for (user_id, movie_id).

        Falls back to global mean for cold-start users/items.

        Args:
            user_id: Raw user identifier.
            movie_id: Raw movie identifier.

        Returns:
            Predicted rating clipped to [1, 5].
        """
        pred = self.algo_.predict(str(user_id), str(movie_id))
        return float(np.clip(pred.est, 1.0, 5.0))

    # -------------------------------------------------- predict_batch (vectorised)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Vectorised batch prediction via Surprise test set API.

        Args:
            test_df: Test DataFrame with [user_id, movie_id, rating].

        Returns:
            np.ndarray of float32 predictions.
        """
        from surprise import Dataset, Reader

        reader = Reader(rating_scale=(1, 5))
        # Surprise expects (uid, iid, rating) tuples
        test_data = [
            (str(row.user_id), str(row.movie_id), float(row.rating))
            for row in test_df.itertuples(index=False)
        ]
        predictions = self.algo_.test(test_data)
        preds = np.array([p.est for p in predictions], dtype=np.float32)
        return np.clip(preds, 1.0, 5.0)

    # ------------------------------------------------------------- recommend

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return top-K unseen movie recommendations.

        Scores all unrated movies and returns the highest-predicted.

        Args:
            user_id: Raw user identifier.
            top_k: Number of items to return.

        Returns:
            List of (movie_id, predicted_score) sorted descending.
        """
        rated = self.user_rated_.get(user_id, set())
        unseen = [m for m in self.all_movie_ids_ if m not in rated]

        if not unseen:
            return []

        scores = np.array(
            [self.algo_.predict(str(user_id), str(m)).est for m in unseen],
            dtype=np.float32,
        )
        scores = np.clip(scores, 1.0, 5.0)

        actual_k = min(top_k, len(unseen))
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(int(unseen[i]), float(scores[i])) for i in top_idx]
