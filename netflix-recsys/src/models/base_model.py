"""Abstract base class for all recommendation models.

Every model in src/models/ inherits from ``BaseRecommender`` and must
implement ``fit()``, ``predict()``, and optionally ``recommend()``.
Save / load and RMSE evaluation helpers are provided here so each
subclass gets them for free.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for Netflix Prize recommendation models.

    Subclasses must implement:
        - fit(train_df, **kwargs) -> self
        - predict(user_id, movie_id) -> float  (clipped to [1, 5])

    Subclasses may override:
        - recommend(user_id, top_k) -> list[(movie_id, score)]
        - predict_batch(test_df)    -> np.ndarray  (vectorised version)
    """

    # Subclasses set this string as a human-readable model name.
    model_name: str = "BaseRecommender"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> "BaseRecommender":
        """Train the model on *train_df*.

        Args:
            train_df: DataFrame with columns [user_id, movie_id, rating, date].

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for (user_id, movie_id).

        Cold-start (unknown user or movie) must return the global mean.
        Output is clipped to [1.0, 5.0].

        Args:
            user_id: Raw user identifier.
            movie_id: Raw movie identifier.

        Returns:
            Predicted rating as a float in [1, 5].
        """

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Return top-K unseen (movie_id, predicted_score) pairs.

        Args:
            user_id: Raw user identifier.
            top_k: Number of recommendations.

        Returns:
            List of (movie_id, score) tuples, sorted descending by score.

        Raises:
            NotImplementedError: if the subclass has not implemented this.
        """
        raise NotImplementedError(
            f"{self.model_name}.recommend() is not implemented."
        )

    # ------------------------------------------------------------------
    # Default batch predict (loop + tqdm) – override for speedup
    # ------------------------------------------------------------------

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for every row in *test_df*.

        Default implementation calls :meth:`predict` in a tqdm loop.
        Subclasses with vectorised internals should override this.

        Args:
            test_df: DataFrame with at least columns [user_id, movie_id].

        Returns:
            np.ndarray of float32 predictions, same length as test_df.
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(
                test_df.itertuples(index=False),
                total=len(test_df),
                desc=f"Predicting [{self.model_name}]",
                unit="row",
            )
        except ImportError:
            iterator = test_df.itertuples(index=False)

        preds = np.fromiter(
            (self.predict(int(row.user_id), int(row.movie_id)) for row in iterator),
            dtype=np.float32,
            count=len(test_df),
        )
        return preds

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error.

        Args:
            y_true: Ground-truth ratings.
            y_pred: Predicted ratings.

        Returns:
            RMSE as a float.
        """
        diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
        return float(np.sqrt(np.mean(diff ** 2)))

    # ------------------------------------------------------------------
    # Persist / restore
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise the model to *path* using joblib.

        Args:
            path: File path (parent dirs created automatically).

        Returns:
            Resolved Path object.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"[{self.model_name}] saved -> {path}  ({size_mb:.2f} MB)")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseRecommender":
        """Deserialise a model from *path*.

        Args:
            path: Path previously produced by :meth:`save`.

        Returns:
            Loaded model instance.
        """
        model = joblib.load(path)
        print(f"[{type(model).model_name}] loaded <- {path}")
        return model

    # ------------------------------------------------------------------
    # Timing helper (used in fit implementations)
    # ------------------------------------------------------------------

    @staticmethod
    def _log_fit_time(model_name: str, elapsed: float) -> None:
        print(f"[{model_name}] fit complete in {elapsed:.3f}s")
