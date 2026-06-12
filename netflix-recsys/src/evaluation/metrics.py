"""Recommendation system evaluation metrics.

Provides rating-prediction metrics (RMSE, MAE) and ranking/retrieval
metrics (MAP@K, Precision@K, Recall@K, Coverage) for evaluating
recommendation models.

All ranking metrics accept a ``recommendations`` dict mapping each
user_id to a **ranked** list of recommended movie_ids, and a ``test_df``
DataFrame containing ground-truth (user_id, movie_id, rating) tuples.
A test item is considered *relevant* if its rating meets or exceeds
``relevance_threshold`` (default 3.5).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error between true and predicted ratings.

    Args:
        y_true: Array-like of ground-truth ratings.
        y_pred: Array-like of predicted ratings (same length as *y_true*).

    Returns:
        RMSE as a float.

    Raises:
        ValueError: If the input arrays have different lengths or are empty.

    Example::

        >>> compute_rmse([3, 4, 5], [3.1, 3.9, 4.8])
        0.1414...
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute RMSE on empty arrays.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error between true and predicted ratings.

    Args:
        y_true: Array-like of ground-truth ratings.
        y_pred: Array-like of predicted ratings (same length as *y_true*).

    Returns:
        MAE as a float.

    Raises:
        ValueError: If the input arrays have different lengths or are empty.

    Example::

        >>> compute_mae([3, 4, 5], [3.5, 3.5, 4.5])
        0.5
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute MAE on empty arrays.")
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Ranking / retrieval metrics  (helpers)
# ---------------------------------------------------------------------------

def _build_relevant_sets(
    test_df: pd.DataFrame,
    relevance_threshold: float = 3.5,
) -> Dict[int, Set[int]]:
    """Build per-user sets of relevant movie_ids from the test DataFrame.

    A movie is relevant if its test rating >= *relevance_threshold*.

    Args:
        test_df: DataFrame with columns [user_id, movie_id, rating].
        relevance_threshold: Minimum rating for an item to be relevant.

    Returns:
        Dict mapping user_id -> set of relevant movie_ids.
    """
    relevant: Dict[int, Set[int]] = defaultdict(set)
    mask = test_df["rating"] >= relevance_threshold
    for row in test_df.loc[mask].itertuples(index=False):
        relevant[int(row.user_id)].add(int(row.movie_id))
    return dict(relevant)


# ---------------------------------------------------------------------------
# MAP@K
# ---------------------------------------------------------------------------

def compute_map_at_k(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> float:
    """Compute Mean Average Precision at K (MAP@K).

    For each user, Average Precision at K is defined as::

        AP@K = (1 / R) * sum_{i=1}^{K} [ rel(i) * Precision@i ]

    where:
    - ``rel(i)`` is 1 if the i-th recommended item is relevant, else 0
    - ``Precision@i`` = (# relevant items in top-i) / i
    - ``R`` = total number of relevant items for that user in the test set

    MAP@K is the mean of AP@K over all users who have at least one
    relevant item in the test set.  Users with no relevant test items
    are excluded (they contribute no information about ranking quality).

    Args:
        recommendations: Dict mapping user_id to a ranked list of
                         recommended movie_ids (most relevant first).
        test_df: Test DataFrame with [user_id, movie_id, rating].
        k: Cutoff rank.  Default 10.
        relevance_threshold: Minimum test rating to consider an item
                             relevant.  Default 3.5.

    Returns:
        MAP@K as a float in [0, 1].  Returns 0.0 if no users have
        relevant items or no recommendations are provided.

    Example::

        >>> recs = {1: [101, 102, 103]}
        >>> test = pd.DataFrame({"user_id": [1, 1], "movie_id": [101, 103],
        ...                      "rating": [5.0, 4.0]})
        >>> compute_map_at_k(recs, test, k=3, relevance_threshold=3.5)
        0.8333...
    """
    relevant_sets = _build_relevant_sets(test_df, relevance_threshold)

    if not relevant_sets or not recommendations:
        return 0.0

    ap_scores: List[float] = []

    for user_id, rec_list in recommendations.items():
        rel_set = relevant_sets.get(user_id)
        if not rel_set:
            continue  # skip users with no relevant test items

        R = len(rel_set)
        hits = 0
        ap_sum = 0.0

        for i, movie_id in enumerate(rec_list[:k], start=1):
            if movie_id in rel_set:
                hits += 1
                precision_at_i = hits / i
                ap_sum += precision_at_i

        ap_at_k = ap_sum / R
        ap_scores.append(ap_at_k)

    if not ap_scores:
        return 0.0

    return float(np.mean(ap_scores))


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

def compute_precision_at_k(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> float:
    """Compute mean Precision at K across users.

    Precision@K for a single user is the fraction of recommended items
    (up to rank K) that are relevant in the test set.

    Args:
        recommendations: Dict mapping user_id to ranked list of movie_ids.
        test_df: Test DataFrame with [user_id, movie_id, rating].
        k: Cutoff rank.  Default 10.
        relevance_threshold: Minimum test rating for relevance.

    Returns:
        Mean Precision@K as a float in [0, 1].

    Example::

        >>> recs = {1: [101, 102, 103]}
        >>> test = pd.DataFrame({"user_id": [1, 1, 1],
        ...     "movie_id": [101, 102, 103], "rating": [5, 2, 4]})
        >>> compute_precision_at_k(recs, test, k=3, relevance_threshold=3.5)
        0.6666...
    """
    relevant_sets = _build_relevant_sets(test_df, relevance_threshold)

    if not relevant_sets or not recommendations:
        return 0.0

    precisions: List[float] = []

    for user_id, rec_list in recommendations.items():
        rel_set = relevant_sets.get(user_id)
        if rel_set is None:
            continue

        top_k = rec_list[:k]
        hits = sum(1 for m in top_k if m in rel_set)
        precisions.append(hits / k)

    if not precisions:
        return 0.0

    return float(np.mean(precisions))


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    recommendations: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> float:
    """Compute mean Recall at K across users.

    Recall@K for a single user is the fraction of relevant test items
    that appear in the user's top-K recommendation list.

    Args:
        recommendations: Dict mapping user_id to ranked list of movie_ids.
        test_df: Test DataFrame with [user_id, movie_id, rating].
        k: Cutoff rank.  Default 10.
        relevance_threshold: Minimum test rating for relevance.

    Returns:
        Mean Recall@K as a float in [0, 1].

    Example::

        >>> recs = {1: [101, 102]}
        >>> test = pd.DataFrame({"user_id": [1, 1, 1],
        ...     "movie_id": [101, 103, 104], "rating": [5, 4, 5]})
        >>> compute_recall_at_k(recs, test, k=2, relevance_threshold=3.5)
        0.3333...
    """
    relevant_sets = _build_relevant_sets(test_df, relevance_threshold)

    if not relevant_sets or not recommendations:
        return 0.0

    recalls: List[float] = []

    for user_id, rec_list in recommendations.items():
        rel_set = relevant_sets.get(user_id)
        if not rel_set:
            continue

        top_k = rec_list[:k]
        hits = sum(1 for m in top_k if m in rel_set)
        recalls.append(hits / len(rel_set))

    if not recalls:
        return 0.0

    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_coverage(
    recommendations: Dict[int, List[int]],
    n_total_items: int,
) -> float:
    """Compute catalogue coverage of the recommendation lists.

    Coverage is the fraction of all catalogue items that appear in at
    least one user's recommendation list.

    Args:
        recommendations: Dict mapping user_id to list of movie_ids.
        n_total_items: Total number of items in the catalogue.

    Returns:
        Coverage as a float in [0, 1].

    Raises:
        ValueError: If *n_total_items* is <= 0.

    Example::

        >>> recs = {1: [101, 102], 2: [102, 103]}
        >>> compute_coverage(recs, n_total_items=5)
        0.6
    """
    if n_total_items <= 0:
        raise ValueError(f"n_total_items must be > 0, got {n_total_items}")

    if not recommendations:
        return 0.0

    recommended_items: Set[int] = set()
    for rec_list in recommendations.values():
        recommended_items.update(rec_list)

    return len(recommended_items) / n_total_items
