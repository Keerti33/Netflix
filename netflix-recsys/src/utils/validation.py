"""Data validation utilities for the recommendation system.

Provides functions to validate and sanitize input data,
ensuring data integrity across the pipeline.

Usage::

    from src.utils.validation import validate_ratings_df, validate_model_input
    
    validate_ratings_df(train_df)
    validate_model_input(user_id=123, movie_id=456)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def validate_ratings_df(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate that a ratings DataFrame has the required structure.

    Args:
        df: DataFrame to validate.
        name: Name for error messages (e.g., "train_df", "test_df").

    Raises:
        ValueError: If required columns are missing or data is invalid.
        TypeError: If the input is not a DataFrame.

    Example::

        validate_ratings_df(train_df, name="train_df")
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df)}")

    required_cols = {"user_id", "movie_id", "rating"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Has columns: {set(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"{name} is empty")

    if df["user_id"].isna().any() or df["movie_id"].isna().any() or df["rating"].isna().any():
        raise ValueError(f"{name} contains NaN values in required columns")

    if not np.all((df["rating"] >= 1.0) & (df["rating"] <= 5.0)):
        raise ValueError(
            f"{name} ratings must be in range [1, 5], "
            f"found range [{df['rating'].min()}, {df['rating'].max()}]"
        )

    if df["user_id"].dtype not in [np.int32, np.int64, int]:
        raise ValueError(f"{name} user_id must be integer type, got {df['user_id'].dtype}")

    if df["movie_id"].dtype not in [np.int32, np.int64, int]:
        raise ValueError(f"{name} movie_id must be integer type, got {df['movie_id'].dtype}")


def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "predictions",
) -> None:
    """Validate prediction arrays for metric computation.

    Args:
        y_true: Ground-truth ratings array.
        y_pred: Predicted ratings array.
        name: Name for error messages.

    Raises:
        ValueError: If arrays are invalid or mismatched.
        TypeError: If inputs are not array-like.

    Example::

        validate_predictions(test_ratings, pred_ratings, name="SVD predictions")
    """
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Could not convert {name} to numpy arrays: {e}")

    if y_true.size == 0:
        raise ValueError(f"{name} y_true is empty")
    if y_pred.size == 0:
        raise ValueError(f"{name} y_pred is empty")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"{name} shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError(f"{name} contains NaN values")


def validate_user_movie_ids(
    user_id: int,
    movie_id: int,
) -> Tuple[int, int]:
    """Validate and convert user and movie IDs.

    Args:
        user_id: User identifier.
        movie_id: Movie identifier.

    Returns:
        Tuple of (user_id, movie_id) as integers.

    Raises:
        ValueError: If IDs are invalid or negative.

    Example::

        uid, mid = validate_user_movie_ids(user_id=123, movie_id=456)
    """
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
    except (TypeError, ValueError) as e:
        raise ValueError(f"user_id and movie_id must be convertible to int: {e}")

    if user_id < 0:
        raise ValueError(f"user_id must be non-negative, got {user_id}")
    if movie_id < 0:
        raise ValueError(f"movie_id must be non-negative, got {movie_id}")

    return user_id, movie_id
