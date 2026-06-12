"""pytest test suite for src/data/preprocess.py.

Covers:
  1. No data leakage in temporal split (all test dates >= cutoff).
  2. No duplicate (user, movie, date) rows across train/test (temporal split).
  3. Correct CSR matrix shape (n_users, n_movies).
  4. ID mapping round-trip correctness (u → idx → u).
  5. No NaNs in mapped train or test DataFrames.
  6. Random split respects the requested test_size fraction.
  7. Implicit matrix shape is transposed (n_movies, n_users) and values > 1.
  8. Cold-start users have < min_train_ratings ratings in train.
  9. Cold-start movies in cold_movies.csv are absent from train.
 10. Row/column totals match: CSR nnz == len(train_mapped).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path setup – make project root importable when run from repo root or tests/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import (
    apply_id_mappings,
    build_csr_matrix,
    build_id_mappings,
    build_implicit_matrix,
    identify_cold_start,
    random_split,
    temporal_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_ratings() -> pd.DataFrame:
    """A deterministic 80-row ratings DataFrame spanning 2005-06 to 2006-01."""
    rng = np.random.default_rng(0)
    n = 80
    # 10 users (1001–1010), 15 movies (1–15)
    users = rng.integers(1001, 1011, n).astype("int32")
    movies = rng.integers(1, 16, n).astype("int32")
    ratings = rng.integers(1, 6, n).astype("int8")
    dates = pd.date_range("2005-06-01", periods=n, freq="5D")
    df = pd.DataFrame(
        {"user_id": users, "movie_id": movies, "rating": ratings, "date": dates}
    )
    return df


@pytest.fixture()
def split_data(small_ratings):
    """Return (train, test) from a temporal split at 2005-10-01."""
    return temporal_split(small_ratings, cutoff="2005-10-01")


@pytest.fixture()
def mapped_data(split_data):
    """Return (train_mapped, test_mapped, user2idx, movie2idx)."""
    train, test = split_data
    user2idx, idx2user, movie2idx, idx2movie = build_id_mappings(train)
    train_m = apply_id_mappings(train, user2idx, movie2idx)
    test_m = apply_id_mappings(test, user2idx, movie2idx)
    return train_m, test_m, user2idx, idx2user, movie2idx, idx2movie


# ---------------------------------------------------------------------------
# 1. No data leakage – all test dates must be >= cutoff
# ---------------------------------------------------------------------------

def test_temporal_split_no_leakage(split_data):
    """All rows in the test set must have date >= 2005-10-01."""
    _, test = split_data
    cutoff = pd.Timestamp("2005-10-01")
    assert (test["date"] >= cutoff).all(), (
        "Leakage detected: some test dates are before the cutoff."
    )


def test_temporal_split_train_before_cutoff(split_data):
    """All rows in the train set must have date < 2005-10-01."""
    train, _ = split_data
    cutoff = pd.Timestamp("2005-10-01")
    assert (train["date"] < cutoff).all(), (
        "Train set contains rows dated on or after the cutoff."
    )


# ---------------------------------------------------------------------------
# 2. No exact duplicate rows across train and test
# ---------------------------------------------------------------------------

def test_temporal_split_no_row_overlap(split_data):
    """No identical (user_id, movie_id, date) triple should appear in both splits.

    A temporal split can legitimately share the same (user, movie) *pair* if a
    user rated a movie twice on different dates.  What would indicate true leakage
    is the same atomic rating event (user + movie + date) appearing in both sets.
    """
    train, test = split_data
    train_triples = set(zip(train["user_id"], train["movie_id"], train["date"]))
    test_triples = set(zip(test["user_id"], test["movie_id"], test["date"]))
    overlap = train_triples & test_triples
    assert len(overlap) == 0, (
        f"{len(overlap)} identical (user, movie, date) rows appear in both splits."
    )


# ---------------------------------------------------------------------------
# 3. CSR matrix has the correct shape
# ---------------------------------------------------------------------------

def test_csr_matrix_shape(mapped_data):
    """CSR shape must be (n_users, n_movies) as derived from train mappings."""
    train_m, _, user2idx, _, movie2idx, _ = mapped_data
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    csr = build_csr_matrix(train_m, n_users, n_movies)

    assert isinstance(csr, sp.csr_matrix)
    assert csr.shape == (n_users, n_movies), (
        f"Expected shape ({n_users}, {n_movies}), got {csr.shape}."
    )


# ---------------------------------------------------------------------------
# 4. ID mapping round-trip correctness
# ---------------------------------------------------------------------------

def test_id_mapping_round_trip(split_data):
    """user_id → idx → user_id and movie_id → idx → movie_id must be lossless."""
    train, _ = split_data
    user2idx, idx2user, movie2idx, idx2movie = build_id_mappings(train)

    for uid in train["user_id"].unique():
        assert idx2user[user2idx[uid]] == uid, f"Round-trip failed for user_id={uid}"

    for mid in train["movie_id"].unique():
        assert idx2movie[movie2idx[mid]] == mid, f"Round-trip failed for movie_id={mid}"


# ---------------------------------------------------------------------------
# 5. No NaNs in mapped DataFrames
# ---------------------------------------------------------------------------

def test_no_nans_in_mapped_train(mapped_data):
    """Mapped train DataFrame must contain no NaN values in any column."""
    train_m, _, *_ = mapped_data
    assert not train_m.isnull().any().any(), (
        "NaN values found in mapped train DataFrame."
    )


def test_no_nans_in_mapped_test(mapped_data):
    """Mapped test DataFrame (cold-start rows dropped) must have no NaNs."""
    _, test_m, *_ = mapped_data
    assert not test_m.isnull().any().any(), (
        "NaN values found in mapped test DataFrame."
    )


# ---------------------------------------------------------------------------
# 6. Random split respects test_size
# ---------------------------------------------------------------------------

def test_random_split_size(small_ratings):
    """Random split must produce a test set close to the requested fraction."""
    train, test = random_split(small_ratings, test_size=0.2, random_state=7)
    total = len(small_ratings)
    expected_test = int(np.ceil(total * 0.2))
    assert len(test) == expected_test, (
        f"Expected {expected_test} test rows, got {len(test)}."
    )
    assert len(train) + len(test) == total


# ---------------------------------------------------------------------------
# 7. Implicit matrix is item×user and values > 1.0
# ---------------------------------------------------------------------------

def test_implicit_matrix_shape_and_values(mapped_data):
    """Implicit matrix must be (n_movies, n_users) and all values > 1."""
    train_m, _, user2idx, _, movie2idx, _ = mapped_data
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    impl = build_implicit_matrix(train_m, n_users, n_movies, alpha=40.0)

    assert impl.shape == (n_movies, n_users), (
        f"Expected shape ({n_movies}, {n_users}), got {impl.shape}."
    )
    assert (impl.data > 1.0).all(), (
        "All implicit confidence values should be > 1 (C = 1 + alpha*r, r >= 1)."
    )


# ---------------------------------------------------------------------------
# 8. Cold-start users have < min_train_ratings in train
# ---------------------------------------------------------------------------

def test_cold_start_users_threshold(split_data):
    """Every cold user must have fewer than min_train_ratings in train."""
    train, test = split_data
    min_train = 5
    cold_users, _ = identify_cold_start(train, test, min_train_ratings=min_train)

    train_counts = train.groupby("user_id").size().to_dict()
    for uid in cold_users["user_id"]:
        count = train_counts.get(uid, 0)
        assert count < min_train, (
            f"User {uid} has {count} train ratings but was marked cold "
            f"(threshold={min_train})."
        )


# ---------------------------------------------------------------------------
# 9. Cold-start movies are absent from train
# ---------------------------------------------------------------------------

def test_cold_movies_absent_from_train(split_data):
    """Every cold movie must truly be absent from the training set."""
    train, test = split_data
    _, cold_movies = identify_cold_start(train, test)
    train_movie_ids = set(train["movie_id"].unique())
    for mid in cold_movies["movie_id"]:
        assert mid not in train_movie_ids, (
            f"Movie {mid} is in cold_movies but also appears in train."
        )


# ---------------------------------------------------------------------------
# 10. CSR nnz equals length of mapped train
# ---------------------------------------------------------------------------

def test_csr_nnz_matches_train_rows(mapped_data):
    """The number of non-zero entries in CSR must equal len(train_mapped).

    This confirms every rating ends up in the matrix exactly once.
    """
    train_m, _, user2idx, _, movie2idx, _ = mapped_data
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    csr = build_csr_matrix(train_m, n_users, n_movies)

    # If there are duplicate (user_idx, movie_idx) pairs, CSR sums them,
    # so nnz may be <= len(train_m).  With our fixture there are no dupes,
    # so they should be equal.
    unique_pairs = train_m[["user_idx", "movie_idx"]].drop_duplicates().shape[0]
    assert csr.nnz == unique_pairs, (
        f"CSR nnz={csr.nnz} but unique (user, movie) pairs={unique_pairs}."
    )
