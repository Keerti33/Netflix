"""Unit tests for recommendation evaluation metrics.

Covers edge cases and known-answer scenarios for all six metric functions
in ``src.evaluation.metrics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_coverage,
    compute_mae,
    compute_map_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_rmse,
)


# ---------------------------------------------------------------------------
# RMSE tests
# ---------------------------------------------------------------------------

class TestRMSE:
    """Tests for compute_rmse."""

    def test_perfect_predictions(self):
        """RMSE should be 0 when predictions exactly match truth."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_rmse(self):
        """RMSE of [3,4,5] vs [3.5,3.5,4.5] should be sqrt(0.25) ≈ 0.5."""
        y_true = [3.0, 4.0, 5.0]
        y_pred = [3.5, 3.5, 4.5]
        # errors: 0.5, -0.5, -0.5 → MSE = (0.25+0.25+0.25)/3 = 0.25
        expected = np.sqrt(0.25)
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-6)

    def test_single_element(self):
        """RMSE on a single element should be the absolute error."""
        assert compute_rmse([5.0], [3.0]) == pytest.approx(2.0, abs=1e-10)

    def test_empty_raises(self):
        """RMSE on empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_rmse([], [])

    def test_length_mismatch_raises(self):
        """RMSE with different-length arrays should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_rmse([1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# MAE tests
# ---------------------------------------------------------------------------

class TestMAE:
    """Tests for compute_mae."""

    def test_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        y = [1.0, 2.0, 3.0]
        assert compute_mae(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_known_mae(self):
        """MAE of [1,2,3] vs [2,3,4] should be 1.0."""
        assert compute_mae([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# MAP@K tests
# ---------------------------------------------------------------------------

class TestMAPAtK:
    """Tests for compute_map_at_k."""

    @pytest.fixture
    def test_df(self) -> pd.DataFrame:
        """Test DataFrame: user 1 has relevant items 101, 103, 105."""
        return pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1],
            "movie_id": [101, 102, 103, 104, 105],
            "rating": [5.0, 2.0, 4.0, 1.0, 5.0],
        })

    def test_perfect_ranking(self, test_df):
        """All relevant items at the top should give MAP@K = 1.0."""
        # Relevant items (rating >= 3.5): 101, 103, 105
        recs = {1: [101, 103, 105, 102, 104]}
        result = compute_map_at_k(recs, test_df, k=5, relevance_threshold=3.5)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_zero_relevant_recommendations(self, test_df):
        """Recommending only irrelevant items should give AP@K = 0."""
        recs = {1: [102, 104, 999, 998, 997]}
        result = compute_map_at_k(recs, test_df, k=5, relevance_threshold=3.5)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_partial_ranking(self, test_df):
        """Hand-verified partial ranking AP@K calculation."""
        # Rec list: [102(irrel), 101(rel), 104(irrel), 103(rel)]
        # rel items = {101, 103, 105}, R = 3
        # i=1: 102 not relevant → 0
        # i=2: 101 relevant → P@2 = 1/2 = 0.5
        # i=3: 104 not relevant → 0
        # i=4: 103 relevant → P@4 = 2/4 = 0.5
        # AP@4 = (0.5 + 0.5) / 3 = 1/3
        recs = {1: [102, 101, 104, 103]}
        result = compute_map_at_k(recs, test_df, k=4, relevance_threshold=3.5)
        assert result == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_no_users_with_relevant_items(self):
        """MAP@K should be 0 when no test user has relevant items."""
        test_df = pd.DataFrame({
            "user_id": [1, 1],
            "movie_id": [101, 102],
            "rating": [1.0, 2.0],
        })
        recs = {1: [101, 102]}
        result = compute_map_at_k(recs, test_df, k=2, relevance_threshold=3.5)
        assert result == pytest.approx(0.0)

    def test_empty_recommendations(self, test_df):
        """MAP@K with empty recs dict should be 0."""
        assert compute_map_at_k({}, test_df, k=10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Precision@K tests
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    """Tests for compute_precision_at_k."""

    def test_all_relevant(self):
        """Precision@K = 1.0 when all K recs are relevant."""
        test_df = pd.DataFrame({
            "user_id": [1, 1, 1],
            "movie_id": [101, 102, 103],
            "rating": [5.0, 4.0, 5.0],
        })
        recs = {1: [101, 102, 103]}
        result = compute_precision_at_k(recs, test_df, k=3, relevance_threshold=3.5)
        assert result == pytest.approx(1.0)

    def test_none_relevant(self):
        """Precision@K = 0.0 when none of the K recs are relevant."""
        test_df = pd.DataFrame({
            "user_id": [1, 1],
            "movie_id": [101, 102],
            "rating": [5.0, 5.0],
        })
        recs = {1: [999, 998, 997]}
        result = compute_precision_at_k(recs, test_df, k=3, relevance_threshold=3.5)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Recall@K tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    """Tests for compute_recall_at_k."""

    def test_full_recall(self):
        """Recall@K = 1.0 when all relevant items are recommended."""
        test_df = pd.DataFrame({
            "user_id": [1, 1],
            "movie_id": [101, 102],
            "rating": [5.0, 4.0],
        })
        recs = {1: [101, 102, 103, 104]}
        result = compute_recall_at_k(recs, test_df, k=4, relevance_threshold=3.5)
        assert result == pytest.approx(1.0)

    def test_partial_recall(self):
        """Recall@K = 0.5 when half the relevant items are recommended."""
        test_df = pd.DataFrame({
            "user_id": [1, 1, 1, 1],
            "movie_id": [101, 102, 103, 104],
            "rating": [5.0, 5.0, 1.0, 1.0],
        })
        # Only 101 is recommended, out of {101, 102} relevant
        recs = {1: [101, 999]}
        result = compute_recall_at_k(recs, test_df, k=2, relevance_threshold=3.5)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

class TestCoverage:
    """Tests for compute_coverage."""

    def test_full_coverage(self):
        """Coverage = 1.0 when every catalogue item is recommended."""
        recs = {1: [1, 2, 3], 2: [4, 5]}
        assert compute_coverage(recs, n_total_items=5) == pytest.approx(1.0)

    def test_empty_recommendations(self):
        """Coverage = 0.0 when no recs are given."""
        assert compute_coverage({}, n_total_items=100) == pytest.approx(0.0)

    def test_partial_coverage(self):
        """Coverage = 0.6 when 3 out of 5 items are recommended."""
        recs = {1: [101, 102], 2: [102, 103]}
        assert compute_coverage(recs, n_total_items=5) == pytest.approx(0.6)

    def test_invalid_n_items_raises(self):
        """Coverage with n_total_items <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_total_items"):
            compute_coverage({1: [1]}, n_total_items=0)
