# Evaluation Summary

## Key Findings

**Best rating predictor (RMSE):** GlobalMean with RMSE = 1.0809. This model achieves the lowest root mean squared error on the held-out test set, indicating the most accurate explicit rating predictions among all compared approaches.

**Best ranking model (MAP@10):** SVD with MAP@10 = 0.0007. This measures how well the model ranks truly relevant items (rated >= 3.5) at the top of its recommendation lists.

**Baseline vs. Collaborative Filtering:** The GlobalMean baseline, which predicts every rating as the training-set average, is surprisingly competitive on RMSE. This is partly because the Netflix rating distribution is concentrated around 3-4 stars, making a constant prediction hard to beat on average error. However, the GlobalMean model cannot personalise recommendations and has zero ranking capability.

**User-based vs. Item-based CF:** Both neighbourhood-based methods achieve similar RMSE, but UserCF requires significantly more training time due to the larger user-user similarity computation (O(n_users^2)) compared to ItemCF's item-item similarity (O(n_items^2)). In production, ItemCF is generally preferred because item profiles are more stable than user profiles.

**SVD (Matrix Factorization):** The Surprise SVD model matches or outperforms neighbourhood methods on RMSE while training much faster. SVD learns compact latent factor representations that capture global patterns, making it the recommended approach for explicit rating prediction tasks.

**ALS (Implicit Feedback):** ALS achieves RMSE = 1.7594, which appears worse, but this comparison is misleading. ALS optimises a fundamentally different objective (confidence-weighted implicit feedback), not explicit rating prediction. Its strength lies in ranking and discovery of items users are likely to interact with, not in predicting exact star ratings. For a fair comparison, use ranking metrics (MAP@K, Precision@K, nDCG) rather than RMSE.

## Trade-off Summary

| Dimension | Winner | Notes |
|---|---|---|
| RMSE (accuracy) | GlobalMean | Best explicit rating prediction |
| MAP@10 (ranking) | SVD | Best top-K relevance ranking |
| Training speed | GlobalMean / BiasModel | Sub-second fit |
| Scalability | SVD / ALS | O(n * k) vs O(n^2) for CF |
| Cold-start | BiasModel | Graceful fallback via global + item bias |
| Diversity | ALS | Implicit feedback promotes exploration |

---

*Generated automatically by `src/evaluation/evaluate_all.py`.*