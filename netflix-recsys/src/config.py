"""Configuration and constants for the Netflix recommendation system.

Centralizes all configuration values, making them easy to modify
and maintain across the project.

Usage::

    from src.config import RATING_SCALE, RANDOM_STATE, DEFAULT_K
    
    print(f"Rating scale: {RATING_SCALE}")
    print(f"Random state: {RANDOM_STATE}")
"""

# ============================================================================
# Rating system configuration
# ============================================================================

# Valid rating range (Netflix Prize uses 1-5 star ratings)
RATING_MIN = 1.0
RATING_MAX = 5.0
RATING_SCALE = (RATING_MIN, RATING_MAX)

# Relevance threshold for ranking metrics (4+ stars = relevant)
RELEVANCE_THRESHOLD = 3.5

# ============================================================================
# Model and training configuration
# ============================================================================

# Default random seed for reproducibility
RANDOM_STATE = 42

# Temporal split date for train/test separation
TEMPORAL_CUTOFF = "2005-10-01"

# SVD hyperparameters
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02

# ALS hyperparameters
ALS_N_FACTORS = 100
ALS_N_EPOCHS = 20
ALS_LEARNING_RATE = 0.01
ALS_REGULARIZATION = 0.01

# ============================================================================
# Recommendation and evaluation configuration
# ============================================================================

# Default number of recommendations to generate
DEFAULT_K = 10

# Ranking metrics cutoff values to evaluate at
EVAL_CUTOFFS = [5, 10, 20]

# Number of CV folds for hyperparameter tuning
CV_FOLDS = 3

# ============================================================================
# Data paths
# ============================================================================

# Raw data directory (Netflix Prize files)
RAW_DATA_DIR = "data/raw"

# Processed data directory
PROCESSED_DATA_DIR = "data/processed"

# Models output directory
MODELS_DIR = "outputs/models"

# Predictions output directory
PREDICTIONS_DIR = "outputs/predictions"

# Recommendations output directory
RECOMMENDATIONS_DIR = "outputs/recommendations"

# Results output directory
RESULTS_DIR = "outputs/results"

# ============================================================================
# Logging configuration
# ============================================================================

# Default logging level (INFO, DEBUG, WARNING, ERROR)
LOG_LEVEL = "INFO"

# Enable file logging
FILE_LOGGING_ENABLED = False

# Log file path (if FILE_LOGGING_ENABLED is True)
LOG_FILE = "logs/netflix_recsys.log"
