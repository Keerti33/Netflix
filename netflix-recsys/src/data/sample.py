"""Data sampling and filtering module for Netflix Prize Recommendation System.

This script filters the ratings dataset to exclude sparse users (with < 20 ratings)
and sparse movies (with < 50 ratings), then generates a reproducible,
stratified sample of N ratings preserving the overall rating distribution.
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_ratings(df: pd.DataFrame, min_user_ratings: int = 20, min_movie_ratings: int = 50) -> pd.DataFrame:
    """Filters ratings to ensure active users and popular movies.

    Runs iteratively (k-core style) until the filtering constraints are stable.

    Args:
        df: Input ratings DataFrame with columns 'user_id' and 'movie_id'.
        min_user_ratings: Minimum number of ratings a user must have.
        min_movie_ratings: Minimum number of ratings a movie must have.

    Returns:
        Filtered pandas DataFrame.
    """
    print(f"Applying filtering: users with >= {min_user_ratings} ratings, movies with >= {min_movie_ratings} ratings...")
    initial_rows = len(df)
    prev_rows = 0
    iterations = 0
    
    while len(df) != prev_rows:
        prev_rows = len(df)
        iterations += 1
        
        # Filter users
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_user_ratings].index)]
        
        # Filter movies
        movie_counts = df['movie_id'].value_counts()
        df = df[df['movie_id'].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
        
        print(f"Iteration {iterations}: remaining ratings = {len(df)} ({len(df)/initial_rows*100:.2f}% of original)")
        
    print(f"Filtering converged after {iterations} iterations. Final rating count = {len(df)}")
    return df


def print_rating_distribution(df: pd.DataFrame, label: str):
    """Prints the distribution of ratings in the dataset.

    Args:
        df: Pandas DataFrame containing a 'rating' column.
        label: Descriptive label for the printed output.
    """
    dist = df['rating'].value_counts(normalize=True).sort_index() * 100
    print(f"\nRating distribution ({label}):")
    for rating, pct in dist.items():
        print(f"  Rating {rating}: {pct:.4f}%")
    print(f"  Total ratings: {len(df):,}\n")


def sample_ratings(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """Generates a reproducible stratified sample of N ratings preserving the rating distribution.

    Args:
        df: Input filtered ratings DataFrame.
        n_samples: Number of samples to draw.
        random_state: Seed for random number generator.

    Returns:
        Sampled pandas DataFrame.
    """
    total_ratings = len(df)
    
    if n_samples <= 0:
        print("Sample size <= 0. Skipping sampling and keeping all filtered ratings.")
        return df

    if n_samples >= total_ratings:
        print(f"Requested sample size {n_samples:,} is greater than or equal to total ratings ({total_ratings:,}). Returning full filtered dataset.")
        return df

    print(f"Sampling {n_samples:,} ratings from {total_ratings:,} ratings (stratified by rating)...")
    
    try:
        df_sample, _ = train_test_split(
            df,
            train_size=n_samples,
            random_state=random_state,
            stratify=df['rating']
        )
    except ValueError as e:
        print(f"Warning: Stratified sampling failed due to class size constraints: {e}")
        print("Falling back to random sampling without stratification.")
        df_sample = df.sample(n=n_samples, random_state=random_state)
    return df_sample


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Filter and sample processed ratings dataset.")
    parser.add_argument(
        "--raw_ratings",
        type=str,
        default="data/processed/ratings.parquet",
        help="Path to the combined ratings Parquet file."
    )
    parser.add_argument(
        "--output_sample",
        type=str,
        default="data/processed/ratings_sample.parquet",
        help="Path to save the sampled ratings Parquet file."
    )
    parser.add_argument(
        "-n", "--sample_size",
        type=int,
        default=5000000,
        help="Number of ratings to sample. Set to 0 to skip sampling."
    )
    parser.add_argument(
        "--min_user_ratings",
        type=int,
        default=20,
        help="Minimum number of ratings per user."
    )
    parser.add_argument(
        "--min_movie_ratings",
        type=int,
        default=50,
        help="Minimum number of ratings per movie."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    
    input_path = Path(args.raw_ratings)
    output_path = Path(args.output_sample)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist. Please run ingest.py first.")
        sys.exit(1)
        
    start_time = time.time()
    
    print(f"Reading processed ratings from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} ratings.")
    
    print_rating_distribution(df, "Original Processed Dataset")
    
    # 1. Filter dataset
    df_filtered = filter_ratings(
        df, 
        min_user_ratings=args.min_user_ratings, 
        min_movie_ratings=args.min_movie_ratings
    )
    
    print_rating_distribution(df_filtered, "Filtered Dataset")
    
    # 2. Sample dataset
    df_sampled = sample_ratings(
        df_filtered, 
        n_samples=args.sample_size, 
        random_state=args.seed
    )
    
    print_rating_distribution(df_sampled, f"Sampled Dataset (N={args.sample_size})")
    
    # 3. Save sample
    print(f"Saving sampled ratings to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sampled.to_parquet(output_path, index=False)
    print(f"Saved {len(df_sampled):,} ratings successfully.")
    
    elapsed_time = time.time() - start_time
    print(f"Data sampling pipeline completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
