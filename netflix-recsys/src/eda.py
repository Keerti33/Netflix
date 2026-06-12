"""Exploratory Data Analysis (EDA) module for Netflix Prize Recommendation System.

This script performs a comprehensive analysis of the processed ratings and movie
data, saves diagnostic plots to outputs/eda/, and summarizes key insights.
It also exposes a function to retrieve key statistics.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_key_stats(ratings_path: str) -> dict:
    """Computes and returns key dataset statistics.

    Args:
        ratings_path: Path to the ratings parquet file.

    Returns:
        A dictionary containing:
            - sparsity (float): Exact dataset sparsity.
            - n_users (int): Number of unique users.
            - n_movies (int): Number of unique movies.
            - n_ratings (int): Number of total ratings.
            - mean_rating (float): Mean rating.
    """
    df = pd.read_parquet(ratings_path)
    
    n_users = int(df['user_id'].nunique())
    n_movies = int(df['movie_id'].nunique())
    n_ratings = len(df)
    mean_rating = float(df['rating'].mean())
    
    # Sparsity = 1 - (observed / (users * movies))
    possible_ratings = n_users * n_movies
    sparsity = 1.0 - (n_ratings / possible_ratings) if possible_ratings > 0 else 1.0
    
    return {
        'sparsity': sparsity,
        'n_users': n_users,
        'n_movies': n_movies,
        'n_ratings': n_ratings,
        'mean_rating': mean_rating
    }


def run_eda(ratings_path: str, movies_path: str, output_dir: str):
    """Performs full exploratory data analysis and saves plots as PNGs.

    Args:
        ratings_path: Path to ratings Parquet file.
        movies_path: Path to movies Parquet file.
        output_dir: Path to directory to save plots.
    """
    start_time = time.time()
    
    # Initialize seaborn style with Netflix-inspired branding colors
    sns.set_theme(style="whitegrid")
    
    # Custom color palette (Netflix Red: #E50914, Charcoal: #221F1F, Soft Red: #F15A5A)
    netflix_colors = ["#E50914", "#221F1F", "#F15A5A", "#F5F5F1", "#A6A6A6"]
    sns.set_palette(sns.color_palette(netflix_colors))
    
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(ratings_path):
        print(f"Error: Ratings file not found at {ratings_path}. Run pipeline first.")
        sys.exit(1)
    if not os.path.exists(movies_path):
        print(f"Warning: Movies file not found at {movies_path}. Titles won't be available.")
        movies_df = None
    else:
        movies_df = pd.read_parquet(movies_path)
        
    print(f"Loading ratings dataset from {ratings_path}...")
    ratings_df = pd.read_parquet(ratings_path)
    
    print("\n" + "="*50)
    print(" SECTION A: DATASET OVERVIEW ")
    print("="*50)
    
    # Shape & memory usage
    shape = ratings_df.shape
    memory_usage_bytes = ratings_df.memory_usage(deep=True).sum()
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    print(f"Ratings DataFrame shape: {shape[0]:,} rows, {shape[1]} columns")
    print(f"Memory usage: {memory_usage_mb:.2f} MB")
    print("\nData types:")
    print(ratings_df.dtypes)
    
    # Missing values
    missing_vals = ratings_df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_vals)
    
    # Date range
    min_date = ratings_df['date'].min()
    max_date = ratings_df['date'].max()
    print(f"\nDate range of ratings: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    print("\n" + "="*50)
    print(" SECTION B: RATING DISTRIBUTION ")
    print("="*50)
    
    # Descriptive stats
    mean_val = ratings_df['rating'].mean()
    median_val = ratings_df['rating'].median()
    std_val = ratings_df['rating'].std()
    print(f"Mean Rating: {mean_val:.4f}")
    print(f"Median Rating: {median_val:.1f}")
    print(f"Standard Deviation: {std_val:.4f}")
    
    # Distribution frequency
    counts = ratings_df['rating'].value_counts().sort_index()
    percentages = ratings_df['rating'].value_counts(normalize=True).sort_index() * 100
    print("\nRating value counts & percentages:")
    for rating, count in counts.items():
        print(f"  Rating {rating}: {count:,} ({percentages[rating]:.2f}%)")
        
    # Visualisation
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="Reds_r", legend=False)
    plt.title("Distribution of Ratings", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Rating (1-5)", fontsize=12)
    plt.ylabel("Number of Ratings", fontsize=12)
    # Add labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(out_path / "rating_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nRating skew note: The ratings are skewed towards higher values (4 and 5 make up a large portion).")
    
    print("\n" + "="*50)
    print(" SECTION C: USER ACTIVITY ")
    print("="*50)
    
    user_counts = ratings_df['user_id'].value_counts()
    
    # Plot user activity distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=50, kde=True, log_scale=True, color="#E50914")
    plt.title("Distribution of Ratings per User (Log Scale)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Ratings submitted by User (Log Scale)", fontsize=12)
    plt.ylabel("Count of Users", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "user_ratings_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Top 20 active users
    print("Top 20 most active users:")
    print(user_counts.head(20).to_string())
    
    # Pareto Analysis
    sorted_user_counts = user_counts.sort_values(ascending=False)
    cum_ratings = sorted_user_counts.cumsum()
    total_ratings = ratings_df.shape[0]
    cum_percentage = (cum_ratings / total_ratings) * 100
    
    user_percentage = (np.arange(1, len(user_counts) + 1) / len(user_counts)) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(user_percentage, cum_percentage, color="#E50914", linewidth=2.5)
    plt.axhline(80, color='gray', linestyle='--', alpha=0.7)
    # Find user % for 80% ratings
    idx_80 = np.where(cum_percentage >= 80)[0][0]
    pct_user_80 = user_percentage[idx_80]
    plt.axvline(pct_user_80, color='gray', linestyle='--', alpha=0.7)
    
    plt.title("Pareto Analysis of User Activity", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cumulative % of Users (sorted by activity)", fontsize=12)
    plt.ylabel("Cumulative % of Ratings", fontsize=12)
    plt.text(pct_user_80 + 2, 40, f"{pct_user_80:.1f}% of users\naccount for 80% of ratings", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path / "user_pareto_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPareto Insight: {pct_user_80:.2f}% of users account for 80% of total ratings.")
    
    print("\n" + "="*50)
    print(" SECTION D: MOVIE POPULARITY ")
    print("="*50)
    
    movie_counts = ratings_df['movie_id'].value_counts()
    
    # Plot movie ratings distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_counts, bins=50, kde=True, log_scale=True, color="#221F1F")
    plt.title("Distribution of Ratings per Movie (Log Scale)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Ratings for Movie (Log Scale)", fontsize=12)
    plt.ylabel("Count of Movies", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "movie_ratings_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Match with movie titles if available
    top_movies_df = pd.DataFrame(movie_counts.head(20)).reset_index()
    top_movies_df.columns = ['movie_id', 'rating_count']
    if movies_df is not None:
        top_movies_df = top_movies_df.merge(movies_df, on='movie_id', how='left')
    print("Top 20 most-rated movies:")
    print(top_movies_df.to_string(index=False))
    
    bottom_movies_df = pd.DataFrame(movie_counts.tail(20)).reset_index()
    bottom_movies_df.columns = ['movie_id', 'rating_count']
    if movies_df is not None:
        bottom_movies_df = bottom_movies_df.merge(movies_df, on='movie_id', how='left')
    print("\nBottom 20 least-rated movies:")
    print(bottom_movies_df.to_string(index=False))
    
    print("\n" + "="*50)
    print(" SECTION E: SPARSITY ANALYSIS ")
    print("="*50)
    
    # Sparsity
    stats = get_key_stats(ratings_path)
    sparsity_val = stats['sparsity']
    n_users = stats['n_users']
    n_movies = stats['n_movies']
    n_ratings = stats['n_ratings']
    print(f"Unique Users (U): {n_users:,}")
    print(f"Unique Movies (M): {n_movies:,}")
    print(f"Observed Ratings (R): {n_ratings:,}")
    print(f"Exact Matrix Sparsity: {sparsity_val * 100:.6f}%")
    
    # Heatmap of 200 users x 200 movies submatrix
    # Extract top 200 users and movies
    top_users = user_counts.head(200).index
    top_movies = movie_counts.head(200).index
    
    submatrix_df = ratings_df[ratings_df['user_id'].isin(top_users) & ratings_df['movie_id'].isin(top_movies)]
    pivot_matrix = submatrix_df.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Reindex to make sure it is exactly 200x200
    pivot_matrix = pivot_matrix.reindex(index=top_users, columns=top_movies)
    
    plt.figure(figsize=(12, 10))
    # We use a custom color scheme where NaN is light gray and ratings are red scale
    sns.heatmap(pivot_matrix, cmap="YlOrRd", cbar_kws={'label': 'Rating'}, mask=pivot_matrix.isnull(),
                xticklabels=False, yticklabels=False, facecolor="#F0F0F0")
    plt.title("Sparsity Heatmap (Top 200 Users × Top 200 Movies)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Movies (Sorted by popularity)", fontsize=12)
    plt.ylabel("Users (Sorted by activity)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "sparsity_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nSparsity visual saved. Dense interaction displays standard collaborative filtering patterns.")
    
    print("\n" + "="*50)
    print(" SECTION F: TEMPORAL TRENDS ")
    print("="*50)
    
    # Group by month
    ratings_df['year_month'] = ratings_df['date'].dt.to_period('M')
    monthly_data = ratings_df.groupby('year_month').agg(
        rating_count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    ).reset_index()
    monthly_data['year_month_dt'] = monthly_data['year_month'].dt.to_timestamp()
    
    # Ratings count over time
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data['year_month_dt'], monthly_data['rating_count'], color="#E50914", marker='o', linewidth=2)
    plt.title("Number of Ratings Over Time (Monthly)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Monthly Ratings count", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "ratings_over_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Average rating over time (daily rating mean with rolling mean)
    daily_mean = ratings_df.groupby('date')['rating'].mean().reset_index()
    daily_mean['rolling_mean'] = daily_mean['rating'].rolling(window=30, min_periods=7).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_mean['date'], daily_mean['rating'], alpha=0.3, color="#A6A6A6", label="Daily Mean")
    plt.plot(daily_mean['date'], daily_mean['rolling_mean'], color="#E50914", linewidth=2.5, label="30-Day Rolling Mean")
    plt.title("Average Rating Over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Rating Value", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path / "average_rating_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Temporal trend plots saved. Note any gradual changes or seasonality in rating patterns.")
    
    print("\n" + "="*50)
    print(" SECTION G: RATING BIAS ")
    print("="*50)
    
    # Per-user mean rating distribution
    user_means = ratings_df.groupby('user_id')['rating'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(user_means, bins=50, kde=True, color="#E50914")
    plt.title("Distribution of Per-User Mean Ratings", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mean Rating per User", fontsize=12)
    plt.ylabel("Count of Users", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "user_mean_rating_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Per-movie mean rating vs number of ratings
    movie_stats = ratings_df.groupby('movie_id').agg(
        mean_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    plt.figure(figsize=(10, 6))
    # Using alpha to deal with overplotting
    sns.scatterplot(data=movie_stats, x='rating_count', y='mean_rating', alpha=0.4, color="#221F1F", edgecolor=None)
    plt.xscale('log')
    plt.title("Per-Movie Mean Rating vs. Popularity (Number of Ratings)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Ratings per Movie (Log Scale)", fontsize=12)
    plt.ylabel("Mean Rating", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "movie_mean_vs_count.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Rating bias analysis plots saved.")
    
    print("\n" + "="*50)
    print(" SECTION H: KEY INSIGHTS SUMMARY ")
    print("="*50)
    
    insights = [
        f"1. **Sparsity Challenge**: The dataset is highly sparse ({sparsity_val * 100:.4f}% sparsity). High sparsity requires collaborative filtering approaches such as matrix factorization (SVD, ALS) or deep learning embeddings rather than simple neighborhood methods.",
        f"2. **Rating Skewness**: The average rating is high ({mean_val:.2f} out of 5), with standard deviation {std_val:.2f}. The median rating is {median_val:.1f}. Recommendation systems must adjust for this global upward bias.",
        f"3. **Pareto Distribution of Activity**: A small fraction of users ({pct_user_80:.2f}%) contributes to 80% of all ratings. Active user models will dominate training, meaning we must evaluate performance separately on tail/inactive users to ensure fairness.",
        "4. **User Bias Variance**: The distribution of per-user mean ratings is wide, indicating severe baseline variance (some users are systematic 'harsh critics' while others are 'generous'). Subtracting user-specific baselines (mean-centering) is a crucial normalization step.",
        "5. **Movie Popularity Bias**: Popular movies (top-rated) show a distinct distribution pattern, typically leaning towards higher ratings, while tail/less popular movies show higher variance in their average ratings. The system must address popularity bias to avoid recommending only blockbuster films.",
        "6. **Temporal Evolution**: The average rating is not static over time, suggesting concept drift due to platform changes, user onboarding, or rating culture. Models should incorporate temporal decay or dynamic user bias variables.",
    ]
    
    for insight in insights:
        print(insight)
        
    elapsed_time = time.time() - start_time
    print(f"\nEDA completed in {elapsed_time:.2f} seconds.")


def main():
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(description="Run full EDA pipeline and output diagnostic plots.")
    parser.add_argument(
        "--ratings_path",
        type=str,
        default="data/processed/ratings_sample.parquet",
        help="Path to the sample ratings parquet file."
    )
    parser.add_argument(
        "--movies_path",
        type=str,
        default="data/processed/movies.parquet",
        help="Path to the movies metadata parquet file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eda",
        help="Directory to save diagnostic plots."
    )
    
    args = parser.parse_args()
    
    run_eda(
        ratings_path=args.ratings_path,
        movies_path=args.movies_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
