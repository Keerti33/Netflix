"""Data ingestion module for Netflix Prize Recommendation System.

This script parses the raw combined_data_*.txt files and the movie_titles.csv
file, casts the columns to memory-efficient data types, and saves the parsed
DataFrames as Parquet files.
"""

import argparse
import glob
import os
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_movie_titles(file_path: Path) -> pd.DataFrame:
    """Reads movie_titles.csv and parses it into a DataFrame.

    Handles cases where titles contain commas and years are missing (NULL).

    Args:
        file_path: Path to the movie_titles.csv file.

    Returns:
        A pandas DataFrame with columns: movie_id (int32), year (str), title (str).
    """
    print(f"Parsing movie titles from: {file_path}")
    records = []
    
    # Netflix movie_titles.csv is typically encoded in 'latin-1' or 'iso-8859-1'.
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by comma at most twice to avoid splitting titles that contain commas
            parts = line.split(',', 2)
            if len(parts) == 3:
                movie_id_str, year_str, title_str = parts
                records.append({
                    'movie_id': int(movie_id_str),
                    'year': year_str,
                    'title': title_str
                })
            elif len(parts) == 2:
                movie_id_str, title_str = parts
                records.append({
                    'movie_id': int(movie_id_str),
                    'year': 'NULL',
                    'title': title_str
                })
            else:
                records.append({
                    'movie_id': int(parts[0]),
                    'year': 'NULL',
                    'title': ''
                })
                
    df = pd.DataFrame(records)
    df['movie_id'] = df['movie_id'].astype('int32')
    df['year'] = df['year'].astype('str')
    df['title'] = df['title'].astype('str')
    
    print(f"Successfully parsed {len(df)} movies.")
    return df


def parse_ratings_file(file_path: Path) -> pd.DataFrame:
    """Parses a single combined_data_*.txt file line by line.

    Converts it to a pandas DataFrame with memory-efficient types.

    Args:
        file_path: Path to the combined_data_*.txt file.

    Returns:
        A pandas DataFrame with columns: user_id (int32), movie_id (int32),
        rating (int8), date (datetime64[ns]).
    """
    print(f"Parsing ratings from: {file_path.name}")
    user_ids = []
    movie_ids = []
    ratings = []
    dates = []
    
    current_movie_id = None
    
    # Read the file line by line to handle the custom structure efficiently
    # Count lines first for tqdm progress bar
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc=f"Reading {file_path.name}"):
            line = line.strip()
            if not line:
                continue
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                user_id_str, rating_str, date_str = line.split(',')
                user_ids.append(int(user_id_str))
                movie_ids.append(current_movie_id)
                ratings.append(int(rating_str))
                dates.append(date_str)
                
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'date': dates
    })
    
    # Optimize data types to reduce memory usage significantly
    df['user_id'] = df['user_id'].astype('int32')
    df['movie_id'] = df['movie_id'].astype('int32')
    df['rating'] = df['rating'].astype('int8')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    print(f"Loaded {len(df)} ratings from {file_path.name}.")
    return df


def main():
    """Main function to run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest and parse raw Netflix Prize dataset files.")
    parser.add_index = False  # Avoid index column in parquet if not needed
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Path to the directory containing raw Netflix files."
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Path to the directory to save processed Parquet files."
    )
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # 1. Parse movie titles
    movie_titles_path = raw_dir / "movie_titles.csv"
    if movie_titles_path.exists():
        movies_df = parse_movie_titles(movie_titles_path)
        movies_output_path = processed_dir / "movies.parquet"
        movies_df.to_parquet(movies_output_path, index=False)
        print(f"Saved movie titles metadata to {movies_output_path}")
    else:
        print(f"Warning: {movie_titles_path} not found. Skipping movie titles parsing.")
        
    # 2. Parse ratings files
    ratings_files = sorted(glob.glob(str(raw_dir / "combined_data_*.txt")))
    
    if not ratings_files:
        print(f"Warning: No combined_data_*.txt files found in {raw_dir}.")
        print("Please place the Netflix Prize dataset files in data/raw/ before running this script.")
        return
        
    ratings_dfs = []
    for filepath in ratings_files:
        file_df = parse_ratings_file(Path(filepath))
        ratings_dfs.append(file_df)
        
    if ratings_dfs:
        print("Concatenating all ratings into a single DataFrame...")
        ratings_df = pd.concat(ratings_dfs, ignore_index=True)
        ratings_output_path = processed_dir / "ratings.parquet"
        
        print(f"Saving combined ratings ({len(ratings_df)} rows) to Parquet...")
        ratings_df.to_parquet(ratings_output_path, index=False)
        print(f"Saved combined ratings to {ratings_output_path}")
        
    elapsed_time = time.time() - start_time
    print(f"Data ingestion pipeline completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
